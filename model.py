"""
model.py  -  ResU-Net with pretrained ResNet-34/50 encoder
===========================================================

Architecture
------------
  Encoder  : ResNet-34 (default) or ResNet-50, ImageNet-pretrained via torchvision.
             Lateral feature maps extracted at 4 scales:
               enc1 →  64 ch  (after stem+pool,  stride 4)
               enc2 →  64 ch  (after layer1,      stride 4)
               enc3 → 128 ch  (after layer2,      stride 8)
               enc4 → 256 ch  (after layer3,      stride 16)
             Bottleneck feature from layer4 (stride 32) fed to ASPP.

  Bottleneck: Atrous Spatial Pyramid Pooling (ASPP) - captures multi-scale
              context; especially important for drone imagery where objects
              span very different pixel footprints.

  Decoder  : 4 upsampling stages, each with:
               - Attention Gate (suppresses irrelevant skip-connection activations)
               - Concatenate skip + upsampled feature
               - Two X (Conv-BN-ReLU) residual blocks

  Head     : 1x1 conv → num_classes logits.

Residual blocks in the decoder mean gradients flow cleanly even with many layers,
which matters when fine-tuning the encoder end-to-end.

Usage
-----
    from model import ResUNet
    model = ResUNet(num_classes=5, backbone="resnet34", pretrained=True)
    logits = model(x)          # x: (B, 3, H, W),  logits: (B, num_classes, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights, ResNet50_Weights
from typing import List, Tuple, Literal

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → ReLU (inplace)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        padding: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch, kernel, padding=padding, dilation=dilation, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Two ConvBnRelu layers with a residual (identity / projection) shortcut.
    Used in the decoder to stabilise training depth.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBnRelu(in_ch, out_ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(out + identity)


class AttentionGate(nn.Module):
    """
    Soft attention gate (Oktay et al., 2018).
    Scales skip-connection features by a learned spatial attention map
    derived from the gating signal (upsampled decoder feature).

    Parameters
    ----------
    F_g   : channels in gating signal
    F_l   : channels in skip connection
    F_int : intermediate channels (typically F_l // 2)
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # g: gating signal  (lower-res decoder feature, already up-sampled to x's size)
        # x: skip connection feature
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    Captures context at 4 receptive-field scales plus a global pooling branch.
    Ideal for drone imagery where buildings, roads, and vegetation exist at
    very different spatial scales within the same tile.

    Input  : (B, in_ch, H, W)
    Output : (B, out_ch, H, W)   default out_ch = 256
    """

    def __init__(
        self, in_ch: int, out_ch: int = 256, dilations: Tuple[int, ...] = (1, 6, 12, 18)
    ):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(
                ConvBnRelu(
                    in_ch,
                    out_ch,
                    kernel=1 if d == 1 else 3,
                    padding=0 if d == 1 else d,
                    dilation=d,
                )
            )
        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        total_ch = out_ch * (len(dilations) + 1)
        self.project = ConvBnRelu(total_ch, out_ch, kernel=1, padding=0)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        features = [branch(x) for branch in self.branches]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=(h, w), mode="bilinear", align_corners=False)
        features.append(gp)
        out = torch.cat(features, dim=1)
        return self.dropout(self.project(out))


class DecoderBlock(nn.Module):
    """
    One decoder stage:
      1. Bilinear upsample the input feature map ×2
      2. Attention-gate the skip connection
      3. Concatenate → two residual conv blocks
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.attention = AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=skip_ch // 2)
        self.res_block = ResidualBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Align spatial dims (handles odd input sizes gracefully)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        skip = self.attention(g=x, x=skip)
        return self.res_block(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# Encoder wrappers
# ---------------------------------------------------------------------------


class ResNet34Encoder(nn.Module):
    """Extract 5 feature maps from a pretrained ResNet-34."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet34(weights=weights)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu)  # /2
        self.pool = base.maxpool  # /4
        self.layer1 = base.layer1  # 64  ch,  /4
        self.layer2 = base.layer2  # 128 ch,  /8
        self.layer3 = base.layer3  # 256 ch,  /16
        self.layer4 = base.layer4  # 512 ch,  /32

        # Channel counts for decoder wiring
        self.out_channels = [64, 64, 128, 256, 512]  # stem,l1,l2,l3,l4

    def forward(self, x):
        e0 = self.stem(x)  # (B,  64, H/2,  W/2)
        e1 = self.layer1(self.pool(e0))  # (B,  64, H/4,  W/4)
        e2 = self.layer2(e1)  # (B, 128, H/8,  W/8)
        e3 = self.layer3(e2)  # (B, 256, H/16, W/16)
        e4 = self.layer4(e3)  # (B, 512, H/32, W/32)
        return e0, e1, e2, e3, e4


class ResNet50Encoder(nn.Module):
    """Extract 5 feature maps from a pretrained ResNet-50."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = models.resnet50(weights=weights)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool = base.maxpool
        self.layer1 = base.layer1  # 256  ch (bottleneck),  /4
        self.layer2 = base.layer2  # 512  ch,  /8
        self.layer3 = base.layer3  # 1024 ch,  /16
        self.layer4 = base.layer4  # 2048 ch,  /32

        self.out_channels = [64, 256, 512, 1024, 2048]

    def forward(self, x):
        e0 = self.stem(x)
        e1 = self.layer1(self.pool(e0))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        return e0, e1, e2, e3, e4


# ---------------------------------------------------------------------------
# Full ResU-Net
# ---------------------------------------------------------------------------


class ResUNet(nn.Module):
    """
    ResU-Net for multiclass semantic segmentation of drone orthomosaics.

    Parameters
    ----------
    num_classes : int
        Number of output classes (e.g. 5: background, building, road,
        waterbody, vegetation).
    backbone    : "resnet34" | "resnet50"
    pretrained  : bool  – use ImageNet weights for encoder
    decoder_ch  : list of 4 ints – output channels per decoder stage
    freeze_encoder : bool – freeze encoder weights for the first N epochs
                            (call model.unfreeze_encoder() when ready)
    """

    def __init__(
        self,
        num_classes: int = 5,
        backbone: Literal["resnet34", "resnet50"] = "resnet34",
        pretrained: bool = True,
        decoder_ch: List[int] = None,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        if decoder_ch is None:
            decoder_ch = [256, 128, 64, 32]

        # --- Encoder ---
        if backbone == "resnet34":
            self.encoder = ResNet34Encoder(pretrained)
        elif backbone == "resnet50":
            self.encoder = ResNet50Encoder(pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        enc_ch = self.encoder.out_channels  # [e0, e1, e2, e3, e4]

        # --- Bottleneck (ASPP on deepest encoder feature) ---
        self.bottleneck = ASPPModule(enc_ch[4], out_ch=decoder_ch[0])

        # --- Decoder stages (deep → shallow) ---
        # dec1: bottleneck + e3 skip
        self.dec1 = DecoderBlock(decoder_ch[0], enc_ch[3], decoder_ch[1])
        # dec2: dec1_out + e2 skip
        self.dec2 = DecoderBlock(decoder_ch[1], enc_ch[2], decoder_ch[2])
        # dec3: dec2_out + e1 skip
        self.dec3 = DecoderBlock(decoder_ch[2], enc_ch[1], decoder_ch[3])
        # dec4: dec3_out + e0 skip  (restores H/2)
        self.dec4 = DecoderBlock(decoder_ch[3], enc_ch[0], decoder_ch[3])

        # Final upsample ×2 to restore full H×W, then classify
        self.final_up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.head = nn.Sequential(
            ConvBnRelu(decoder_ch[3], decoder_ch[3]),
            nn.Conv2d(decoder_ch[3], num_classes, 1),
        )

        if freeze_encoder:
            self.freeze_encoder()

        self._init_decoder_weights()

    # ------------------------------------------------------------------
    # Weight initialisation for decoder (encoder keeps pretrained weights)
    # ------------------------------------------------------------------
    def _init_decoder_weights(self):
        for module in [
            self.bottleneck,
            self.dec1,
            self.dec2,
            self.dec3,
            self.dec4,
            self.head,
        ]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Encoder freeze / unfreeze helpers
    # ------------------------------------------------------------------
    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self, lr_scale: float = 0.1):
        """
        Unfreeze encoder. Returns parameter groups suitable for optimiser:
          [{'params': encoder_params, 'lr': base_lr * lr_scale},
           {'params': decoder_params, 'lr': base_lr}]
        Call with optimizer param_groups update after unfreezing.
        """
        for p in self.encoder.parameters():
            p.requires_grad = True

    def get_param_groups(self, base_lr: float):
        """Differential learning rates: encoder gets 0.1× of decoder LR."""
        encoder_ids = {id(p) for p in self.encoder.parameters()}
        encoder_params = [p for p in self.parameters() if id(p) in encoder_ids]
        decoder_params = [p for p in self.parameters() if id(p) not in encoder_ids]
        return [
            {"params": encoder_params, "lr": base_lr * 0.1},
            {"params": decoder_params, "lr": base_lr},
        ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W)  float32, normalised to [0,1] or ImageNet stats

        Returns
        -------
        logits : (B, num_classes, H, W)
        """
        e0, e1, e2, e3, e4 = self.encoder(x)

        b = self.bottleneck(e4)  # (B, 256, H/32, W/32)

        d1 = self.dec1(b, e3)  # (B, 128, H/16, W/16)
        d2 = self.dec2(d1, e2)  # (B,  64, H/8,  W/8)
        d3 = self.dec3(d2, e1)  # (B,  32, H/4,  W/4)
        d4 = self.dec4(d3, e0)  # (B,  32, H/2,  W/2)

        out = self.final_up(d4)  # (B,  32, H,    W)
        return self.head(out)  # (B, num_classes, H, W)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResUNet(num_classes=5, backbone="resnet34", pretrained=False).to(device)
    x = torch.randn(2, 3, 512, 512, device=device)
    out = model(x)
    assert out.shape == (2, 5, 512, 512), f"Unexpected shape: {out.shape}"
    print("model.py  OK  -  output shape:", out.shape)
    print("Parameters:", model.count_parameters())
