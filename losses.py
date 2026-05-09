import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ---------------------------------------------------------------------------
# Focal Loss (Corrected)
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C = inputs.shape[:2]

        log_probs = F.log_softmax(inputs, dim=1)
        probs = log_probs.exp()

        targets = targets.long()
        valid_mask = targets != self.ignore_index

        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0

        log_pt = log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - pt) ** self.gamma
        loss = -self.alpha * focal_weight * log_pt

        # class weights
        if self.weight is not None:
            w = self.weight[safe_targets]
            loss = loss * w

        # label smoothing (correct CE-style smoothing)
        if self.label_smoothing > 0:
            smooth_loss = -log_probs.mean(dim=1)
            loss = (
                1 - self.label_smoothing
            ) * loss + self.label_smoothing * smooth_loss

        # apply ignore mask
        loss = loss * valid_mask

        if self.reduction == "mean":
            return loss.sum() / valid_mask.sum().clamp_min(1)
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


# ---------------------------------------------------------------------------
# Dice Loss (Corrected)
# ---------------------------------------------------------------------------


class DiceLoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1e-6,
        per_image: bool = False,
        ignore_index: int = -100,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.smooth = smooth
        self.per_image = per_image
        self.ignore_index = ignore_index
        self.register_buffer(
            "class_weights", class_weights if class_weights is not None else None
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = preds.shape
        prob = F.softmax(preds, dim=1)

        targets = targets.long()

        valid_mask = targets != self.ignore_index
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0

        targets_oh = F.one_hot(safe_targets, num_classes=C).permute(0, 3, 1, 2).float()

        mask = valid_mask.unsqueeze(1)
        prob = prob * mask
        targets_oh = targets_oh * mask

        if self.per_image:
            inter = (prob * targets_oh).sum(dim=(2, 3))
            denom = prob.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3))
            dice = (2 * inter + self.smooth) / (denom + self.smooth)
        else:
            inter = (prob * targets_oh).sum(dim=(0, 2, 3))
            denom = prob.sum(dim=(0, 2, 3)) + targets_oh.sum(dim=(0, 2, 3))
            dice = (2 * inter + self.smooth) / (denom + self.smooth)

        if self.class_weights is not None:
            dice = dice * self.class_weights

        return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# Combo Loss (Corrected)
# ---------------------------------------------------------------------------


class ComboLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ):
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")

        self.alpha = alpha

        self.focal = FocalLoss(
            alpha=1.0,  # scalar only
            gamma=focal_gamma,
            weight=class_weights,
            ignore_index=ignore_index,
        )

        self.dice = DiceLoss(
            class_weights=class_weights,
            ignore_index=ignore_index,
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        focal_val = self.focal(preds, targets)
        dice_val = self.dice(preds, targets)

        total = self.alpha * focal_val + (1 - self.alpha) * dice_val

        return total, {
            "focal": float(focal_val.detach()),
            "dice": float(dice_val.detach()),
            "total": float(total.detach()),
        }


# ---------------------------------------------------------------------------
# Class Weights (unchanged, but good)
# ---------------------------------------------------------------------------


def compute_class_weights(
    pixel_counts: torch.Tensor,
    num_classes: int,
    method: str = "inv_freq",
) -> torch.Tensor:

    counts = pixel_counts.float().clamp(min=1)
    total = counts.sum()
    freq = counts / total

    if method == "inv_freq":
        w = 1.0 / (num_classes * freq)
    elif method == "sqrt_inv":
        w = 1.0 / torch.sqrt(freq)
    elif method == "median_freq":
        med = torch.median(freq)
        w = med / freq
    else:
        raise ValueError(f"Unknown method: {method}")

    return (w / w.sum() * num_classes).float()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, C, H, W = 2, 5, 64, 64

    preds = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))

    loss_fn = ComboLoss(alpha=0.5, ignore_index=0)

    total, info = loss_fn(preds, targets)

    print(info)
