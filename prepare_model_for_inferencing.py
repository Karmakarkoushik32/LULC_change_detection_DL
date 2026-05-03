import torch
from model import ResUNet  # make sure import path is correct

# ── config (same as training) ─────────────────────────────
DEVICE = torch.device("cpu")  # export on CPU for portability
CHECKPOINT_PATH = "checkpoints/best.pt"
EXPORT_PATH = "final_model/resunet_scripted.pt"

NUM_CLASSES = 5
BACKBONE = "resnet34"
PRETRAINED = False  # important: not needed when loading weights

# ── load model ────────────────────────────────────────────
model = ResUNet(
    num_classes=NUM_CLASSES,
    backbone=BACKBONE,
    pretrained=PRETRAINED
)

ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])

model.to(DEVICE)
model.eval()

print("Model loaded")

# ── dummy input (important for tracing) ───────────────────
# Adjust size based on your training
dummy_input = torch.randn(1, 3, 512, 512)

# ── convert to TorchScript ────────────────────────────────
with torch.no_grad():
    scripted_model = torch.jit.trace(model, dummy_input)

# ── save ─────────────────────────────────────────────────
scripted_model.save(EXPORT_PATH)

print(f"TorchScript model saved at: {EXPORT_PATH}")