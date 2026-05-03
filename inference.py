import rasterio
from rasterio.windows import Window
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── config ─────────────────────────────────────────────
MODEL_PATH = "final_model/resunet_scripted.pt"
INPUT_TIF  = r"E:\Drone_tech_lab_assignment\project_change_detection_v2\LULC_change_detection_DL\datasets\annotations\image\Phase1_processed.tif"
OUTPUT_TIF = "prediction.tif"

PATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

# ── load model ─────────────────────────────────────────
model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# ── normalization ──────────────────────────────────────
def normalize(img):
    img = img / 255.0
    img = (img - MEAN) / STD
    return img

# ── inference with tiling ──────────────────────────────



   
with rasterio.open(INPUT_TIF, masked=True) as src:

    profile = src.profile
    height, width = src.height, src.width
    profile.update(count=1, dtype=rasterio.uint8)

    with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:

        # calculate total tiles
        total_tiles = ((height + PATCH_SIZE - 1) // PATCH_SIZE) * \
                    ((width  + PATCH_SIZE - 1) // PATCH_SIZE)

        with tqdm(total=total_tiles, desc="Inference", unit="tile") as pbar:
            for y in range(0, height, PATCH_SIZE):
                for x in range(0, width, PATCH_SIZE):

                    h = min(PATCH_SIZE, height - y)
                    w = min(PATCH_SIZE, width - x)
                    window = Window(x, y, w, h)

                    # ── read & normalize ───────────────
                    patch = src.read([1, 2, 3], window=window)          # (3, h, w)
                    valid_mask = np.any(src.read_masks(window=window), axis=0).astype(np.uint8)  # (h, w)
                    patch = np.transpose(patch, (1, 2, 0))               # (h, w, 3)
                    patch = normalize(patch.astype(np.float32))

                    # ── to tensor ──────────────────────
                    tensor = torch.from_numpy(patch).float()
                    tensor = tensor.permute(2, 0, 1).unsqueeze(0)        # (1, 3, h, w)

                    # ── pad to PATCH_SIZE if edge tile ──
                    pad_h = PATCH_SIZE - h
                    pad_w = PATCH_SIZE - w
                    if pad_h > 0 or pad_w > 0:
                        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

                    tensor = tensor.to(DEVICE)

                    # ── predict ────────────────────────
                    with torch.no_grad():
                        logits = model(tensor)
                        pred = torch.argmax(logits, dim=1)               # (1, PATCH_SIZE, PATCH_SIZE)

                    # ── crop → numpy → mask nodata ─────
                    pred = pred[0, :h, :w].cpu().numpy().astype(np.uint8)
                    pred *= valid_mask                                    # zero-out nodata pixels

                    dst.write(pred, 1, window=window)
                    pbar.update(1)

print("Inference complete → prediction.tif")