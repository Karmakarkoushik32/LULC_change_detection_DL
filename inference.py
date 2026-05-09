import numpy as np
import torch
import torch.nn.functional as F
from rasterio.windows import Window
from tqdm import tqdm
from utils import normalize
import rasterio


class Inferencer:
    model = None

    def __init__(
        self, model_path: str, patch_size: int = 512, device: torch.device | None = None
    ):
        self.patch_size = patch_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        Inferencer.model = torch.jit.load(model_path, map_location=self.device)
        Inferencer.model.eval()

    def predict(
        self,
        image_path: str,
        segmented_image_path: str,
    ) -> np.ndarray:
        """
        Run tiled inference on a GeoTIFF and write the segmentation result.

        Parameters
        ----------
        image_path           : path to the input multi-band GeoTIFF
        segmented_image_path : path where the predicted single-band uint8
                               GeoTIFF is saved
        """
        patch_size = self.patch_size

        with rasterio.open(image_path, masked=True) as src:
            profile = src.profile.copy()
            height, width = src.height, src.width
            profile.update(count=1, dtype=rasterio.uint8)

            total_tiles = ((height + patch_size - 1) // patch_size) * (
                (width + patch_size - 1) // patch_size
            )

            with rasterio.open(segmented_image_path, "w", **profile) as dst:
                with tqdm(total=total_tiles, desc="Inference", unit="tile") as pbar:
                    for y in range(0, height, patch_size):
                        for x in range(0, width, patch_size):
                            h = min(patch_size, height - y)
                            w = min(patch_size, width - x)
                            window = Window(x, y, w, h)

                            #  read & normalize
                            patch = src.read([1, 2, 3], window=window)  # (3, h, w)
                            valid_mask = np.any(
                                src.read_masks(window=window), axis=0
                            ).astype(
                                np.uint8
                            )  # (h, w)
                            patch = np.transpose(patch, (1, 2, 0))  # (h, w, 3)
                            patch = normalize(patch.astype(np.float32))

                            #  to tensor ─
                            tensor = torch.from_numpy(patch).float()
                            tensor = tensor.permute(2, 0, 1).unsqueeze(
                                0
                            )  # (1, 3, h, w)

                            #  pad edge tiles to full patch_size
                            pad_h = patch_size - h
                            pad_w = patch_size - w
                            if pad_h > 0 or pad_w > 0:
                                tensor = F.pad(
                                    tensor, (0, pad_w, 0, pad_h), mode="reflect"
                                )

                            tensor = tensor.to(self.device)

                            #  forward pass
                            with torch.no_grad():
                                logits = Inferencer.model(tensor)
                                pred = torch.argmax(logits, dim=1)  # (1, PS, PS)

                            #  crop → numpy → mask nodata
                            pred = pred[0, :h, :w].cpu().numpy().astype(np.uint8)
                            pred *= valid_mask  # zero-out nodata pixels

                            dst.write(pred, 1, window=window)
                            pbar.update(1)
