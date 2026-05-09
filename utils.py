import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from affine import Affine
from config import TARGET_RES, IMAGENET_NORM_MEAN, IMAGENET_NORM_STD
from geopandas import GeoDataFrame
from shapely.geometry import shape


def normalize(img):
    img = img / 255.0
    img = (img - IMAGENET_NORM_MEAN) / IMAGENET_NORM_STD
    return img


def denormalize(img, mean=None, std=None):
    """
    img: (C, H, W)
    """
    if mean is not None and std is not None:
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
        img = img * std + mean
    return img


def plot_batch(loader, class_map=None, mean=None, std=None, n=4):
    imgs, masks = next(iter(loader))  # one batch

    imgs = imgs[:n]
    masks = masks[:n]

    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))

    if n == 1:
        axes = [axes]

    for i in range(n):
        img = imgs[i].cpu()
        mask = masks[i].cpu()

        # denormalize if needed
        img = denormalize(img, mean, std)

        # convert to HWC
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        # plot image
        axes[i][0].imshow(img)
        axes[i][0].set_title("Image")
        axes[i][0].axis("off")

        # plot mask
        axes[i][1].imshow(mask.numpy(), cmap="tab20")
        axes[i][1].set_title("Mask")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.show()


def is_geographic(crs):
    return crs.is_geographic


def choose_target_crs(src1, src2):
    crs1, crs2 = src1.crs, src2.crs

    if is_geographic(crs1) and is_geographic(crs2):
        return "EPSG:3857"

    if not is_geographic(crs1) and is_geographic(crs2):
        return crs1

    if is_geographic(crs1) and not is_geographic(crs2):
        return crs2

    # both projected but maybe different
    return crs1


def reproject_to_target(src, target_crs, bounds):
    transform, width, height = calculate_default_transform(
        src.crs, target_crs, src.width, src.height, *bounds, resolution=TARGET_RES
    )

    band_count = min(3, src.count)
    bands = list(range(1, band_count + 1))

    data = np.zeros((band_count, height, width), dtype=np.uint8)

    for i, b in enumerate(bands):
        reproject(
            source=rasterio.band(src, b),
            destination=data[i],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
        )

    return data, transform


def write_data(
    filaneme: str, data: np.ndarray, transform: Affine, crs: rasterio.crs.CRS
):
    with rasterio.open(
        filaneme,
        mode="w",
        **{
            "count": data.shape[0],
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": transform,
            "nodata": 0,
            "crs": crs,
            "dtype": data.dtype,
        }
    ) as dst:
        dst.write(data)


def prepare_image_pair(tif1_path, tif2_path):
    with rasterio.open(tif1_path, masked=True) as src1, rasterio.open(
        tif2_path, masked=True
    ) as src2:

        print("=== INPUT CRS ===")
        print(src1.crs, src2.crs)

        # 1. Choose CRS
        target_crs = choose_target_crs(src1, src2)
        print("\n=== TARGET CRS ===")
        print(target_crs)

        # 2. Compute intersection in original CRS (approx safe)
        b1, b2 = src1.bounds, src2.bounds

        left = max(b1.left, b2.left)
        right = min(b1.right, b2.right)
        bottom = max(b1.bottom, b2.bottom)
        top = min(b1.top, b2.top)

        if left >= right or bottom >= top:
            raise ValueError("No overlap!")

        common_bounds = (left, bottom, right, top)

        # 3. Reproject both to target CRS + 10cm
        data1, transform1 = reproject_to_target(src1, target_crs, common_bounds)
        data2, transform2 = reproject_to_target(src2, target_crs, common_bounds)

        # 4. Ensure same shape (important!)
        min_h = min(data1.shape[1], data2.shape[1])
        min_w = min(data1.shape[2], data2.shape[2])

        data1 = data1[:, :min_h, :min_w]
        data2 = data2[:, :min_h, :min_w]

        print("\n=== OUTPUT ===")
        print("data1 Shape:", data1.shape)
        print("data2 Shape:", data2.shape)
        print("data1 transform:", transform1)
        print("data2 transform:", transform2)

        return data1, data2, transform1, transform2, target_crs


def polygonize(
    segmented: np.ndarray,
    class_mappings: dict[int, str],
    transform: Affine,
    crs: rasterio.crs.CRS,
    skip_classes: list[int],
) -> GeoDataFrame:
    """
    Convert a segmentation raster to GeoJSON-style polygon features.

    Parameters
    ----------
    segmented : np.ndarray  (H, W) uint8
    transform : rasterio.Affine
    crs       : rasterio CRS

    Returns
    -------
    Geodataframe
    """
    from rasterio.features import shapes

    featuresdf = GeoDataFrame(
        [
            {
                "geometry": shape(geom),
                "class_id": value,
                "clann_name": class_mappings.get(value),
            }
            for geom, value in shapes(segmented, transform=transform)
            if value not in skip_classes
        ],
        crs=crs,
        geometry="geometry",
    )

    featuresdf["geometry"] = (
        featuresdf.geometry
        .buffer(0)
        .simplify(0.2, preserve_topology=True)
    )
    featuresdf = featuresdf.to_crs('EPSG:4326')

    return featuresdf
