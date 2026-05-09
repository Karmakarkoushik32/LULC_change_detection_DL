import json
import numpy as np


def compute_change_metrics(
    segmented_1: np.ndarray,
    segmented_2: np.ndarray,
    class_mappings: dict[int, str],
    pixel_size_m: float = 0.1,
    output_json_path: str | None = None,
) -> dict:
    """
    Compute per-class change metrics between two segmentation arrays.

    Parameters
    ----------
    segmented_1      : np.ndarray (H, W) uint8 - segmentation at time T1
    segmented_2      : np.ndarray (H, W) uint8 - segmentation at time T2
    class_mappings   : dict mapping class id (int) -> class name (str)
                       class 0 is always ignored
    pixel_size_m     : ground resolution in metres (default 0.1 m → 10 cm)
    output_json_path : if provided, results are written to this .json file

    Returns
    -------
    dict with keys:
        class_mappings    - {class_id: class_name, ...}
        transition_matrix - {from_id: {to_id: area_m2, ...}, ...}
        bin_count         - {class_id: {before: area_m2, after: area_m2}, ...}
        change_percent    - {class_id: change_%}
    """
    pixel_area = pixel_size_m**2  # m² per pixel
    class_ids = [cid for cid in class_mappings if cid != 0]

    #  1. bin counts (area per class, before & after)
    bc1 = np.bincount(segmented_1.ravel())
    bc2 = np.bincount(segmented_2.ravel())

    def _area(bincount: np.ndarray, cid: int) -> float:
        return float(bincount[cid]) * pixel_area if cid < len(bincount) else 0.0

    bin_count = {
        cid: {
            "before": _area(bc1, cid),
            "after": _area(bc2, cid),
        }
        for cid in class_ids
    }

    #  2. transition matrix
    # count co-occurrences of (T1 class, T2 class) at every pixel
    max_id = max(class_ids) + 1
    matrix = np.zeros((max_id, max_id), dtype=np.int64)

    valid = (segmented_1 != 0) & (segmented_2 != 0)  # ignore class-0 pixels
    np.add.at(matrix, (segmented_1[valid], segmented_2[valid]), 1)

    transition_matrix = {
        from_id: {
            to_id: float(matrix[from_id, to_id]) * pixel_area for to_id in class_ids
        }
        for from_id in class_ids
    }

    #  3. per-class change percentage
    change_percent = {}
    for cid in class_ids:
        before = bin_count[cid]["before"]
        after = bin_count[cid]["after"]
        if before > 0:
            change_percent[cid] = round((after - before) / before * 100, 4)
        else:
            change_percent[cid] = None  # class absent in T1

    #  assemble output
    result = {
        "class_mappings": {cid: class_mappings[cid] for cid in class_ids},
        "transition_matrix": transition_matrix,
        "bin_count": bin_count,
        "change_percent": change_percent,
    }

    if output_json_path:
        with open(output_json_path, "w") as f:
            json.dump(result, f, indent=2)

    return result
