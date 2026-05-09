import torch
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _tp_fp_fn(cm: torch.Tensor):
    cm = cm.float()
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    return tp, fp, fn


def iou(cm, class_map, ignore_index=None):
    tp, fp, fn = _tp_fp_fn(cm)

    denom = tp + fp + fn
    iou_vals = torch.where(denom > 0, tp / denom.clamp(min=1e-8), torch.zeros_like(tp))

    present = cm.sum(dim=1) > 0

    if ignore_index is not None and ignore_index < len(present):
        present[ignore_index] = False

    miou = iou_vals[present].mean().item() if present.any() else 0.0

    per_class = {class_map[i]: round(iou_vals[i].item(), 4) for i in class_map}

    return miou, per_class


def commission_error(cm, class_map):
    tp, fp, _ = _tp_fp_fn(cm)

    denom = tp + fp
    comm = torch.where(denom > 0, fp / denom.clamp(min=1e-8), torch.zeros_like(tp))

    return {class_map[i]: round(comm[i].item(), 4) for i in class_map}


def omission_error(cm, class_map):
    tp, _, fn = _tp_fp_fn(cm)

    denom = tp + fn
    omiss = torch.where(denom > 0, fn / denom.clamp(min=1e-8), torch.zeros_like(tp))

    return {class_map[i]: round(omiss[i].item(), 4) for i in class_map}


def pixel_accuracy(cm):
    cm = cm.float()
    tp = torch.diag(cm)
    return (tp.sum() / cm.sum().clamp(min=1)).item()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SegmentationMetrics:
    def __init__(
        self,
        class_map: dict[int, str],
        ignore_index: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        class_map    : {int: str}
        ignore_index : int or None
        """
        self.class_map = dict(sorted(class_map.items()))
        self.ignore_index = ignore_index

        self.num_classes = max(self.class_map.keys()) + 1

        # safety check
        expected = set(range(self.num_classes))
        if set(self.class_map.keys()) != expected:
            raise ValueError("class_map must be continuous from 0 to N-1")

        self._conf_matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long
        )

    # ------------------------------------------------------------------
    def reset(self):
        self._conf_matrix.zero_()

    # ------------------------------------------------------------------
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)

        preds = preds.cpu().long().flatten()
        targets = targets.cpu().long().flatten()

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            preds = preds[mask]
            targets = targets[mask]

        preds = preds.clamp(0, self.num_classes - 1)
        targets = targets.clamp(0, self.num_classes - 1)

        combined = self.num_classes * targets + preds
        conf = torch.bincount(combined, minlength=self.num_classes**2)

        self._conf_matrix += conf.reshape(self.num_classes, self.num_classes)

    # ------------------------------------------------------------------
    def compute(self):
        cm = self._conf_matrix

        miou, iou_pc = iou(cm, self.class_map, self.ignore_index)
        commission_pc = commission_error(cm, self.class_map)
        omission_pc = omission_error(cm, self.class_map)
        acc = pixel_accuracy(cm)

        per_class = {}

        for idx, name in self.class_map.items():
            if self.ignore_index is not None and idx == self.ignore_index:
                continue

            per_class[name] = {
                "iou": iou_pc[name],
                "commission_error": commission_pc[name],
                "omission_error": omission_pc[name],
                "support": int(cm[idx].sum().item()),
            }

        return {
            "mIoU": round(miou, 4),
            "pixel_accuracy": round(acc, 4),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        }

    # ------------------------------------------------------------------
    def confusion_matrix(self) -> np.ndarray:
        return self._conf_matrix.numpy()

    # ------------------------------------------------------------------
    def log_status(self):
        r = self.compute()

        lines = []
        lines.append(f"\n{'-'*50}")
        lines.append(f"  mIoU            : {r['mIoU']:.4f}")
        lines.append(f"  Pixel accuracy  : {r['pixel_accuracy']:.4f}")

        lines.append(f"\n  Per-class breakdown:")
        lines.append(f"  {'Class':<18} {'IoU':>7} {'Comm':>7} {'Omiss':>7}")
        lines.append(f"  {'-'*50}")

        for name, m in r["per_class"].items():
            lines.append(
                f"  {name:<18} "
                f"{m['iou']:>7.4f} "
                f"{m['commission_error']:>7.4f} "
                f"{m['omission_error']:>7.4f}"
            )

        lines.append(f"{'-'*50}\n")

        print("\n".join(lines))

    # ------------------------------------------------------------------
    def confusion_matrix(self) -> np.ndarray:
        """Return raw (C, C) numpy confusion matrix."""
        return self._conf_matrix.numpy()


# ---------------------------------------------------------------------------
# Lightweight single-batch functions (for quick eval without accumulation)
# ---------------------------------------------------------------------------


def batch_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Returns per-class IoU for a single batch. Useful for fast sanity checks
    during training (not for final evaluation - use SegmentationMetrics).

    preds   : (B, C, H, W) logits or (B, H, W) argmax
    targets : (B, H, W) long
    """
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)

    preds = preds.flatten().long()
    targets = targets.flatten().long()

    if ignore_index >= 0:
        mask = targets != ignore_index
        preds = preds[mask]
        targets = targets[mask]

    ious = []
    for c in range(num_classes):
        pred_c = preds == c
        target_c = targets == c
        inter = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        ious.append(
            (inter / union.clamp(min=1e-8)).item() if union > 0 else float("nan")
        )
    return torch.tensor(ious)


def batch_miou(preds, targets, num_classes, ignore_index=-100) -> float:
    """Convenience: mean IoU over non-nan classes for a single batch."""
    iou = batch_iou(preds, targets, num_classes, ignore_index)
    valid = ~torch.isnan(iou)
    return iou[valid].mean().item() if valid.any() else 0.0


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    class_map = {
        0: "background",
        1: "building",
        2: "road",
        3: "waterbody",
        4: "vegetation",
    }

    C = len(class_map)
    meter = SegmentationMetrics(class_map=class_map, ignore_index=0)

    for _ in range(4):
        preds = torch.randn(2, C, 128, 128)
        targets = torch.randint(0, C, (2, 128, 128))
        meter.update(preds, targets)

    meter.log_status()

    preds = torch.randn(2, C, 64, 64)
    targets = torch.randint(0, C, (2, 64, 64))
    print("Batch mIoU:", round(batch_miou(preds, targets, C), 4))
