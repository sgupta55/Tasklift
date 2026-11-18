"""Evaluation metrics (TD Sec. 11.4): Dice, IoU, ASSD on hard 256x256 masks; episode bootstrap."""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

_CROSS = ndimage.generate_binary_structure(2, 1)   # 4-connectivity


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def boundary(mask: np.ndarray) -> np.ndarray:
    """4-connected inner boundary: mask pixels with at least one 4-neighbour outside."""
    mask = mask.astype(bool)
    return mask & ~ndimage.binary_erosion(mask, structure=_CROSS, border_value=0)


def assd(pred: np.ndarray, gt: np.ndarray, empty_value: float | None = None) -> float:
    """Average symmetric surface distance in pixels.

    If exactly one mask is empty the distance is undefined; `empty_value`
    (default: the image diagonal) is returned and documented in the README."""
    bp, bg = boundary(pred), boundary(gt)
    if not bp.any() and not bg.any():
        return 0.0
    if not bp.any() or not bg.any():
        return float(empty_value if empty_value is not None else math.hypot(*pred.shape))
    dt_g = ndimage.distance_transform_edt(~bg)
    dt_p = ndimage.distance_transform_edt(~bp)
    return float((dt_g[bp].sum() + dt_p[bg].sum()) / (bp.sum() + bg.sum()))


def bootstrap_ci(values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
                 ) -> tuple[float, float, float, float]:
    """Mean, percentile CI (lo, hi) and half-width from an episode bootstrap."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi), float((hi - lo) / 2)
