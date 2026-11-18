"""Training-free control used in the qualitative figure."""
from __future__ import annotations

import torch
from torch import Tensor


def mean_support_mask(ys: Tensor, n_query: int) -> Tensor:
    """Pixelwise mean of the support masks, thresholded at 1/2 (uses no image information).
    ys: (n_s, 1, H, W) -> probabilities (n_query, 1, H, W)."""
    return ys.float().mean(0, keepdim=True).expand(n_query, -1, -1, -1).clone()
