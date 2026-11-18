"""Episodic two-parameter logistic calibration (paper Sec. III-C).

(gamma, beta) = argmin  sum omega * BCE(sigmoid(gamma z + beta), y) + eta_c (gamma^2 + beta^2)

Strongly convex for eta_c > 0; solved by damped Newton from the cold start (1, 0).
The result is detached: meta-gradients flow through z only (TD Sec. 6.8).
"""
from __future__ import annotations

import torch
from torch import Tensor


def _objective(g: Tensor, z: Tensor, y: Tensor, w: Tensor, ridge: float) -> Tensor:
    s = g[0] * z + g[1]
    # numerically stable BCE with logits
    bce = torch.clamp(s, min=0) - s * y + torch.log1p(torch.exp(-s.abs()))
    return (w * bce).sum() + ridge * (g ** 2).sum()


@torch.no_grad()
def fit_calibration(z: Tensor, y: Tensor, w: Tensor, ridge: float = 1e-2, steps: int = 10
                    ) -> tuple[float, float]:
    """z, y, w: any matching shapes (flattened internally). Returns python floats."""
    z = z.detach().double().flatten()
    y = y.detach().double().flatten()
    w = w.detach().double().flatten()
    g = torch.tensor([1.0, 0.0], dtype=torch.float64, device=z.device)
    f = _objective(g, z, y, w, ridge)
    for _ in range(steps):
        p = torch.sigmoid(g[0] * z + g[1])
        r = w * (p - y)
        grad = torch.stack([(r * z).sum(), r.sum()]) + 2 * ridge * g
        h = w * p * (1 - p)
        H = torch.stack([torch.stack([(h * z * z).sum(), (h * z).sum()]),
                         torch.stack([(h * z).sum(), h.sum()])])
        H = H + 2 * ridge * torch.eye(2, dtype=H.dtype, device=H.device)
        step = torch.linalg.solve(H, grad)
        t = 1.0
        while t > 1e-6:                      # backtracking (damped Newton)
            g_new = g - t * step
            f_new = _objective(g_new, z, y, w, ridge)
            if f_new <= f:
                break
            t *= 0.5
        if t <= 1e-6:
            break
        g, f = g_new, f_new
    return float(g[0]), float(g[1])
