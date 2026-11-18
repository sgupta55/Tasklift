"""Closed-form episodic compilation (paper Sec. III-B, Eq. (3), Theorem 1).

    w = argmin_w  sum_rows omega (phi^T w - t)^2 + lambda ||w - mu||^2     (Gamma = I)
      = A^{-1} b,  A = T~^T T~ + lambda I,  b = T~^T t~ + lambda mu.

Gradients flow through the Cholesky solve exactly (PyTorch differentiates
cholesky / cholesky_solve analytically), which is Theorem 1(b).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .features import psi_rows


def class_balance_weights(t: Tensor, p_hat: Tensor) -> Tensor:
    """omega = 1{t=1}/(2 p) + 1{t=0}/(2 (1-p)); t: (B,1,h,w), p_hat: (B,) clipped fractions."""
    p = p_hat.view(-1, 1, 1, 1)
    return t / (2 * p) + (1 - t) / (2 * (1 - p))


def positive_fraction(y: Tensor, eps: float) -> Tensor:
    """Per-image positive fraction of full-resolution masks y: (B,1,H,W), clipped to [eps, 1-eps]."""
    return y.flatten(1).mean(1).clamp(eps, 1 - eps)


def _gram_part(v: Tensor, ell: Tensor | None, t: Tensor, om: Tensor, degree: int
               ) -> tuple[Tensor, Tensor]:
    phi = psi_rows(v, degree)
    if ell is not None:
        phi = torch.cat([phi, ell], dim=-1)
    phi_w = phi * om.unsqueeze(-1)
    return phi_w.T @ phi, phi_w.T @ t


def accumulate_normal_equations(v: Tensor, ell: Tensor | None, t: Tensor, om: Tensor,
                                degree: int, chunk: int = 65536, use_checkpoint: bool = False
                                ) -> tuple[Tensor, Tensor]:
    """Stream rows in chunks: returns (T~^T T~, T~^T t~). v: (n, k), ell: (n, 9), t, om: (n,)."""
    A = None
    b = None
    n = v.shape[0]
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        args = (v[s:e], None if ell is None else ell[s:e], t[s:e], om[s:e], degree)
        if use_checkpoint and torch.is_grad_enabled():
            Ap, bp = checkpoint(_gram_part, *args, use_reentrant=False)
        else:
            Ap, bp = _gram_part(*args)
        A = Ap if A is None else A + Ap
        b = bp if b is None else b + bp
    return A, b


@dataclass
class SolveInfo:
    cond: float
    promoted: bool


def ridge_solve(A_data: Tensor, b_data: Tensor, lam: Tensor, mu: Tensor,
                fp64_threshold: float = 1e8) -> tuple[Tensor, SolveInfo]:
    """Solve (A_data + lam I) w = b_data + lam mu by Cholesky; promote to float64 if the
    condition estimate exceeds `fp64_threshold` (TD Prop. 14)."""
    F = A_data.shape[0]
    eye = torch.eye(F, dtype=A_data.dtype, device=A_data.device)
    A = A_data + lam * eye
    b = b_data + lam * mu
    with torch.no_grad():
        ev = torch.linalg.eigvalsh(A.detach().double())
        cond = float(ev[-1] / ev[0].clamp_min(torch.finfo(torch.float64).tiny))
    promoted = cond > fp64_threshold and A.dtype != torch.float64
    if promoted:
        L = torch.linalg.cholesky(A.double())
        w = torch.cholesky_solve(b.double().unsqueeze(-1), L).squeeze(-1).to(A.dtype)
    else:
        L = torch.linalg.cholesky(A)
        w = torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
    return w, SolveInfo(cond=cond, promoted=promoted)
