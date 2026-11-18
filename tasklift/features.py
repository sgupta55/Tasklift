"""Lattice-aligned gathers and polynomial feature maps (paper Sec. III-A).

Window ordering: index (di + 1) * 3 + (dj + 1) for di, dj in {-1, 0, 1}, where the
gathered value at site (i, j) is field[i + di, j + dj] (periodic).
"""
from __future__ import annotations

import torch
from torch import Tensor

WINDOW = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
CENTER, BELOW, RIGHT = 4, 7, 5   # window indices of (0,0), (+1,0), (0,+1)


def gather3x3(f: Tensor) -> Tensor:
    """(B, C, h, w) -> (B, 9C, h, w), channel-major, periodic."""
    outs = [torch.roll(f, shifts=(-di, -dj), dims=(-2, -1)) for di, dj in WINDOW]
    return torch.stack(outs, dim=2).flatten(1, 2)


def psi_dim(k: int, degree: int) -> int:
    """Size of psi(v) for v in R^k: 1 + k (+ k(k+1)/2 if degree 2)."""
    if degree == 1:
        return 1 + k
    if degree == 2:
        return 1 + k + k * (k + 1) // 2
    raise ValueError("degree must be 1 or 2")


def psi_rows(v: Tensor, degree: int) -> Tensor:
    """Explicit polynomial expansion for row-major features v: (..., k) -> (..., psi_dim)."""
    ones = torch.ones_like(v[..., :1])
    if degree == 1:
        return torch.cat([ones, v], dim=-1)
    k = v.shape[-1]
    iu = torch.triu_indices(k, k, device=v.device)
    quad = v[..., iu[0]] * v[..., iu[1]]
    return torch.cat([ones, v, quad], dim=-1)


def apply_psi_readout(v: Tensor, w_img: Tensor, degree: int) -> Tensor:
    """Evaluate psi(v)^T w_img on fields v: (B, k, h, w) without materialising psi.

    Identical (up to rounding) to psi_rows(v).dot(w_img); used in the synthesis sweep."""
    k = v.shape[1]
    out = w_img[0] + torch.einsum("bkhw,k->bhw", v, w_img[1:1 + k])
    if degree == 2:
        iu = torch.triu_indices(k, k, device=v.device)
        U = torch.zeros(k, k, dtype=v.dtype, device=v.device)
        U = U.index_put((iu[0], iu[1]), w_img[1 + k:])
        out = out + torch.einsum("bphw,pq,bqhw->bhw", v, U, v)
    return out.unsqueeze(1)


def project(raw: Tensor, P: Tensor) -> Tensor:
    """Per-site linear projection: (B, R, h, w) -> (B, k, h, w) with P: (k, R)."""
    return torch.einsum("kr,brhw->bkhw", P, raw)


def to_rows(f: Tensor) -> Tensor:
    """(B, F, h, w) -> (B*h*w, F)."""
    return f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])
