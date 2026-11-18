"""Lifting analysis and synthesis on a periodic dyadic lattice (paper Sec. II).

All fields are tensors shaped (B, C, H, W). The x-axis is the last dimension
(columns), the y-axis the second-to-last (rows). Boundaries are periodic, so
every kernel application is circulant (Prop. 1, Prop. 2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor

from .stencils import PREDICT_OFFSETS, UPDATE_OFFSETS

Operator = Callable[[Tensor], Tensor]


# ----------------------------------------------------------------------------- polyphase helpers
def even(x: Tensor, dim: int) -> Tensor:
    return x.movedim(dim, -1)[..., 0::2].movedim(-1, dim)


def odd(x: Tensor, dim: int) -> Tensor:
    return x.movedim(dim, -1)[..., 1::2].movedim(-1, dim)


def interleave(e: Tensor, o: Tensor, dim: int) -> Tensor:
    """Inverse of the polyphase split: out[2k] = e[k], out[2k+1] = o[k] along `dim`."""
    e_, o_ = e.movedim(dim, -1), o.movedim(dim, -1)
    out = torch.stack((e_, o_), dim=-1).flatten(-2)
    return out.movedim(-1, dim)


def fir(x: Tensor, taps: Tensor, offsets: Sequence[int], dim: int) -> Tensor:
    """Periodic FIR filter y[k] = sum_m taps[m] * x[k + offsets[m]] along `dim`."""
    out = torch.zeros_like(x)
    for i, m in enumerate(offsets):
        out = out + taps[i] * torch.roll(x, shifts=-m, dims=dim)
    return out


# ----------------------------------------------------------------------------- one lifting step
def lift(x: Tensor, a: Tensor | None, u: Tensor | None, dim: int,
         predict: Operator | None = None, update: Operator | None = None,
         offsets_a: Sequence[int] = PREDICT_OFFSETS,
         offsets_u: Sequence[int] = UPDATE_OFFSETS) -> tuple[Tensor, Tensor]:
    """Eq. (1): d = o - P e,  c = e + U d.

    `predict` / `update` may be arbitrary (nonlinear) callables; otherwise the
    linear periodic FIR operators with taps `a`, `u` are used (Lemma 1)."""
    e, o = even(x, dim), odd(x, dim)
    P = predict if predict is not None else (lambda v: fir(v, a, offsets_a, dim))
    U = update if update is not None else (lambda v: fir(v, u, offsets_u, dim))
    d = o - P(e)
    c = e + U(d)
    return c, d


def unlift(c: Tensor, d: Tensor, a: Tensor | None, u: Tensor | None, dim: int,
           predict: Operator | None = None, update: Operator | None = None,
           offsets_a: Sequence[int] = PREDICT_OFFSETS,
           offsets_u: Sequence[int] = UPDATE_OFFSETS) -> Tensor:
    """Exact inverse of `lift`: e = c - U d,  o = d + P e."""
    P = predict if predict is not None else (lambda v: fir(v, a, offsets_a, dim))
    U = update if update is not None else (lambda v: fir(v, u, offsets_u, dim))
    e = c - U(d)
    o = d + P(e)
    return interleave(e, o, dim)


# ----------------------------------------------------------------------------- 2-D cascade
@dataclass
class Level:
    """Bands produced by one analysis level (all at half resolution in y; LL/LH/HL/HH also in x)."""
    LL: Tensor
    LH: Tensor
    HL: Tensor
    HH: Tensor
    Cx: Tensor   # x-even (smooth) strip, needed by the x-stage features


def analysis_level(x: Tensor, ax: Tensor, ux: Tensor, ay: Tensor, uy: Tensor) -> Level:
    """x-lift giving strips (Cx, Dx); y-lifts Cx -> (LL, LH) and Dx -> (HL, HH)."""
    Cx, Dx = lift(x, ax, ux, dim=-1)
    LL, LH = lift(Cx, ay, uy, dim=-2)
    HL, HH = lift(Dx, ay, uy, dim=-2)
    return Level(LL=LL, LH=LH, HL=HL, HH=HH, Cx=Cx)


def synthesis_level(lv: Level, ax: Tensor, ux: Tensor, ay: Tensor, uy: Tensor,
                    LL: Tensor | None = None) -> Tensor:
    """Exact inverse of `analysis_level` (LL may be replaced, e.g. by a reconstructed one)."""
    Cx = unlift(lv.LL if LL is None else LL, lv.LH, ay, uy, dim=-2)
    Dx = unlift(lv.HL, lv.HH, ay, uy, dim=-2)
    return unlift(Cx, Dx, ax, ux, dim=-1)


def analysis(x: Tensor, a: Tensor, u: Tensor) -> list[Level]:
    """J-level cascade recursing on LL. a, u: (J, 2, taps) with axis index 0 = x, 1 = y."""
    levels: list[Level] = []
    cur = x
    for j in range(a.shape[0]):
        lv = analysis_level(cur, a[j, 0], u[j, 0], a[j, 1], u[j, 1])
        levels.append(lv)
        cur = lv.LL
    return levels


def synthesis(levels: list[Level], a: Tensor, u: Tensor) -> Tensor:
    """Exact inverse of `analysis`."""
    cur = levels[-1].LL
    for j in reversed(range(len(levels))):
        cur = synthesis_level(levels[j], a[j, 0], u[j, 0], a[j, 1], u[j, 1], LL=cur)
    return cur


def stability_constant(alpha: float, beta: float) -> float:
    """Prop. 1: B(alpha, beta) = sqrt(1 + alpha^2 + beta^2 + (1 + alpha*beta)^2)."""
    return float((1 + alpha ** 2 + beta ** 2 + (1 + alpha * beta) ** 2) ** 0.5)
