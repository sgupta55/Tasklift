"""Canonical lifting stencils and sum-rule moments (paper Sec. II-B, Thm. S1).

A predictor (P_a e)[k] = sum_m a_m e[k+m] annihilates polynomials of degree < r
("order-r sum rules") iff  sum_m a_m (2m - 1)^s = delta_{s,0}  for s = 0..r-1,
where 2m-1 is the position of the even sample e[k+m] relative to the odd sample.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import numpy as np

#: Predictor offsets for the default 4-tap kernels.
PREDICT_OFFSETS: tuple[int, ...] = (-1, 0, 1, 2)
#: Update offsets (c[k] = e[k] + sum_m u_m d[k+m]); d[k-1], d[k] are the odd neighbours of e[k].
UPDATE_OFFSETS: tuple[int, ...] = (-2, -1, 0, 1)

#: Offsets used for the order-r canonical stencil inside the 4-tap support.
_ORDER_SUBSETS = {1: (0,), 2: (0, 1), 3: (-1, 0, 1), 4: (-1, 0, 1, 2)}


def sum_rule_stencil_exact(r: int) -> dict[int, Fraction]:
    """Exact rational solution of the order-r sum rules on the canonical offsets."""
    if r not in _ORDER_SUBSETS:
        raise ValueError(f"order r must be in {sorted(_ORDER_SUBSETS)}, got {r}")
    offs = _ORDER_SUBSETS[r]
    n = len(offs)
    # Vandermonde system V a = e0 with V[s, m] = (2m - 1)^s, solved exactly.
    V = [[Fraction(2 * m - 1) ** s for m in offs] for s in range(n)]
    rhs = [Fraction(1)] + [Fraction(0)] * (n - 1)
    # Gauss-Jordan elimination over the rationals.
    M = [row[:] + [rhs[i]] for i, row in enumerate(V)]
    for col in range(n):
        piv = next(i for i in range(col, n) if M[i][col] != 0)
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        M[col] = [x / p for x in M[col]]
        for i in range(n):
            if i != col and M[i][col] != 0:
                f = M[i][col]
                M[i] = [a - f * b for a, b in zip(M[i], M[col])]
    return {m: M[i][n] for i, m in enumerate(offs)}


def canonical_predictor(r: int = 4, offsets: Sequence[int] = PREDICT_OFFSETS) -> np.ndarray:
    """Order-r stencil embedded in the tap vector on `offsets` (zeros elsewhere)."""
    exact = sum_rule_stencil_exact(r)
    missing = set(exact) - set(offsets)
    if missing:
        raise ValueError(f"offsets {offsets} do not contain stencil offsets {sorted(missing)}")
    return np.array([float(exact.get(m, 0)) for m in offsets])


def canonical_update(offsets: Sequence[int] = UPDATE_OFFSETS) -> np.ndarray:
    """Haar-type update: a single tap 1/2 at offset 0."""
    return np.array([0.5 if m == 0 else 0.0 for m in offsets])


def sum_rule_moments(a: Sequence[float], offsets: Sequence[int], r: int) -> np.ndarray:
    """nu_s = sum_m a_m (2m-1)^s for s = 0..r-1 (target: 1, 0, 0, ...)."""
    a = np.asarray(a, dtype=float)
    return np.array([sum(am * (2 * m - 1) ** s for am, m in zip(a, offsets)) for s in range(r)])
