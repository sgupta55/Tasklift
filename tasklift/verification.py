"""Numerical audit of every structural statement in the paper (float64).

Each check returns {"name", "value", "threshold", "passed", "detail"}. Used by
scripts/verify.py (Table V) and by the pytest suite.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .compiler import ridge_solve
from .config import ModelConfig
from .lifting import analysis, fir, lift, stability_constant, synthesis, unlift
from .model import TaskLift
from .stencils import (PREDICT_OFFSETS, UPDATE_OFFSETS, canonical_predictor, canonical_update,
                       sum_rule_moments)

DT = torch.float64


def _res(name, value, threshold, detail="", higher_is_bad=True):
    passed = bool(value <= threshold) if higher_is_bad else bool(value >= threshold)
    return {"name": name, "value": float(value), "threshold": float(threshold),
            "passed": passed, "detail": detail}


# ----------------------------------------------------------------------------- Lemma 1, Prop. 1
def check_pr_1d_nonlinear(n_signals=200, length=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    worst = 0.0
    for _ in range(n_signals):
        x = torch.randn(1, 1, 1, length, generator=g, dtype=DT)
        a = torch.randn(4, generator=g, dtype=DT)
        u = torch.randn(4, generator=g, dtype=DT)
        P = lambda e: fir(e, a, PREDICT_OFFSETS, -1) + torch.tanh(e)          # nonlinear predict
        U = lambda d: fir(d, u, UPDATE_OFFSETS, -1) + 0.3 * torch.tanh(2 * d)  # nonlinear update
        c, d = lift(x, None, None, -1, predict=P, update=U)
        xr = unlift(c, d, None, None, -1, predict=P, update=U)
        worst = max(worst, float((xr - x).abs().max()))
    return _res("Lemma 1: 1-D PR, nonlinear P,U", worst, 1e-12,
                f"{n_signals} signals, length {length}, random 4-tap + tanh")


def check_pr_2d_cascade(size=64, levels=4, seed=0, near_canonical=False):
    """2-D cascade PR. N(0,1) taps (l1-norm ~3) make the cascade ill-conditioned (Prop. 1),
    so the round-off is larger than for kernels near the canonical stencils."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 4, size, size, generator=g, dtype=DT)
    if near_canonical:
        a = torch.tensor(canonical_predictor(4), dtype=DT).repeat(levels, 2, 1)
        u = torch.tensor(canonical_update(), dtype=DT).repeat(levels, 2, 1)
        a = a + 0.1 * torch.randn(a.shape, generator=g, dtype=DT)
        u = u + 0.1 * torch.randn(u.shape, generator=g, dtype=DT)
        kind = "canonical + N(0,0.1^2) kernels"
    else:
        a = torch.randn(levels, 2, 4, generator=g, dtype=DT)
        u = torch.randn(levels, 2, 4, generator=g, dtype=DT)
        kind = "N(0,1) kernels"
    err = float((synthesis(analysis(x, a, u), a, u) - x).abs().max())
    return _res(f"Lemma 1: 2-D cascade PR ({'near-canonical' if near_canonical else 'random'})",
                err, 1e-9, f"{size}x{size}, J={levels}, {kind}")


def _lift_matrix(a, u, n):
    """Dense matrices of the 1-D lifting step and its inverse on length-n periodic signals."""
    eye = torch.eye(n, dtype=DT)
    cols_f, cols_i = [], []
    for k in range(n):
        x = eye[k].view(1, 1, 1, n)
        c, d = lift(x, a, u, -1)
        cols_f.append(torch.cat([c.flatten(), d.flatten()]))
        cd = eye[k]
        cols_i.append(unlift(cd[: n // 2].view(1, 1, 1, -1), cd[n // 2:].view(1, 1, 1, -1), a, u, -1).flatten())
    return torch.stack(cols_f, 1), torch.stack(cols_i, 1)


def check_prop1_bound(n_pairs=500, n=32, seed=0):
    """Ratio of the exact operator 2-norms (both directions) to B. The bound is sharp:
    for some N(0,1) kernel pairs the ratio approaches 1."""
    g = torch.Generator().manual_seed(seed)
    ratios = []
    for _ in range(n_pairs):
        a = torch.randn(4, generator=g, dtype=DT)
        u = torch.randn(4, generator=g, dtype=DT)
        B = stability_constant(float(a.abs().sum()), float(u.abs().sum()))
        Tf, Ti = _lift_matrix(a, u, n)
        ratios.append(max(float(torch.linalg.matrix_norm(Tf, 2)), float(torch.linalg.matrix_norm(Ti, 2))) / B)
    return _res("Prop. 1: both directions, max ratio to B", max(ratios), 1.0 + 1e-12,
                f"{n_pairs} N(0,1) kernel pairs, n={n}; median ratio {np.median(ratios):.3f}")


def check_sum_rules():
    worst = 0.0
    for r in (1, 2, 3, 4):
        nu = sum_rule_moments(canonical_predictor(r), PREDICT_OFFSETS, r)
        target = np.zeros(r)
        target[0] = 1
        worst = max(worst, float(np.abs(nu - target).max()))
    l1 = float(np.abs(canonical_predictor(4)).sum())
    return _res("Sum rules, canonical stencils r=1..4", worst, 1e-12, f"||a*_4||_1 = {l1:.4f}")


# ----------------------------------------------------------------------------- Theorem 1
def check_closed_form(rows=500, F=334, seed=0):
    g = torch.Generator().manual_seed(seed)
    T = torch.randn(rows, F, generator=g, dtype=DT)
    t = torch.randn(rows, generator=g, dtype=DT)
    om = torch.rand(rows, generator=g, dtype=DT) + 0.1
    mu = torch.randn(F, generator=g, dtype=DT)
    lam = torch.tensor(0.7, dtype=DT)
    Tw = T * om.sqrt()[:, None]
    tw = t * om.sqrt()
    w, _ = ridge_solve(Tw.T @ Tw, Tw.T @ tw, lam, mu)
    aug_T = torch.cat([Tw, lam.sqrt() * torch.eye(F, dtype=DT)])
    aug_t = torch.cat([tw, lam.sqrt() * mu])
    w_ls = torch.linalg.lstsq(aug_T, aug_t.unsqueeze(-1)).solution.squeeze(-1)
    return _res("Closed form (3) vs least squares", float((w - w_ls).abs().max()), 1e-10,
                f"{rows}x{F} weighted system")


def check_theorem1_gradient(rows=200, F=20, seed=0):
    """Theorem 1(b): exact differential vs central finite differences."""
    g = torch.Generator().manual_seed(seed)
    T = torch.randn(rows, F, generator=g, dtype=DT)
    t = torch.randn(rows, generator=g, dtype=DT)
    mu = torch.randn(F, generator=g, dtype=DT, requires_grad=True)
    lam = torch.tensor(0.5, dtype=DT, requires_grad=True)
    c = torch.randn(F, generator=g, dtype=DT)

    def f(mu_, lam_):
        w, _ = ridge_solve(T.T @ T, T.T @ t, lam_, mu_)
        return (w * c).sum()

    out = f(mu, lam)
    gmu, glam = torch.autograd.grad(out, (mu, lam))
    h = 1e-6
    with torch.no_grad():
        fd_lam = (f(mu, lam + h) - f(mu, lam - h)) / (2 * h)
        fd_mu = torch.stack([(f(mu + h * e, lam) - f(mu - h * e, lam)) / (2 * h)
                             for e in torch.eye(F, dtype=DT)])
    err = max(float((gmu - fd_mu).abs().max()), float((glam - fd_lam).abs()))
    return _res("Theorem 1(b): meta-gradient vs finite differences", err, 1e-6, "central differences, h=1e-6")


def check_theorem1_stability(trials=200, rows=300, F=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    worst = 0.0
    for _ in range(trials):
        T = torch.randn(rows, F, generator=g, dtype=DT)
        t = torch.randn(rows, generator=g, dtype=DT)
        dT = 0.05 * torch.randn(rows, F, generator=g, dtype=DT)
        dt = 0.05 * torch.randn(rows, generator=g, dtype=DT)
        lam = torch.tensor(float(torch.rand(1, generator=g)) + 0.1, dtype=DT)
        mu = torch.randn(F, generator=g, dtype=DT)
        w, _ = ridge_solve(T.T @ T, T.T @ t, lam, mu)
        T2, t2 = T + dT, t + dt
        w2, _ = ridge_solve(T2.T @ T2, T2.T @ t2, lam, mu)
        n2 = lambda M: float(torch.linalg.matrix_norm(M, 2)) if M.dim() == 2 else float(M.norm())
        bound = (n2(T) * (n2(dt) + n2(dT) * n2(w)) + n2(dT) * (n2(t2) + n2(T2) * n2(w))) / float(lam)
        worst = max(worst, float((w2 - w).norm()) / bound)
    return _res("Theorem 1(c): data-stability bound, max ratio", worst, 1.0 + 1e-12, f"{trials} perturbations")


# ----------------------------------------------------------------------------- Theorem 2, Prop. 2
def _toy_episode(size=64, levels=3, seed=0, n_s=3, dtype=DT):
    torch.manual_seed(seed)
    cfg = ModelConfig(image_size=size, levels=levels)
    model = TaskLift(cfg, seed=seed).to(dtype)
    with torch.no_grad():                         # random (non-canonical) kernels and priors
        model.a.add_(0.05 * torch.randn_like(model.a))
        model.u.add_(0.05 * torch.randn_like(model.u))
        model.mu_y.normal_(0, 0.1)
        model.mu_x.normal_(0, 0.1)
    g = torch.Generator().manual_seed(seed)
    xs = torch.randn(n_s, 4, size, size, generator=g, dtype=DT).to(dtype)
    ys = (torch.rand(n_s, 1, size, size, generator=g, dtype=DT) > 0.6).to(dtype)
    with torch.no_grad():
        ep = model.compile_episode(xs, ys)
    return model, ep, xs, ys, g


@torch.no_grad()
def check_theorem2_exactness(seed=0):
    model, ep, xs, ys, g = _toy_episode(seed=seed)
    V = model.stage_inputs(model.normalize(model.analyze(xs), ep.stats))
    d = model.label_details(V, ep, ys)
    rec = model.sweep(V, ep, root=model.label_fields(ys)[model.J], details=d)
    return _res("Theorem 2(a): sweep with true details returns Y", float((rec - ys).abs().max()), 1e-12)


@torch.no_grad()
def check_theorem2_affinity(seed=0):
    model, ep, xs, ys, g = _toy_episode(seed=seed)
    V = model.stage_inputs(model.normalize(model.analyze(xs), ep.stats))
    shape = model.label_fields(ys)[model.J].shape
    r1 = torch.randn(shape, generator=g, dtype=DT)
    r2 = torch.randn(shape, generator=g, dtype=DT)
    al = 0.3
    lhs = model.sweep(V, ep, root=al * r1 + (1 - al) * r2)
    rhs = al * model.sweep(V, ep, root=r1) + (1 - al) * model.sweep(V, ep, root=r2)
    return _res("Theorem 2(b): affinity of the sweep", float((lhs - rhs).abs().max()), 1e-10)


@torch.no_grad()
def check_theorem2_propagation(trials=20, seed=0):
    model, ep, xs, ys, g = _toy_episode(seed=seed)
    V = model.stage_inputs(model.normalize(model.analyze(xs), ep.stats))
    J = model.J
    rho = {k: max(1.0, float(ep.readouts[k][-9:].abs().sum())) for k in ep.readouts if k != "root"}
    base_root = model.label_fields(ys)[J]
    worst = 0.0
    for _ in range(trials):
        dJ = 0.1 * torch.randn(base_root.shape, generator=g, dtype=DT)
        zero = model.label_details(V, ep, ys)
        zero = {k: torch.zeros_like(v) for k, v in zero.items()}
        res = {k: 0.05 * torch.randn(v.shape, generator=g, dtype=DT) for k, v in zero.items()}
        z1 = model.sweep(V, ep, root=base_root, details=zero)
        z2 = model.sweep(V, ep, root=base_root + dJ, details=res)
        delta0 = float((z2 - z1).abs().max())
        prod = 1.0
        bound = 0.0
        for j in range(1, J + 1):     # sum over j of prod_{i<j} rho_i (rho^x_j |r^y_j| + |r^x_j|)
            bound += prod * (rho[("x", j)] * float(res[("y", j)].abs().max()) + float(res[("x", j)].abs().max()))
            prod *= rho[("y", j)] * rho[("x", j)]
        bound += prod * float(dJ.abs().max())
        worst = max(worst, delta0 / bound)
    return _res("Theorem 2(c): propagation bound, max ratio", worst, 1.0 + 1e-12, f"{trials} perturbations")


@torch.no_grad()
def check_equivariance(size=64, levels=3, seed=0, dtype=DT):
    model, ep, xs, ys, g = _toy_episode(size=size, levels=levels, seed=seed, dtype=dtype)
    xq = torch.randn(2, 4, size, size, generator=g, dtype=DT).to(dtype)
    s = 2 ** levels
    z, _ = model.predict(xq, ep)
    z_shift, _ = model.predict(torch.roll(xq, shifts=(s, 2 * s), dims=(-2, -1)), ep)
    err = float((z_shift - torch.roll(z, shifts=(s, 2 * s), dims=(-2, -1))).abs().max())
    tol = 1e-10 if dtype == torch.float64 else 1e-4
    return _res(f"Prop. 2: equivariance to {s}-px shifts ({str(dtype).split('.')[-1]})", err, tol)


@torch.no_grad()
def check_determinism(size=64, levels=3):
    outs = []
    for _ in range(2):
        model, ep, xs, ys, g = _toy_episode(size=size, levels=levels, seed=3)
        z, _ = model.predict(xs, ep)
        outs.append(z)
    return _res("Determinism: identical runs are bit-identical", float((outs[0] - outs[1]).abs().max()), 0.0)


def check_parameter_count():
    m = TaskLift(ModelConfig())
    ok = m.num_meta_parameters() == 7810 and m.F == 334 and m.F0 == 153 and m.num_compiled_parameters() == 2827
    return {"name": "Parameter count (default config)", "value": m.num_meta_parameters(), "threshold": 7810,
            "passed": ok, "detail": f"F={m.F}, F0={m.F0}, compiled={m.num_compiled_parameters()}"}


ALL_CHECKS = [check_pr_1d_nonlinear, check_pr_2d_cascade,
              lambda: check_pr_2d_cascade(near_canonical=True), check_prop1_bound, check_sum_rules,
              check_closed_form, check_theorem1_gradient, check_theorem1_stability,
              check_theorem2_exactness, check_theorem2_affinity, check_theorem2_propagation,
              check_equivariance, lambda: check_equivariance(dtype=torch.float32),
              check_determinism, check_parameter_count]


def run_all() -> list[dict]:
    out = []
    for fn in ALL_CHECKS:
        t = time.perf_counter()
        r = fn()
        r["seconds"] = time.perf_counter() - t
        out.append(r)
    return out
