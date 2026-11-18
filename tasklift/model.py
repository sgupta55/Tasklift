"""TaskLift model: lifting analysis -> closed-form compiler -> inverse-shear synthesis.

Stage keys: "root", ("y", j), ("x", j) for j = 1..J (j = 1 is the finest level).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, Union

import torch
from torch import Tensor, nn

from .calibration import fit_calibration
from .compiler import (SolveInfo, accumulate_normal_equations, class_balance_weights,
                       positive_fraction, ridge_solve)
from .config import ModelConfig
from .features import apply_psi_readout, gather3x3, project, psi_dim, to_rows
from .lifting import Level, analysis, interleave, stability_constant
from .stencils import canonical_predictor, canonical_update

StageKey = Union[str, Tuple[str, int]]
BANDS = ("LL", "LH", "HL", "HH", "Cx")


@dataclass
class Episode:
    """Everything compiled for one task, plus diagnostics."""
    readouts: Dict[StageKey, Tensor]
    stats: list                      # per-level band normalisation statistics
    gamma: float = 1.0
    beta: float = 0.0
    solve_info: Dict[StageKey, SolveInfo] = field(default_factory=dict)

    def label_l1(self, J: int) -> list[Tensor]:
        """||w_label||_1 for every y/x stage (Corollary 1)."""
        return [self.readouts[(nu, j)][-9:].abs().sum() for j in range(1, J + 1) for nu in ("y", "x")]

    def amplification(self, J: int) -> float:
        """prod_j rho~_j with rho~ = max(1, ||w_label||_1) (Theorem 2(c))."""
        return float(math.prod(max(1.0, float(v.detach())) for v in self.label_l1(J)))


def _orthonormal_rows(k: int, r: int, gen: torch.Generator) -> Tensor:
    w = torch.empty(k, r)
    nn.init.orthogonal_(w, generator=gen)
    return w


class TaskLift(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None, seed: int = 0):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        J, C, Cp = cfg.levels, cfg.in_channels, cfg.mix_channels
        if cfg.image_size % (2 ** J):
            raise ValueError("image_size must be divisible by 2**levels")
        gen = torch.Generator().manual_seed(seed)

        # Channel mixing: padded identity (TD Sec. 9.4).
        W0 = torch.zeros(Cp, C)
        for i in range(min(Cp, C)):
            W0[i, i] = 1.0
        self.W0 = nn.Parameter(W0)

        # Lifting kernels a, u: (J, 2 axes [x, y], 4 taps).
        a0 = torch.tensor(canonical_predictor(cfg.init_order), dtype=torch.float32)
        u0 = torch.tensor(canonical_update(), dtype=torch.float32)
        a = a0.repeat(J, 2, 1).clone()
        u = u0.repeat(J, 2, 1).clone()
        if cfg.learn_kernels:
            self.a, self.u = nn.Parameter(a), nn.Parameter(u)
        else:
            self.register_buffer("a", a)
            self.register_buffer("u", u)

        # Projections (seeded orthonormal rows).
        self.raw_y, self.raw_x, self.raw_0 = 27 * Cp, 18 * Cp, 9 * Cp
        self.P_y = nn.Parameter(_orthonormal_rows(cfg.width, self.raw_y, gen))
        self.P_x = nn.Parameter(_orthonormal_rows(cfg.width, self.raw_x, gen))
        self.P_0 = nn.Parameter(_orthonormal_rows(cfg.root_width, self.raw_0, gen))

        # Priors: means mu (zero init) and precisions lambda = softplus(raw), lambda_0 = 1.
        self.F = psi_dim(cfg.width, cfg.degree) + 9
        self.F0 = psi_dim(cfg.root_width, cfg.degree)
        self.mu_y = nn.Parameter(torch.zeros(J, self.F))
        self.mu_x = nn.Parameter(torch.zeros(J, self.F))
        self.mu_0 = nn.Parameter(torch.zeros(self.F0))
        raw = math.log(math.expm1(cfg.lam_init))
        self.lam_raw = nn.Parameter(torch.full((2 * J + 1,), raw))

    # ------------------------------------------------------------------ bookkeeping
    @property
    def J(self) -> int:
        return self.cfg.levels

    def num_meta_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_compiled_parameters(self) -> int:
        return 2 * self.J * self.F + self.F0 + 2   # readouts + (gamma, beta)

    def lam(self, key: StageKey) -> Tensor:
        idx = 0 if key == "root" else (2 * key[1] - 1 if key[0] == "y" else 2 * key[1])
        return nn.functional.softplus(self.lam_raw[idx])

    def mu(self, key: StageKey) -> Tensor:
        if key == "root":
            return self.mu_0
        return (self.mu_y if key[0] == "y" else self.mu_x)[key[1] - 1]

    def stage_keys(self) -> list[StageKey]:
        keys: list[StageKey] = ["root"]
        for j in range(self.J, 0, -1):
            keys += [("y", j), ("x", j)]
        return keys

    def analysis_constant(self) -> float:
        """B(rho_a, rho_u) of Prop. 1 for the current kernels."""
        return stability_constant(float(self.a.abs().sum(-1).max()), float(self.u.abs().sum(-1).max()))

    # ------------------------------------------------------------------ image side
    def analyze(self, x: Tensor) -> list[Level]:
        z = torch.einsum("oc,bchw->bohw", self.W0, x)
        return analysis(z, self.a, self.u)

    @staticmethod
    def band_stats(levels: list[Level], eps: float = 1e-6) -> list[dict]:
        """Per-level, per-band, per-channel mean/std over the support set."""
        stats = []
        for lv in levels:
            s = {}
            for b in BANDS:
                f = getattr(lv, b)
                m = f.mean(dim=(0, 2, 3), keepdim=True)
                sd = (f - m).pow(2).mean(dim=(0, 2, 3), keepdim=True).add(eps).sqrt()
                s[b] = (m, sd)
            stats.append(s)
        return stats

    @staticmethod
    def normalize(levels: list[Level], stats: list[dict]) -> list[Level]:
        out = []
        for lv, s in zip(levels, stats):
            out.append(Level(**{b: (getattr(lv, b) - s[b][0]) / s[b][1] for b in BANDS}))
        return out

    def stage_inputs(self, nlevels: list[Level]) -> dict:
        """Projected image features v for every stage (fields (B, k, h, w))."""
        V: dict = {}
        for j, lv in enumerate(nlevels, start=1):
            raw_y = torch.cat([gather3x3(lv.LH), gather3x3(lv.HH), gather3x3(lv.LL)], dim=1)
            V[("y", j)] = project(raw_y, self.P_y)
            D = interleave(lv.HL, lv.HH, dim=-2)
            raw_x = torch.cat([gather3x3(D), gather3x3(lv.Cx)], dim=1)
            V[("x", j)] = project(raw_x, self.P_x)
        V["root"] = project(gather3x3(nlevels[-1].LL), self.P_0)
        return V

    # ------------------------------------------------------------------ label side
    def label_fields(self, y: Tensor) -> list[Tensor]:
        """[Y^(0), ..., Y^(J)] by pure subsampling."""
        return [y[..., :: 2 ** j, :: 2 ** j] for j in range(self.J + 1)]

    @staticmethod
    def split_level(Yprev: Tensor) -> dict:
        """Label split at one level: x-even strip E, x-odd O, coarse Ec = Y^(j), y-odd Oc."""
        E, O = Yprev[..., :, 0::2], Yprev[..., :, 1::2]
        return {"E": E, "O": O, "Ec": E[..., 0::2, :], "Oc": E[..., 1::2, :]}

    def stage_rows(self, V: dict, Ys: list[Tensor], p_hat: Tensor) -> dict:
        """(v rows, label-window rows, targets, weights) for every stage of the support set."""
        rows = {}
        t = Ys[self.J]
        rows["root"] = (to_rows(V["root"]), None, to_rows(t)[:, 0],
                        to_rows(class_balance_weights(t, p_hat))[:, 0])
        for j in range(1, self.J + 1):
            sp = self.split_level(Ys[j - 1])
            for nu, ell_src, tgt in (("y", sp["Ec"], sp["Oc"]), ("x", sp["E"], sp["O"])):
                rows[(nu, j)] = (to_rows(V[(nu, j)]), to_rows(gather3x3(ell_src)), to_rows(tgt)[:, 0],
                                 to_rows(class_balance_weights(tgt, p_hat))[:, 0])
        return rows

    # ------------------------------------------------------------------ compiler
    def compile_episode(self, xs: Tensor, ys: Tensor, calibrate: bool = True) -> Episode:
        """Algorithm: analysis of supports, 2J+1 ridge solves, calibration on swept supports."""
        cfg = self.cfg
        levels = self.analyze(xs)
        stats = self.band_stats(levels)
        V = self.stage_inputs(self.normalize(levels, stats))
        readouts: Dict[StageKey, Tensor] = {}
        infos: Dict[StageKey, SolveInfo] = {}
        if cfg.compile:
            Ys = self.label_fields(ys)
            p_hat = positive_fraction(ys, cfg.class_balance_eps)
            rows = self.stage_rows(V, Ys, p_hat)
            for key in self.stage_keys():
                v, ell, t, om = rows[key]
                A, b = accumulate_normal_equations(v, ell, t, om, cfg.degree,
                                                   cfg.gram_row_chunk, cfg.checkpoint_gram)
                w, info = ridge_solve(A, b, self.lam(key), self.mu(key), cfg.fp64_cond_threshold)
                readouts[key], infos[key] = w, info
        else:
            readouts = {key: self.mu(key) for key in self.stage_keys()}
        ep = Episode(readouts=readouts, stats=stats, solve_info=infos)
        if calibrate:
            with torch.no_grad():
                zs = self.sweep(V, ep)
                p_hat = positive_fraction(ys, cfg.class_balance_eps)
                om = class_balance_weights(ys, p_hat)
                ep.gamma, ep.beta = fit_calibration(zs, ys, om, cfg.calib_ridge, cfg.calib_newton_steps)
        return ep

    # ------------------------------------------------------------------ synthesis
    def _readout(self, v: Tensor, ell: Tensor | None, w: Tensor) -> Tensor:
        k_img = w.shape[0] - (0 if ell is None else 9)
        out = apply_psi_readout(v, w[:k_img], self.cfg.degree)
        if ell is not None:
            out = out + torch.einsum("bchw,c->bhw", ell, w[k_img:]).unsqueeze(1)
        return out

    def sweep(self, V: dict, ep: Episode, root: Tensor | None = None,
              details: dict | None = None) -> Tensor:
        """Inverse-shear synthesis (Sec. IV-A). `root` replaces the root prediction and
        `details` are added to stage predictions (Theorem 2(a),(c))."""
        R = ep.readouts
        z = root if root is not None else self._readout(V["root"], None, R["root"])
        for j in range(self.J, 0, -1):
            pred = self._readout(V[("y", j)], gather3x3(z), R[("y", j)])
            if details is not None:
                pred = pred + details[("y", j)]
            strip = interleave(z, pred, dim=-2)
            pred = self._readout(V[("x", j)], gather3x3(strip), R[("x", j)])
            if details is not None:
                pred = pred + details[("x", j)]
            z = interleave(strip, pred, dim=-1)
        return z

    def label_details(self, V: dict, ep: Episode, y: Tensor) -> dict:
        """Forward label-side analysis: true details d = odd - prediction (Eq. (2))."""
        Ys = self.label_fields(y)
        d = {}
        for j in range(1, self.J + 1):
            sp = self.split_level(Ys[j - 1])
            d[("y", j)] = sp["Oc"] - self._readout(V[("y", j)], gather3x3(sp["Ec"]), ep.readouts[("y", j)])
            d[("x", j)] = sp["O"] - self._readout(V[("x", j)], gather3x3(sp["E"]), ep.readouts[("x", j)])
        return d

    def query_inputs(self, xq: Tensor, ep: Episode) -> dict:
        return self.stage_inputs(self.normalize(self.analyze(xq), ep.stats))

    def predict(self, xq: Tensor, ep: Episode) -> tuple[Tensor, Tensor]:
        """Returns (scores z^(0), calibrated probabilities)."""
        z = self.sweep(self.query_inputs(xq, ep), ep)
        return z, torch.sigmoid(ep.gamma * z + ep.beta)

    def forward(self, xs: Tensor, ys: Tensor, xq: Tensor) -> tuple[Tensor, Tensor, Episode]:
        ep = self.compile_episode(xs, ys)
        z, prob = self.predict(xq, ep)
        return z, prob, ep
