import math

import numpy as np
import pytest
import torch

from tasklift import ModelConfig, TaskLift
from tasklift.config import load_config
from tasklift.features import apply_psi_readout, psi_dim, psi_rows
from tasklift.metrics import assd, bootstrap_ci, dice, iou
from tasklift.stencils import canonical_predictor


def test_canonical_stencil_values():
    assert np.allclose(canonical_predictor(4), np.array([-1, 9, 9, -1]) / 16)
    assert np.allclose(canonical_predictor(3), [-1 / 8, 3 / 4, 3 / 8, 0])
    assert math.isclose(np.abs(canonical_predictor(4)).sum(), 1.25)


@pytest.mark.parametrize("degree", [1, 2])
def test_fast_readout_equals_explicit(degree):
    g = torch.Generator().manual_seed(0)
    v = torch.randn(2, 5, 4, 3, generator=g, dtype=torch.float64)
    w = torch.randn(psi_dim(5, degree), generator=g, dtype=torch.float64)
    fast = apply_psi_readout(v, w, degree)[:, 0]
    explicit = psi_rows(v.permute(0, 2, 3, 1), degree) @ w
    assert torch.allclose(fast, explicit, atol=1e-12)


def test_feature_dimensions():
    assert psi_dim(24, 2) + 9 == 334 and psi_dim(16, 2) == 153 and psi_dim(24, 1) + 9 == 34


def test_config_overrides_and_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  fp64_cond_threshold: 1.0e8\n  levels: 3\ntrain:\n  train_shots: [1, 5]\n")
    m, t = load_config(p, ["degree=1", "learn_kernels=false", "eta2=0.0", "train_shots=2,3"])
    assert m.fp64_cond_threshold == 1e8 and isinstance(m.fp64_cond_threshold, float)
    assert m.levels == 3 and m.degree == 1 and m.learn_kernels is False
    assert t.eta2 == 0.0 and t.train_shots == (2, 3)


def test_ablation_parameter_counts():
    assert TaskLift(ModelConfig(learn_kernels=False)).num_meta_parameters() == 7810 - 64
    assert TaskLift(ModelConfig(degree=1)).F == 34


def test_metrics():
    a = np.zeros((32, 32), bool)
    a[8:24, 8:24] = True
    assert dice(a, a) == 1.0 and iou(a, a) == 1.0 and assd(a, a) == 0.0
    b = np.roll(a, 2, axis=1)
    assert 0 < dice(a, b) < 1 and iou(a, b) < dice(a, b)
    assert dice(np.zeros_like(a), a) == 0.0                     # empty prediction scores 0
    assert assd(np.zeros_like(a), a) == pytest.approx(math.hypot(32, 32))
    assert 0 < assd(a, b) <= 2.0
    mean, lo, hi, hw = bootstrap_ci([0.5, 0.6, 0.7], n_boot=500)
    assert lo <= mean <= hi and hw >= 0


def test_episode_forward_backward_and_compile_off():
    cfg = ModelConfig(image_size=32, levels=2)
    torch.manual_seed(0)
    xs, xq = torch.randn(3, 4, 32, 32), torch.randn(2, 4, 32, 32)
    ys = (torch.rand(3, 1, 32, 32) > 0.6).float()
    for compile_ in (True, False):
        m = TaskLift(ModelConfig(**{**cfg.to_dict(), "compile": compile_}))
        z, p, ep = m(xs, ys, xq)
        assert z.shape == (2, 1, 32, 32) and torch.all((p >= 0) & (p <= 1))
        p.mean().backward()
        grads = [q.grad for q in m.parameters() if q.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
    m = TaskLift(cfg)
    ep = m.compile_episode(xs, ys)
    assert all(ep.readouts[k].shape[0] == (m.F if k != "root" else m.F0) for k in m.stage_keys())
