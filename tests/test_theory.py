"""Every structural statement of the paper, checked numerically (float64)."""
import pytest
import torch

from tasklift import verification as V

CHECKS = {
    "lemma1_1d_nonlinear": V.check_pr_1d_nonlinear,
    "lemma1_2d_random": V.check_pr_2d_cascade,
    "lemma1_2d_near_canonical": lambda: V.check_pr_2d_cascade(near_canonical=True),
    "prop1_bound": lambda: V.check_prop1_bound(n_pairs=100),
    "sum_rules": V.check_sum_rules,
    "closed_form": V.check_closed_form,
    "thm1b_gradient": V.check_theorem1_gradient,
    "thm1c_stability": lambda: V.check_theorem1_stability(trials=50),
    "thm2a_exactness": V.check_theorem2_exactness,
    "thm2b_affinity": V.check_theorem2_affinity,
    "thm2c_propagation": lambda: V.check_theorem2_propagation(trials=10),
    "prop2_equivariance_fp64": V.check_equivariance,
    "prop2_equivariance_fp32": lambda: V.check_equivariance(dtype=torch.float32),
    "determinism": V.check_determinism,
    "parameter_count": V.check_parameter_count,
}


@pytest.mark.parametrize("name", list(CHECKS))
def test_check(name):
    r = CHECKS[name]()
    assert r["passed"], r
