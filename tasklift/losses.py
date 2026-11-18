"""Meta-training objective (TD Sec. 9.1): soft-Dice + BCE on queries, plus R1, R2, ||Theta||^2."""
from __future__ import annotations

import torch
from torch import Tensor


def soft_dice_bce(prob: Tensor, y: Tensor, smooth: float = 1.0, clip: float = 1e-4) -> Tensor:
    """Mean over queries of  1 - soft Dice  +  mean pixel BCE.  prob, y: (B,1,H,W)."""
    p = prob.flatten(1)
    t = y.flatten(1)
    dice = 1 - (2 * (p * t).sum(1) + smooth) / (p.sum(1) + t.sum(1) + smooth)
    pc = p.clamp(clip, 1 - clip)
    bce = -(t * torch.log(pc) + (1 - t) * torch.log(1 - pc)).mean(1)
    return (dice + bce).mean()


def r1_kernel_penalty(a: Tensor, u: Tensor) -> Tensor:
    """R1 = sum_{j,nu} ||a||_1^2 + ||u||_1^2 (controls the constant B of Prop. 1)."""
    return (a.abs().sum(-1) ** 2).sum() + (u.abs().sum(-1) ** 2).sum()


def r2_amplification_penalty(label_l1: list[Tensor]) -> Tensor:
    """R2 = sum (rho - 1)_+^2 with rho = ||w_label||_1 (Corollary 1)."""
    if not label_l1:
        return torch.zeros(())
    return sum(torch.relu(r - 1) ** 2 for r in label_l1)


def l2_penalty(params) -> Tensor:
    return sum((p ** 2).sum() for p in params)
