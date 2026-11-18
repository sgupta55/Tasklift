#!/usr/bin/env python
"""Qualitative figure (paper Fig. 3) built from real model outputs, with the Dice of every
prediction printed on its panel.

  python scripts/qualitative.py \
      --ckpt isic=runs/W_isic/best.pt brats=runs/W_brats/best.pt kvasir=runs/W_kvasir/best.pt chase=runs/W_chase/best.pt \
      --fixed isic=runs/abl_fixed_stencils_isic/best.pt ... \
      --select median --out figures/qualitative.pdf

--select first   : the first episode in the fixed-seed evaluation order (and its first query)
--select median  : the episode whose TaskLift Dice is the median over the evaluation episodes
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasklift.baselines import mean_support_mask  # noqa: E402
from tasklift.data import ItemStore, eval_episodes  # noqa: E402
from tasklift.engine import load_episode, load_model  # noqa: E402
from tasklift.metrics import dice  # noqa: E402
from tasklift.utils import resolve_device  # noqa: E402

NAMES = {"isic": "ISIC 2018", "brats": "BraTS 2021", "kvasir": "Kvasir-SEG", "chase": "CHASE_DB1"}


def to_rgb(x: np.ndarray, dataset: str) -> np.ndarray:
    img = x[3] if dataset == "brats" else x[:3].transpose(1, 2, 0)   # FLAIR for MRI
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return np.stack([img] * 3, -1) if img.ndim == 2 else img


def parse_pairs(items):
    return dict(i.split("=", 1) for i in items or [])


@torch.no_grad()
def predict(model, xs, ys, xq):
    _, prob, _ = model(xs, ys, xq)
    return (prob >= 0.5).float()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True, help="dataset=path")
    ap.add_argument("--fixed", nargs="*", default=[], help="dataset=path (fixed-stencil control)")
    ap.add_argument("--data-root", default="data/processed")
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--select", choices=["first", "median"], default="median")
    ap.add_argument("--episodes", type=int, default=100, help="episodes scanned for --select median")
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--no-distinct-groups", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="figures/qualitative.pdf")
    a = ap.parse_args()
    device = resolve_device(a.device)
    ck, fx = parse_pairs(a.ckpt), parse_pairs(a.fixed)
    rows = [(d, s) for d in ck for s in a.shots if not (d == "chase" and s > 5)]
    cols = ["Support (+mask)", "Query", "Ground truth", "TaskLift", "Mean support mask"] + \
           (["Fixed stencils"] if fx else [])
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.6 * len(cols), 1.6 * len(rows)), squeeze=False)
    for r, (d, s) in enumerate(rows):
        model, _ = load_model(ck[d], device)
        fixed = load_model(fx[d], device)[0] if d in fx else None
        store = ItemStore(a.data_root, d)
        specs = eval_episodes(store, s, a.queries, a.episodes,
                              distinct_groups=not a.no_distinct_groups)
        if a.select == "first":
            spec, qi = specs[0], 0
        else:
            scores = []
            for sp in specs:
                xs, ys, xq, yq = load_episode({d: store}, sp, device)
                p = predict(model, xs, ys, xq).cpu().numpy()[:, 0]
                g = yq.cpu().numpy()[:, 0] > 0.5
                scores.append([dice(pp, gg) for pp, gg in zip(p, g)])
            ep_means = np.array([np.mean(v) for v in scores])
            k = int(np.argsort(ep_means)[len(ep_means) // 2])
            spec = specs[k]
            qi = int(np.argsort(scores[k])[len(scores[k]) // 2])
        xs, ys, xq, yq = load_episode({d: store}, spec, device)
        preds = {"TaskLift": predict(model, xs, ys, xq)[qi, 0].cpu().numpy(),
                 "Mean support mask": (mean_support_mask(ys, 1)[0, 0] >= 0.5).float().cpu().numpy()}
        if fixed is not None:
            preds["Fixed stencils"] = predict(fixed, xs, ys, xq)[qi, 0].cpu().numpy()
        gt = yq[qi, 0].cpu().numpy() > 0.5
        sup = to_rgb(xs[0].cpu().numpy(), d)
        sup_m = ys[0, 0].cpu().numpy() > 0.5
        panels = [sup, to_rgb(xq[qi].cpu().numpy(), d), gt] + [preds.get(c) for c in cols[3:]]
        for c, (ax, img) in enumerate(zip(axes[r], panels)):
            if img is None:                       # no checkpoint for this control
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=7, transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(cols[c], fontsize=7)
                continue
            ax.imshow(img, cmap=None if img.ndim == 3 else "gray", interpolation="nearest")
            if c == 0:
                ax.contour(sup_m, levels=[0.5], colors="lime", linewidths=0.6)
                ax.set_ylabel(f"{NAMES[d]}\n{s}-shot", fontsize=7)
            if c >= 3:
                ax.text(0.03, 0.97, f"Dice {100 * dice(img > 0.5, gt):.1f}", transform=ax.transAxes,
                        fontsize=6, va="top", color="yellow", bbox=dict(fc="black", alpha=0.5, pad=1, lw=0))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=7)
    fig.tight_layout(pad=0.2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
