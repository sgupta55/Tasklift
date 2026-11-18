#!/usr/bin/env python
"""Evaluate a checkpoint (or the mean-support-mask control) under the paper protocol.

  python scripts/evaluate.py --ckpt runs/W_isic/best.pt --dataset isic --shots 1 5
  python scripts/evaluate.py --ckpt runs/W_isic/best.pt --dataset isic --shots 1 2 3 5 10   # Fig. 2
  python scripts/evaluate.py --baseline mean_mask --dataset isic --shots 1 5                 # Fig. 3 control
  python scripts/evaluate.py --ckpt ... --dataset isic --shots 1 5 --support-variance       # Sec. V-D

Writes <out>/<dataset>_<n>shot.json with summary (mean, 95% CI half-width), per-episode and
per-query records, tertiles of target extent, amplification statistics and timing.
CHASE_DB1 always uses R-CHASE (all C(6, n_s) support draws, n_s <= 5).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasklift.data import ItemStore, eval_episodes  # noqa: E402
from tasklift.engine import evaluate, load_model, support_variance  # noqa: E402
from tasklift.utils import resolve_device, save_json, set_determinism  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--baseline", default=None, choices=[None, "mean_mask"])
    ap.add_argument("--dataset", required=True, choices=["isic", "brats", "kvasir", "chase"])
    ap.add_argument("--data-root", default="data/processed")
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--support-variance", action="store_true")
    ap.add_argument("--no-distinct-groups", action="store_true",
                    help="allow several images of one case/patient in an episode")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="output folder (default: next to the checkpoint)")
    a = ap.parse_args()
    if (a.ckpt is None) == (a.baseline is None):
        ap.error("give exactly one of --ckpt or --baseline")
    set_determinism(a.seed)
    device = resolve_device(a.device)
    model = None
    if a.ckpt:
        model, _ = load_model(a.ckpt, device)
    out = Path(a.out or (Path(a.ckpt).parent / "eval" if a.ckpt else Path("runs/baselines") / a.baseline))
    store = ItemStore(a.data_root, a.dataset)
    for s in a.shots:
        if a.dataset == "chase" and s > 5:
            print("R-CHASE has 6 test images: skipping n_s =", s)
            continue
        specs = eval_episodes(store, s, a.queries, a.episodes, a.seed,
                              distinct_groups=not a.no_distinct_groups)
        res = evaluate(model, store, specs, device, baseline=a.baseline)
        if a.support_variance and model is not None:
            res["summary"]["support_variance"] = support_variance(model, store, s, device, n_query=a.queries, seed=a.seed)
        save_json(res, out / f"{a.dataset}_{s}shot.json")
        sm = res["summary"]
        print(json.dumps({"dataset": a.dataset, "shots": s,
                          "dice": f"{sm['dice']['mean']:.1f} +- {sm['dice']['half_width']:.1f}",
                          "iou": f"{sm['iou']['mean']:.1f} +- {sm['iou']['half_width']:.1f}",
                          "assd": f"{sm['assd']['mean']:.1f} +- {sm['assd']['half_width']:.1f}",
                          "episodes": sm["n_episodes"], "queries": sm["n_query_predictions"]}))


if __name__ == "__main__":
    main()
