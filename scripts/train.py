#!/usr/bin/env python
"""Meta-train TaskLift.

  # within-dataset (W)
  python scripts/train.py --config configs/default.yaml --set protocol=within dataset=isic out_dir=runs/W_isic
  # leave-one-dataset-out (X): meta-train on the other three, hold out `dataset`
  python scripts/train.py --config configs/default.yaml --set protocol=cross dataset=brats out_dir=runs/X_brats
  # ablation: add an ablation file on top of the default
  python scripts/train.py --config configs/default.yaml --ablation configs/ablations/compile_off.yaml ...
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasklift.config import load_config  # noqa: E402
from tasklift.engine import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ablation", default=None, help="YAML with `model:`/`train:` overrides")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    a = ap.parse_args()
    overrides = []
    if a.ablation:
        raw = yaml.safe_load(Path(a.ablation).read_text()) or {}
        for sec in ("model", "train"):
            for k, v in (raw.get(sec) or {}).items():
                overrides.append(f"{k}={','.join(map(str, v)) if isinstance(v, list) else v}")
    model_cfg, train_cfg = load_config(a.config, overrides + a.set)
    best = train(model_cfg, train_cfg)
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
