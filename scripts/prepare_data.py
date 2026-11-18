#!/usr/bin/env python
"""Preprocess the four datasets and write deterministic splits.

Example:
  python scripts/prepare_data.py --dataset isic   --root /data/ISIC2018 --out data/processed
  python scripts/prepare_data.py --dataset brats  --root /data/BraTS2021_Training_Data --out data/processed
  python scripts/prepare_data.py --dataset kvasir --root /data/Kvasir-SEG --out data/processed
  python scripts/prepare_data.py --dataset chase  --root /data/CHASE_DB1 --out data/processed
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasklift.data.preprocess import DATASETS, prepare_brats, prepare_rgb, write_splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--root", required=True, help="raw dataset folder")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--max-slices", type=int, default=96, help="BraTS: max non-empty slices per case")
    ap.add_argument("--seed", type=int, default=0, help="split seed")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N items/cases")
    a = ap.parse_args()
    if a.dataset == "brats":
        n = prepare_brats(a.root, a.out, a.size, a.max_slices, a.limit)
    else:
        n = prepare_rgb(a.dataset, a.root, a.out, a.size, a.limit)
    counts = write_splits(a.out, a.dataset, a.seed)
    print(f"{a.dataset}: {n} items -> {a.out}/{a.dataset}; split sizes {counts}")


if __name__ == "__main__":
    main()
