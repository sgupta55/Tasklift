#!/usr/bin/env python
"""Create tiny synthetic raw datasets in the real on-disk layouts (smoke tests only).

  python scripts/make_synthetic_data.py --out data/synthetic_raw
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def blob_image(rng, h, w, thin=False):
    yy, xx = np.mgrid[:h, :w]
    mask = np.zeros((h, w), bool)
    if thin:
        for _ in range(6):
            a, b = rng.uniform(0, np.pi), rng.uniform(0, w)
            mask |= np.abs((xx - b) * np.sin(a) - yy * np.cos(a)) < 2.0
    else:
        cy, cx = rng.uniform(0.3, 0.7) * h, rng.uniform(0.3, 0.7) * w
        r = rng.uniform(0.12, 0.3) * min(h, w)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
    img = rng.normal(0.4, 0.1, (h, w, 3))
    img[mask] += rng.uniform(0.2, 0.4, 3)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8), mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_raw")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = Path(a.out)
    # ISIC
    for d in ("ISIC2018_Task1-2_Training_Input", "ISIC2018_Task1_Training_GroundTruth"):
        (out / "isic" / d).mkdir(parents=True, exist_ok=True)
    for i in range(a.n):
        img, m = blob_image(rng, 300, 400)
        Image.fromarray(img).save(out / "isic/ISIC2018_Task1-2_Training_Input" / f"ISIC_{i:07d}.jpg")
        Image.fromarray((m * 255).astype(np.uint8)).save(
            out / "isic/ISIC2018_Task1_Training_GroundTruth" / f"ISIC_{i:07d}_segmentation.png")
    # Kvasir-SEG
    for d in ("images", "masks"):
        (out / "kvasir" / d).mkdir(parents=True, exist_ok=True)
    for i in range(a.n):
        img, m = blob_image(rng, 320, 320)
        Image.fromarray(img).save(out / "kvasir/images" / f"k{i:04d}.jpg")
        Image.fromarray((m * 255).astype(np.uint8)).save(out / "kvasir/masks" / f"k{i:04d}.png")
    # CHASE_DB1 (28 images)
    (out / "chase").mkdir(parents=True, exist_ok=True)
    for i in range(14):
        for side in "LR":
            img, m = blob_image(rng, 240, 250, thin=True)
            Image.fromarray(img).save(out / "chase" / f"Image_{i + 1:02d}{side}.jpg")
            Image.fromarray((m * 255).astype(np.uint8)).save(out / "chase" / f"Image_{i + 1:02d}{side}_1stHO.png")
    # BraTS 2021
    try:
        import nibabel as nib
        for c in range(12):
            cid = f"BraTS2021_{c:05d}"
            d = out / "brats" / cid
            d.mkdir(parents=True, exist_ok=True)
            seg = np.zeros((64, 64, 20), np.uint8)
            zz, yy, xx = np.mgrid[:64, :64, :20]
            seg[((zz - 32) ** 2 + (yy - 30) ** 2 + (xx - 10) ** 2 * 4) < rng.uniform(80, 200)] = 2
            for s in ("t1", "t1ce", "t2", "flair"):
                vol = rng.normal(0, 1, seg.shape).astype(np.float32) + 2.0 * (seg > 0)
                nib.save(nib.Nifti1Image(vol, np.eye(4)), str(d / f"{cid}_{s}.nii.gz"))
            nib.save(nib.Nifti1Image(seg, np.eye(4)), str(d / f"{cid}_seg.nii.gz"))
    except ImportError:
        print("nibabel missing: BraTS skipped")
    print("synthetic raw data in", out)


if __name__ == "__main__":
    main()
