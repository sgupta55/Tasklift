#!/usr/bin/env bash
# Reproduce every experiment in the paper. Expects preprocessed data in $DATA (see README).
# Usage: DATA=data/processed DEVICE=cuda bash scripts/run_paper.sh
set -euo pipefail
DATA=${DATA:-data/processed}
DEVICE=${DEVICE:-auto}
RUNS=${RUNS:-runs}
DATASETS="isic brats kvasir chase"
ABLATIONS="compile_off fixed_stencils r2_off J2 J3 J5 r2 r3 degree1 width16 width32 cp1 cp8"

python scripts/verify.py --out "$RUNS/verification"            # Table V (structural checks)
python scripts/benchmark.py --device "$DEVICE" --out "$RUNS/benchmark.json"   # Table V (cost)

for d in $DATASETS; do
  # Within-dataset (Table I, Fig. 2, Table IV, support variance)
  python scripts/train.py --config configs/default.yaml \
      --set protocol=within dataset=$d data_root=$DATA out_dir=$RUNS/W_$d device=$DEVICE
  python scripts/evaluate.py --ckpt $RUNS/W_$d/best.pt --dataset $d --data-root $DATA \
      --shots 1 2 3 5 10 --support-variance --device $DEVICE
  # Leave-one-dataset-out (Table I, Table II)
  python scripts/train.py --config configs/default.yaml \
      --set protocol=cross dataset=$d data_root=$DATA out_dir=$RUNS/X_$d device=$DEVICE
  python scripts/evaluate.py --ckpt $RUNS/X_$d/best.pt --dataset $d --data-root $DATA --shots 1 5 --device $DEVICE
  # Training-free control (Fig. 3)
  python scripts/evaluate.py --baseline mean_mask --dataset $d --data-root $DATA --shots 1 5 \
      --out $RUNS/baselines/mean_mask
  # Ablations (Table III, within-dataset 5-shot)
  for a in $ABLATIONS; do
    python scripts/train.py --config configs/default.yaml --ablation configs/ablations/$a.yaml \
        --set protocol=within dataset=$d data_root=$DATA out_dir=$RUNS/abl_${a}_$d device=$DEVICE
    python scripts/evaluate.py --ckpt $RUNS/abl_${a}_$d/best.pt --dataset $d --data-root $DATA --shots 5 --device $DEVICE
  done
done

python scripts/make_tables.py --runs "$RUNS" --out tables
python scripts/qualitative.py --data-root $DATA --device $DEVICE --select median \
    --ckpt isic=$RUNS/W_isic/best.pt brats=$RUNS/W_brats/best.pt kvasir=$RUNS/W_kvasir/best.pt chase=$RUNS/W_chase/best.pt \
    --fixed isic=$RUNS/abl_fixed_stencils_isic/best.pt brats=$RUNS/abl_fixed_stencils_brats/best.pt \
            kvasir=$RUNS/abl_fixed_stencils_kvasir/best.pt chase=$RUNS/abl_fixed_stencils_chase/best.pt \
    --out figures/qualitative.pdf
