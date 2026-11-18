# TaskLift

**Few-shot medical image segmentation as closed-form inverse lifting.**

TaskLift segments a query image from one or a few annotated support images. It has no backbone and no pretraining, and it takes no gradient steps at test time.

1. **Analysis.** The image is analysed by a meta-learned multiresolution **lifting cascade**.
2. **Compilation.** Every task-specific parameter is **compiled from the support set in closed form**: 2J + 1 = 9 ridge solves.
3. **Synthesis.** The mask is **synthesised coarse to fine by the exact inverse of label-side predict shears**.

The default model has **7,810 meta-parameters** and **2,827 compiled parameters per episode**.

This repository implements the method, the training and evaluation protocol, all ablations, and a numerical audit of every theorem in the paper.

---

## Contents

```
tasklift/
  stencils.py      canonical sum-rule stencils a*_r and moments                (Sec. II-B)
  lifting.py       lifting step (1), 2-D cascade, exact inverse, constant B     (Lemma 1, Prop. 1)
  features.py      3x3 lattice gathers, projections, polynomial map psi        (Sec. III-A)
  compiler.py      class-balanced ridge compiler, Cholesky, fp64 promotion      (Eq. (3), Thm. 1)
  calibration.py   2-parameter logistic calibration, damped Newton              (Sec. III-C)
  model.py         TaskLift: analysis -> compile -> inverse-shear sweep         (Secs. II-IV, Thm. 2)
  losses.py        soft-Dice + BCE, R1, R2                                       (Sec. III-C, Cor. 1)
  metrics.py       Dice, IoU, ASSD (4-connected boundaries), episode bootstrap
  engine.py        meta-training with early stopping; evaluation, tertiles, support variance
  verification.py  float64 audit of every structural statement                  (Table V)
  data/            preprocessing, splits, episodes (incl. R-CHASE)
scripts/
  prepare_data.py  make_synthetic_data.py  train.py  evaluate.py  verify.py
  benchmark.py     qualitative.py          make_tables.py          run_paper.sh
configs/default.yaml, configs/ablations/*.yaml
tests/             27 tests: theory, components, data pipeline
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"            # or: pip install -r requirements.txt
```

PyTorch ≥ 2.1 is required. A CUDA GPU is recommended for meta-training; inference also runs on a CPU.

## Five-minute check (no data needed)

```bash
pytest -q                                   # 27 tests
python scripts/verify.py                    # numerical audit (Table V), ~10 s on a laptop CPU

# full pipeline on synthetic data in the real on-disk layouts
python scripts/make_synthetic_data.py --out /tmp/raw
for d in isic kvasir chase brats; do
  python scripts/prepare_data.py --dataset $d --root /tmp/raw/$d --out /tmp/proc --size 64
done
python scripts/train.py --set image_size=64 levels=3 data_root=/tmp/proc dataset=isic \
    out_dir=/tmp/runs/W_isic max_episodes=40 val_every=20 val_episodes=4 n_query=4
python scripts/evaluate.py --ckpt /tmp/runs/W_isic/best.pt --dataset isic --data-root /tmp/proc \
    --shots 1 5 --queries 1 --episodes 20
```

## Data

All four datasets are public. Download them yourself and point `--root` at the extracted folder. The scanners search recursively.

| Dataset | Source | Expected files |
|---|---|---|
| ISIC 2018 Task 1 (training release, 2,594 images) | challenge.isic-archive.com | `ISIC_xxxxxxx.jpg`, `ISIC_xxxxxxx_segmentation.png` |
| BraTS 2021 (1,251 training cases) | RSNA-ASNR-MICCAI BraTS 2021 (Synapse / TCIA) | `BraTS2021_xxxxx/BraTS2021_xxxxx_{t1,t1ce,t2,flair,seg}.nii.gz` |
| Kvasir-SEG (1,000 images) | datasets.simula.no/kvasir-seg | `images/<id>.jpg`, `masks/<id>.jpg` |
| CHASE_DB1 (28 images) | Kingston University research data | `Image_XXL.jpg`, `Image_XXL_1stHO.png` |

```bash
python scripts/prepare_data.py --dataset isic   --root /data/ISIC2018 --out data/processed
python scripts/prepare_data.py --dataset brats  --root /data/BraTS2021_Training_Data --out data/processed
python scripts/prepare_data.py --dataset kvasir --root /data/Kvasir-SEG --out data/processed
python scripts/prepare_data.py --dataset chase  --root /data/CHASE_DB1 --out data/processed
```

Preprocessing follows the paper and is fully deterministic:

- **Resizing.** Images are resized to 256 × 256 with bicubic interpolation. Masks are resized bilinearly and re-thresholded at 0.5.
- **Normalisation.** Each image is z-scored per channel.
- **Channels.** RGB images are zero-padded to four channels.
- **BraTS.** Labels are binary whole tumour. The four channels are T1, T1c, T2 and FLAIR. Each case contributes at most 96 non-empty axial slices, chosen with an even stride.

Splits are 60/20/20:

- by case for BraTS;
- by image for ISIC and Kvasir-SEG;
- 16/6/6 for CHASE_DB1, whose 6-image test set defines **R-CHASE**.

## Reproducing the paper

`scripts/run_paper.sh` runs every experiment in order. The individual steps:

| Paper item | Command |
|---|---|
| Table I, within (W) | `train.py --set protocol=within dataset=<d> out_dir=runs/W_<d>` then `evaluate.py --ckpt runs/W_<d>/best.pt --dataset <d> --shots 1 5` |
| Table I, leave-one-out (X) | `train.py --set protocol=cross dataset=<held-out> out_dir=runs/X_<d>` then `evaluate.py ... --shots 1 5` |
| Fig. 2 (shot scaling) | `evaluate.py --ckpt runs/W_<d>/best.pt --dataset <d> --shots 1 2 3 5 10` (R-CHASE stops at 5) |
| Table III (ablations) | `train.py --ablation configs/ablations/<name>.yaml ...` (13 variants) |
| Table IV (tertiles) | written automatically by `evaluate.py` (`summary.tertiles`) |
| Support-set SD | `evaluate.py ... --support-variance` |
| Mean-support-mask control | `evaluate.py --baseline mean_mask --dataset <d> --shots 1 5` |
| Fig. 3 (qualitative) | `qualitative.py --ckpt d=path ... --fixed d=path ... --select median` (prints Dice on every panel) |
| Table V (verification) | `verify.py` |
| Table V (cost, hardware) | `benchmark.py --device cuda` / `--device cpu` (records GPU/CPU names) |
| LaTeX tables | `make_tables.py --runs runs --out tables` |

`evaluate.py` writes, for each dataset and shot count:

- the mean and the 95% bootstrap half-width (10,000 resamples over episodes) for Dice, IoU and ASSD;
- per-episode and per-query records;
- the per-episode amplification ∏ρ̃ⱼ of Theorem 2(c);
- timing.

## Paper ↔ code

| Paper | Code |
|---|---|
| Lifting step (1), Lemma 1 | `lifting.lift`, `lifting.unlift` (accept arbitrary nonlinear `predict`/`update`) |
| Prop. 1, constant B | `lifting.stability_constant`; checked by `verification.check_prop1_bound` |
| 2-D cascade, condition ≤ B^{4J} | `lifting.analysis`, `lifting.synthesis`; `TaskLift.analysis_constant` |
| a\*₄ = (−1, 9, 9, −1)/16 | `stencils.canonical_predictor(4)` (exact rational solve) |
| Label shears (2) | `TaskLift.split_level`, `TaskLift.label_details` |
| Features φ ∈ ℝ³³⁴, φ⁽⁰⁾ ∈ ℝ¹⁵³ | `TaskLift.stage_inputs`, `features.psi_rows` |
| Compiler (3), Theorem 1 | `compiler.accumulate_normal_equations`, `compiler.ridge_solve` (exact autograd through Cholesky) |
| Calibration | `calibration.fit_calibration` (detached, damped Newton) |
| Sweep, Theorem 2 | `TaskLift.sweep` (`root=` and `details=` arguments expose Theorem 2(a), (c)) |
| Corollary 1, R₂ | `Episode.label_l1`, `Episode.amplification`, `losses.r2_amplification_penalty` |
| Prop. 2 (equivariance) | periodic `torch.roll` throughout; `verification.check_equivariance` |

## Defaults

The defaults follow the paper and the technical document (Appendix C):

| Setting | Value |
|---|---|
| Image size N | 256 |
| Levels J | 4 |
| Mixed channels C′ | 4 |
| Predictor taps | 4, on offsets {−1, 0, 1, 2}, initialised at a\*₄ |
| Update | single tap ½ |
| Projection width | 24 (16 at the root) |
| Feature degree | 2 |
| Prior | Γ = I, λ = softplus(·) initialised to 1, μ = 0 |
| Class-balance ε | 10⁻³ |
| Calibration | η_c = 10⁻², 10 Newton steps |
| Loss weights (η₁, η₂, η₃) | (10⁻², 10⁻¹, 10⁻⁴) |
| Optimiser | AdamW, 10⁻³ → 10⁻⁵ cosine, weight decay 10⁻⁴ |
| Gradient clipping | 1.0 |
| Episodes | 4 per batch, at most 6 × 10⁴ |
| Validation | every 500 episodes, patience 10 |

## Implementation choices where the paper leaves room

These choices are explicit and configurable:

- **Kernel compilation (technical document, M2b) is not implemented.** At the default r = M_t = 4, the sum-rule constraint matrix is an invertible 4 × 4 Vandermonde matrix. Its null-space projector is therefore zero, so episodic kernel corrections vanish identically. The paper describes the method without it, and the parameter count (7,810) excludes it.
- **Training shots.** Each training episode draws n_s uniformly from `train_shots` (default {1, 5}), with n_q = 8 queries.
- **Distinct cases.** By default, the images of one episode come from distinct cases or patients (`distinct_groups`), which prevents same-volume leakage in BraTS. Pass `--no-distinct-groups` to `evaluate.py` to relax this.
- **Cross-dataset protocol.** Meta-training uses the train splits of the three source datasets, with episodes drawn uniformly over datasets. Early stopping uses their validation splits. Testing uses the held-out dataset's test split, with R-CHASE for CHASE_DB1.
- **Undefined ASSD.** When exactly one of prediction and ground truth is empty, ASSD is set to the image diagonal (362 px at 256²). Empty predictions score 0 Dice and 0 IoU.
- **Numerics.** Each stage solve is promoted to float64 when its condition estimate exceeds 10⁸. `checkpoint_gram: true` recomputes feature rows in the backward pass to save memory.
- **Calibration gradients.** (γ, β) are fitted on swept support scores and detached; meta-gradients flow through the query scores.

## Verification

`python scripts/verify.py` measured the following with this release (float64):

| Check | Result |
|---|---|
| Lemma 1, 1-D, nonlinear (tanh-biased) P, U | max error 4.4 × 10⁻¹⁵ |
| Lemma 1, 2-D cascade, kernels near a\* | max error 1.4 × 10⁻¹⁵ |
| Lemma 1, 2-D cascade, N(0,1) kernels | max error 4.5 × 10⁻¹¹ (ill-conditioned, as Prop. 1 predicts) |
| Prop. 1, both directions, 500 N(0,1) pairs | max ratio to B: 1.000; median 0.576. **The bound is sharp.** |
| Sum rules of canonical stencils | exact |
| Closed form (3) vs least squares | 1.2 × 10⁻¹⁵ |
| Theorem 1(b) vs finite differences | 1.4 × 10⁻¹⁰ |
| Theorem 1(c), Theorem 2(c) bounds | hold (max ratios 6 × 10⁻⁴, 0.26) |
| Theorem 2(a) exactness, 2(b) affinity | 1.1 × 10⁻¹⁶, 2.2 × 10⁻¹⁶ |
| Prop. 2 equivariance (float64 and float32) | exact (0) |
| Determinism | bit-identical |
| Parameter count | 7,810 / 2,827 |

## Reproducibility

- **Seeds and determinism.** Every run fixes its seeds and enables deterministic algorithms. Episodes are generated from fixed seeds, and the compiler has no stochastic step. On fixed hardware and software, repeated runs give identical outputs.
- **Hardware and timing.** Wall-clock figures depend on hardware; `benchmark.py` records the GPU, CPU, thread count and PyTorch version alongside the timings.

## Citation

```bibtex
@article{tasklift,
  title   = {TaskLift: Few-Shot Medical Image Segmentation as Closed-Form Inverse Lifting},
  author  = {Raj, Gaurav and Choudhury, Tanupriya and Sharma, Aviral and Sar, Ayan},
  journal = {IEEE Signal Processing Letters},
  year    = {2026},
  note    = {under review}
}
```

Add a `LICENSE` file of your choice before publishing the repository.
