# UNet — Tissue Segmentation Ablation

5-level UNet for 3-class histopathology segmentation: **Tumor / Stroma / Other**.

## Results

All numbers are **best-checkpoint Dice** on the held-out test set (per-image, absent classes skipped).

| Exp | What changes | Tumor | Stroma | Other | Avg Dice |
|-----|-------------|-------|--------|-------|----------|
| exp1 — baseline | BatchNorm, plain CE | 0.837 | 0.337 | 0.000 | 0.391 |
| exp2 — domain | + InstanceNorm, HED stain aug | 0.837 | 0.387 | 0.055 | 0.426 |
| exp3a — loss balance | + sqrt class weights + Dice loss | 0.838 | 0.346 | 0.093 | 0.426 |
| exp3b — oversampling | data-level balance (negative result) | 0.835 | 0.406 | 0.048 | 0.430 |
| exp4 — architecture | + residual decoder, deep sup, dropout | 0.840 | 0.352 | 0.018 | 0.403 |
| **exp5 — Lovász** | + Focal + Lovász-Softmax loss | **0.840** | **0.404** | **0.147** | **0.463** |
| exp6 — CRF | dense CRF post-processing on exp5 | 0.841 | 0.512 | 0.154 | **0.502** |

CRF post-processing on exp5 best checkpoint: **avg Dice 0.502** (Stroma +0.108, Other +0.007).

## Ablation Story

```
exp1  →  exp2   : domain adaptation only (norm + augmentation)
exp2  →  exp3a  : class imbalance via loss (weighted CE + Dice)
exp2  →  exp3b  : class imbalance via data (oversampling) — worse than exp3a
exp3a →  exp4   : architecture (residual decoder + deep supervision + dropout)
exp4  →  exp5   : loss (Focal + Lovász replaces weighted CE + Dice)
exp5  →  exp6   : inference-time CRF post-processing (no retraining)
```

Each step changes exactly one axis. exp3b is kept as a negative result: the
oversampled images were white-background-heavy slides, not informative Other
(blood vessels / epidermis), so the model learned to predict Other on blank areas.

## Usage

```bash
# From the repo root (histopathology-cv/)

# Run a single experiment (train + eval):
python 1a_unet/run.py --exp 5

# Run CRF post-processing on exp5 checkpoint (exp6):
python 1a_unet/run.py --exp 6

# Submit all experiments as parallel SLURM jobs:
bash 1a_unet/submit.sh 1 2 3a 3b 4 5 6
```

## Files

| File | Purpose |
|------|---------|
| `run.py` | Experiment configs + entry point |
| `train.py` | Training loop (W&B logging, early stopping) |
| `eval.py` | Test set evaluation — quantitative + qualitative |
| `eval_crf.py` | Dense CRF post-processing on a saved checkpoint |
| `colab_run.ipynb` | Colab notebook runner for the UNet experiments |
| `arch/unet.py` | Model architecture |
| `slurm_run.sh` | SLURM job script |
| `submit.sh` | Submits one job per experiment in parallel |
| `experiments.md` | Raw log of all pre-ablation exploratory runs |
| `checkpoints/` | Saved model weights (gitignored) |
| `results/` | Per-experiment metrics.json + qualitative grids |

## Architecture (exp5 / final)

```
Input (3, 512, 512)
  ↓ DoubleConv(3 → 64)                           [skip x1]
  ↓ DownLayer(64 → 128)                          [skip x2]
  ↓ DownLayer(128 → 256)                         [skip x3]
  ↓ DownLayer(256 → 512)                         [skip x4]
  ↓ Dropout + DownLayer(512 → 1024)              [bottleneck]
  ↓ Dropout + ResidualUpLayer(1024 → 512)  → aux head (×0.5)
  ↓ Dropout + ResidualUpLayer(512 → 256)   → aux head (×0.25)
  ↓ ResidualUpLayer(256 → 128)             → aux head (×0.125)
  ↓ ResidualUpLayer(128 → 64)
  ↓ Conv2d(64 → 3)                               [output logits]
```

- **Norm:** InstanceNorm2d (affine=True) throughout
- **Residual decoder:** 1×1 conv shortcut from concatenated skip+up features to block output
- **Deep supervision:** auxiliary CE heads weighted 0.5 / 0.25 / 0.125 (training only)
- **Dropout2d** p=0.3 at bottleneck and first two decoder stages
- ~31M parameters (base=64)

## Training Details (all experiments)

| Setting | Value |
|---------|-------|
| Optimiser | AdamW (lr=1e-4, wd=1e-2) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=10, mode=min) |
| Epochs | 100 max, early stop patience=25 on val loss |
| Batch size | 8 |
| Image size | 512×512 |
| Seed | 42 |
| Augmentation (exp2+) | rot90 + hflip + vflip + HED stain jitter (±0.2) + brightness ±10% |
