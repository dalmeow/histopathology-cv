# Autoencoder — Tissue Segmentation Ablation

Self-supervised pre-training approach: an encoder is first trained via image
reconstruction, then the decoder is replaced with a segmentation head (SegDecoder)
and fine-tuned with the encoder frozen.

## Results

All numbers are **best-checkpoint Dice** on the held-out test set (per-image, absent classes skipped).

| Exp | Pretrain | Finetune | Tumor | Stroma | Other | Avg Dice |
|-----|----------|----------|-------|--------|-------|----------|
| exp1 — baseline | vanilla MSE (BatchNorm) | vanilla CE | 0.850 | 0.276 | 0.001 | 0.375 |
| **exp2 — full finetune** | vanilla MSE (InstanceNorm) | full | **0.829** | **0.336** | **0.051** | **0.406** |
| exp3 — masked pretrain | masked MAE (BatchNorm) | vanilla CE | 0.812 | 0.254 | 0.000 | 0.356 |
| exp4 — masked + full | masked MAE (InstanceNorm) | full | 0.773 | 0.288 | 0.062 | 0.374 |

## Ablation Story

```
exp1  →  exp2  : finetune quality only (same vanilla MSE pretrain, upgrade finetune)
exp1  →  exp3  : pretrain strategy only (same vanilla CE finetune, add masking)
exp2 + exp3  →  exp4  : both combined
```

Key findings:
- The full finetune pipeline (exp2) is the main driver — +0.031 avg Dice over baseline
- Masked pretraining **hurts** when paired with vanilla CE (exp3 < exp1)
- Masked pretraining also **hurts** when paired with full finetune (exp4 < exp2)
- The reconstruction pre-training strategy adds limited value over a good finetune pipeline

## Usage

```bash
# From the repo root (histopathology-cv/)

# Run a single experiment (pretrain + finetune + eval):
python 1b_autoencoder/run.py --exp 2

# Submit all experiments as parallel SLURM jobs:
bash 1b_autoencoder/submit.sh 1 2 3 4
```

Pre-training is skipped automatically if a checkpoint already exists for that
pretrain config — experiments sharing a pretrain checkpoint only train it once.

## Files

| File | Purpose |
|------|---------|
| `run.py` | Experiment configs + entry point |
| `train_ae.py` | Autoencoder pre-training loop (vanilla MSE or masked MAE) |
| `finetune.py` | SegDecoder fine-tuning with frozen encoder |
| `eval.py` | Test set evaluation — quantitative + qualitative |
| `arch/autoencoder.py` | Encoder, ReconDecoder, SegDecoder, Autoencoder |
| `slurm_run.sh` | SLURM job script |
| `submit.sh` | Submits one job per experiment in parallel |
| `checkpoints/` | Pretrain + finetune checkpoints (gitignored) |
| `results/` | Per-experiment metrics.json + qualitative grids |

## Architecture

```
Encoder (shared, frozen during finetune):
  Input (3, 512, 512)
  ↓ DoubleConv(3 → 64)       [skip x1]
  ↓ DownLayer(64 → 128)      [skip x2]
  ↓ DownLayer(128 → 256)     [skip x3]
  ↓ DownLayer(256 → 512)     [skip x4]
  ↓ Dropout + DownLayer(512 → 1024)   [bottleneck x5]

ReconDecoder (pre-training only — no skip connections):
  x5 → 4× ConvTranspose2d + DoubleConv → Conv2d(→3) + Sigmoid

SegDecoder (fine-tuning — skip connections from frozen encoder):
  x5 + x4 → ResidualUpLayer(1024 → 512)  → aux head
  x3      → ResidualUpLayer(512 → 256)   → aux head
  x2      → ResidualUpLayer(256 → 128)   → aux head
  x1      → ResidualUpLayer(128 → 64)
           → Conv2d(64 → 3)              [output logits]
```

ResidualUpLayer and deep supervision are only used in exp2/exp4 (full finetune).
exp1/exp3 use plain UpLayer, no deep supervision, no dropout.

## Pre-training Details

**Vanilla MSE** (`pretrain_masked=False`): full image reconstructed from full input.
Loss: MSE over all pixels.

**Masked MAE** (`pretrain_masked=True`): 75% of 40×40 patches zeroed out, model
reconstructs masked regions only. Loss: MSE + 0.5×(1−SSIM) on masked patches.

## Training Details (all experiments)

| Setting | Value |
|---------|-------|
| Optimiser | AdamW (lr=1e-4, wd=1e-2) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=10, mode=min) |
| Epochs | 100 max pretrain + 100 max finetune, early stop patience=25 on val loss |
| Batch size | 8 |
| Image size | 512×512 |
| Seed | 42 |
| Finetune | Encoder frozen; only SegDecoder parameters optimised |
| Augmentation (exp2/4) | rot90 + hflip + vflip + HED stain jitter (±0.2) + brightness ±10% |
