# Task 2a — End-to-End Nuclei Classification

Five-experiment additive ablation for 3-class nucleus classification:
**Tumour / Lymphocyte / Histiocyte** from 100×100 px patches.

## Results

All numbers are on the held-out test set with **TTA enabled** (8 views: hflip × vflip × rot90).

| Exp | Configuration | Params | Acc | Macro F1 |
|-----|--------------|--------|-----|----------|
| 1 | SimpleClassifier (naive) | ~94K | 0.607 | 0.588 |
| 2 | NucleiResNet base | ~2.8M | 0.742 | 0.732 |
| 3 | + ECA attention | ~2.8M | 0.733 | 0.722 |
| **4** | **+ Mixup** | **~2.8M** | **0.759** | **0.754** |
| 5 | ResNet-18 (ImageNet) | ~11.2M | 0.745 | 0.737 |

### Per-class breakdown

| Exp | Tumour P / R / F1 | Lymphocyte P / R / F1 | Histiocyte P / R / F1 |
|-----|-------------------|----------------------|----------------------|
| 1 | 0.741 / 0.753 / 0.747 | 0.610 / 0.557 / 0.583 | 0.413 / 0.459 / 0.435 |
| 2 | 0.752 / 0.819 / 0.784 | 0.825 / 0.716 / 0.767 | **0.624** / 0.666 / 0.644 |
| 3 | 0.742 / 0.817 / 0.778 | 0.858 / 0.691 / 0.766 | 0.583 / 0.666 / 0.622 |
| **4** | **0.829** / 0.767 / **0.797** | 0.833 / 0.743 / 0.786 | 0.604 / **0.773** / **0.678** |
| 5 | 0.791 / 0.750 / 0.770 | 0.813 / **0.763** / **0.787** | 0.607 / 0.712 / 0.655 |

Provided baseline: **0.708 accuracy**.

## Ablation Story

```
Exp 1 → Exp 2 : SimpleClassifier → NucleiResNet (+improved aug, cosine schedule)
Exp 2 → Exp 3 : + ECA channel attention in residual blocks
Exp 3 → Exp 4 : + Mixup (Beta(0.3, 0.3))                     ← best model
Exp 4 → Exp 5 : swap NucleiResNet → ImageNet ResNet-18
```

ECA adds negligible parameters but slightly hurts accuracy in this setting (exp3 < exp2), suggesting the residual architecture already captures enough channel interactions. Mixup gives the strongest boost (+0.026 accuracy, +0.032 macro F1). ResNet-18 with ImageNet weights matches NucleiResNet+Mixup but at 4× the parameter count.

## Usage

```bash
# From the repo root (histopathology-cv/)

# Extract patches first (run once):
python 2_data/extract_patches.py

# Run a single experiment (train + eval):
python 2a_classifier/run.py --exp 4

# Run all experiments sequentially:
for i in 1 2 3 4 5; do python 2a_classifier/run.py --exp $i; done
```

Training is skipped automatically if the best checkpoint already exists.

## Files

| File | Purpose |
|------|---------|
| `run.py` | Experiment configs + entry point |
| `train.py` | Training loop (ReduceLROnPlateau / CosineAnnealing, Mixup, early stopping) |
| `eval.py` | Test set evaluation with TTA (8 views) — JSON + TXT + confusion matrix |
| `model.py` | SimpleClassifier, NucleiResNet (+ ECA), ResNet18Encoder |
| `dataset.py` | NucleiDataset (train/val aug), TestDataset, CLASS_NAMES |
| `checkpoints/` | Saved model weights (gitignored) |

## Architecture

### NucleiResNet (Exp 2–4)

```
Input (3, 100, 100)
  ↓ Stem: Conv(3→64, 7×7 stride2) → BN → ReLU → MaxPool(3, stride2)   → (64, 25, 25)
  ↓ Stage 1: 2× ResBlock(64→64)                                          → (64, 25, 25)
  ↓ Stage 2: 2× ResBlock(64→128, stride2)                               → (128, 13, 13)
  ↓ Stage 3: 2× ResBlock(128→256, stride2)                              → (256,  7,  7)
  ↓ GAP → Dropout(0.3) → Linear(256→3)
```

Each ResBlock: Conv-BN-ReLU-Conv-BN + optional ECA + 1×1 shortcut when dims change.
ECA kernel size is derived from channel count: k = nearest odd to |log₂(C)/2 + 0.5|.

### ResNet-18 (Exp 5)

ImageNet-pretrained backbone (torchvision `ResNet18_Weights.IMAGENET1K_V1`) with the
final FC replaced by `Linear(512→3)`. Full end-to-end fine-tuning.

## Training Details

| Setting | Exp 1 | Exp 2–5 |
|---------|-------|---------|
| Optimiser | Adam (lr=1e-3, wd=1e-4) | Adam (lr=1e-3, wd=5e-4) |
| Scheduler | ReduceLROnPlateau (patience=8, factor=0.5) | CosineAnnealingWarmRestarts (T₀=20, T_mult=2, η_min=1e-6) |
| Batch size | 64 | 64 |
| Epochs | 100 max, early stop patience=15 | 100 max, early stop patience=15 |
| Augmentation | Baseline (hflip + vflip) | Improved (rot90 + Affine + HED + ColorJitter + GaussianBlur + GaussNoise + CoarseDropout) |
| Mixup | — | α=0.3 (Exp 4–5 only) |
| Loss | CrossEntropyLoss | FocalLoss (γ=2) |
| Seed | 42 | 42 |
| TTA (eval) | 8 views | 8 views |
