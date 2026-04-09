# Task 2b — Contrastive Pre-training

Four-experiment additive ablation for SimCLR-based self-supervised pre-training
followed by frozen-encoder supervised fine-tuning. Same 3-class nucleus task as 2a:
**Tumour / Lymphocyte / Histiocyte**.

## Results

All numbers are on the held-out test set with **TTA enabled** (8 views: hflip × vflip × rot90).

| Exp | Configuration | Params | Acc | Macro F1 |
|-----|--------------|--------|-----|----------|
| 1 | SimCLR + MLP head | ~2.8M | 0.658 | 0.645 |
| 2 | + K-means init | ~2.8M | 0.662 | 0.654 |
| 3 | − Mixup + moderate SimCLR aug | ~2.8M | 0.687 | 0.676 |
| 4 | ResNet-18 (ImageNet) SimCLR | ~11.2M | 0.664 | 0.654 |

### Per-class breakdown

| Exp | Tumour P / R / F1 | Lymphocyte P / R / F1 | Histiocyte P / R / F1 |
|-----|-------------------|----------------------|----------------------|
| 1 | 0.709 / 0.656 / 0.682 | 0.675 / 0.740 / 0.706 | 0.556 / 0.539 / 0.548 |
| 2 | 0.718 / 0.659 / 0.687 | 0.696 / 0.699 / 0.697 | 0.546 / 0.611 / 0.577 |
| 3 | 0.731 / 0.689 / 0.709 | 0.710 / 0.743 / 0.726 | 0.587 / 0.598 / 0.592 |
| 4 | 0.738 / 0.699 / 0.718 | 0.709 / 0.677 / 0.693 | 0.515 / 0.592 / 0.551 |

Provided baseline: **0.708 accuracy**. Best contrastive model (Exp 3): **0.687 accuracy** — below
the supervised baseline (Task 2a Exp 4: **0.759**), reflecting the cost of encoder freezing.

## Ablation Story

```
Exp 1 → Exp 2 : add K-means head initialisation (random → k-means centroid init)
Exp 2 → Exp 3 : swap strong → moderate SimCLR aug, drop Mixup    ← best model
Exp 3 → Exp 4 : swap NucleiResNet → ImageNet ResNet-18 backbone
```

K-means initialisation gives a small but consistent boost (+0.009 macro F1) by
warm-starting the classification head using cluster structure in the embedding space.
Moderate augmentation (Exp 3) outperforms strong augmentation: preserving H&E colour
signal during contrastive pre-training leads to more class-discriminative features.
ResNet-18 (Exp 4) underperforms NucleiResNet despite ImageNet weights — the ImageNet
features may be less suited to H&E histopathology when the encoder is frozen.

## Leakage-Safe Data Split

SimCLR pre-training uses a **dedicated contrastive patch set** — never `train/` or `validation/`:

- **Part A**: all nuclei from 20 training ROIs held out at the ROI level
- **Part B**: unused nuclei from the remaining training ROIs that are ≥100 px from any
  training centroid (cKDTree spatial buffer)
- UUID overlap with the supervised training set is **always zero** (verified before saving)

Extraction is handled by `2_data/extract_patches.py`. Output: `task2_patches/contrastive/`.

## Shared Pre-training Checkpoints

| Checkpoint | Experiments |
|-----------|------------|
| `checkpoints/pretrain/nuclresnet_strong/best.pth` | Exp 1, Exp 2 (shared) |
| `checkpoints/pretrain/nuclresnet_moderate/best.pth` | Exp 3 |
| `checkpoints/pretrain/resnet18/best.pth` | Exp 4 |

A pretrain run is skipped if its checkpoint already exists, so running Exp 1 and then
Exp 2 only trains the encoder once.

## Usage

```bash
# From the repo root (histopathology-cv/)

# Extract patches first (run once — builds train/, validation/, contrastive/):
python 2_data/extract_patches.py

# Run a single experiment (pretrain → [k-means init] → finetune → eval):
python 2b_contrastive/run.py --exp 1
python 2b_contrastive/run.py --exp 3

# Run all experiments:
for i in 1 2 3 4; do python 2b_contrastive/run.py --exp $i; done
```

Each stage is skipped if its checkpoint already exists.

## Files

| File | Purpose |
|------|---------|
| `run.py` | Experiment configs + 4-stage entry point |
| `pretrain.py` | SimCLR pre-training loop (NT-Xent, cosine schedule with warmup) |
| `finetune.py` | K-means head init + frozen-encoder supervised head training |
| `eval.py` | Test set evaluation with TTA (8 views) — JSON + TXT + confusion matrix |
| `model.py` | Re-exports NucleiResNet, ResNet18Encoder; adds SimCLRProjectionHead, NTXentLoss |
| `dataset.py` | SimCLRDataset, PlainDataset, `_SimCLRAug`, `_R18Aug` |
| `checkpoints/pretrain/` | Encoder checkpoints from SimCLR (gitignored) |
| `checkpoints/` | Fine-tuned head checkpoints (gitignored) |

## Architecture

### SimCLR Pre-training

```
Contrastive patches
  ↓ Two independently augmented views (view1, view2)
  ↓ Encoder (NucleiResNet or ResNet-18) → feature vector h ∈ ℝ^{256 or 512}
  ↓ Projection head: Linear(h_dim→256, no bias) → BN → ReLU → Linear(256→128)
  ↓ NT-Xent loss (temperature τ, L2-normalised embeddings)
```

The projection head is discarded after pre-training — only the encoder is kept.

### Supervised Head (all experiments)

```
Frozen encoder → h ∈ ℝ^{256 or 512}
  ↓ Linear(h_dim → h_dim//2) → ReLU → Dropout(0.3) → Linear(h_dim//2 → 3)
```

All encoder parameters are frozen. Only the head is trained.

### K-means Head Initialisation (Exp 2, 3)

1. Extract features from the full contrastive set using the frozen encoder.
2. Fit K-means (k=3) on those features.
3. Map clusters → class labels via majority vote on the labelled training set.
4. Initialise `head[0]` (Linear h_dim → h_dim//2) weights from PCA components of the
   cluster-projected features.
5. Initialise `head[3]` (Linear h_dim//2 → 3) weights from L2-normalised projected
   class centroids.

## Training Details

### SimCLR Pre-training

| Setting | NucleiResNet (Exp 1–3) | ResNet-18 (Exp 4) |
|---------|------------------------|-------------------|
| Batch size | 256 | 256 |
| Epochs | 500 max, early stop patience=30 | 150 max, early stop patience=30 |
| Optimiser | Adam (lr=3e-4, wd=1e-4) | Adam (lr=1e-3, wd=1e-4) |
| LR schedule | Linear warmup (10 epochs) → CosineAnnealing → 1e-6 | Linear warmup (10 epochs) → CosineAnnealing → 1e-6 |
| Temperature τ | 0.3 | 0.3 |
| Projection dim | 128 | 128 |
| Augmentation (strong) | ColorJitter(b=0.4,c=0.4,s=0.3,h=0.1,p=0.8) + ToGray(p=0.2) | — |
| Augmentation (moderate) | ColorJitter(b=0.2,c=0.2,s=0.15,h=0.05,p=0.5), no ToGray | — |
| Augmentation (R18) | — | ColorJitter(b=0.3,c=0.3,s=0.2,h=0.05,p=0.7), no ToGray |
| Common aug ops | RandomResizedCrop(100, 0.75–1.0) + hflip + vflip + rot90 + GaussianBlur + GaussNoise | same |

### Supervised Head Fine-tuning

| Setting | Exp 1, 2, 4 | Exp 3 |
|---------|-------------|-------|
| Optimiser | Adam (lr=1e-3, wd=5e-4) | Adam (lr=1e-3, wd=5e-4) |
| Scheduler | ReduceLROnPlateau (patience=8, factor=0.5, mode=max on val acc) | ReduceLROnPlateau (patience=8, factor=0.5, mode=max on val acc) |
| Batch size | 64 | 64 |
| Epochs | 100 max, early stop patience=15 | 100 max, early stop patience=15 |
| Augmentation | Mild: hflip + vflip + rot90 + ColorJitter(b=0.2,c=0.2,s=0.15,h=0.05,p=0.5) | Moderate: + RandomResizedCrop(0.75–1.0) + GaussianBlur + GaussNoise |
| Mixup | α=0.3 (Exp 1–2); disabled (Exp 4) | disabled |
| Loss | CrossEntropyLoss |
| Encoder | Fully frozen |
| Seed | 42 |
| TTA (eval) | 8 views |
