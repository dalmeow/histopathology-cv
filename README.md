# Histopathology CV

Tissue segmentation on the Computer Vision Mini Project dataset.  
3-class pixel-wise classification: **Tumor / Stroma / Other**.

## Results Summary

| Model | Best Avg Dice | Tumor | Stroma | Other |
|-------|--------------|-------|--------|-------|
| UNet exp5 (Focal + Lovász) | **0.463** | 0.840 | 0.404 | 0.147 |
| UNet exp5 + CRF | **0.502** | 0.841 | 0.512 | 0.154 |
| AE exp2 (MSE pretrain + full finetune) | 0.406 | 0.829 | 0.336 | 0.051 |
| Baseline (provided) | 0.467 | — | — | — |

## Repo Structure

```
histopathology-cv/
├── 1_shared/              # Shared data pipeline and losses
│   ├── data_processing.py # TissueDataset, GeoJSON→mask, HED augmentation
│   └── losses.py          # FocalLoss, LovászSoftmax, DiceLoss, build_criterion
│
├── 1a_unet/               # Task 1a — UNet end-to-end segmentation (6 experiments)
└── 1b_autoencoder/        # Task 1b — Autoencoder pre-training + SegDecoder (4 experiments)
```

See [1a_unet/README.md](1a_unet/README.md) and [1b_autoencoder/README.md](1b_autoencoder/README.md) for full ablation tables, architecture details, and usage.

## Data

Expected at `../Coumputer_Vision_Mini_Project_Data/Dataset_Splits/` relative to this repo, with `train/`, `validation/`, and `test/` splits each containing `image/` and `tissue/` subdirectories.

## Setup

```bash
pip install -r requirements.txt
```
