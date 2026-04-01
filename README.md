# Histopathology CV

Histopathology image analysis on the Computer Vision Mini Project dataset.

## Tasks

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Tissue segmentation (Tumor / Stroma / Other) | Done — 0.416 avg Dice |
| Task 2 | Nuclei classification (Tumor / Lymphocyte / Histiocyte) | In progress |

## Repo Structure

```
histopathology-cv/
├── 1_shared/                 # Shared data pipeline and losses
│   ├── data_processing.py    # TissueDataset, HED augmentation, TTA, sampling weights
│   └── losses.py             # FocalLoss, DiceLoss, build_primary_criterion
│
├── 0_data_exploration/       # Dataset analysis (not needed for training)
│   ├── task1.py
│   ├── task2.py
│   └── output/
│       ├── task1/
│       └── task2/
│
├── 1a_unet/                  # Task 1 — UNet segmentation
└── 1b_autoencoder/           # Task 1 — Autoencoder pre-training experiments
```

## Data

Expected at `../Coumputer_Vision_Mini_Project_Data/Dataset_Splits/` relative to this directory, with `train/`, `validation/`, and `test/` splits each containing `image/` and `tissue/` subdirectories.

## Setup

```bash
pip install -r requirements.txt
```
