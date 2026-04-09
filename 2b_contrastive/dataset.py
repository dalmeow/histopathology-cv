"""
dataset.py — 2b_contrastive

SimCLRDataset  : two-view unlabelled dataset for SimCLR pre-training.
PlainDataset   : single-view (normalise-only) for k-means feature extraction.
_SimCLRAug     : NucleiResNet augmentation (strong / moderate).
_R18Aug        : ResNet-18 augmentation.

NucleiDataset and TestDataset are imported directly from 2a_classifier in
finetune.py and eval.py — no re-export needed here.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

NORMALISE_MEAN = (0.2237, 0.2237, 0.2237)
NORMALISE_STD  = (0.3384, 0.2318, 0.3410)


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

class _SimCLRAug:
    """
    Two independently augmented views for NucleiResNet SimCLR pre-training.

    color_strength:
      "strong"   — original SimCLR: ColorJitter(b=0.4,c=0.4,s=0.3,h=0.1,p=0.8)
                   + ToGray(p=0.2). Encoder learns to ignore colour.
      "moderate" — histopathology-aware: ColorJitter(b=0.2,c=0.2,s=0.15,h=0.05,p=0.5),
                   no ToGray. Preserves H&E stain signal.
    """

    def __init__(self, color_strength: str = "strong"):
        if color_strength == "strong":
            color_ops = [
                A.ColorJitter(
                    brightness=0.4, contrast=0.4,
                    saturation=0.3, hue=0.1, p=0.8,
                ),
                A.ToGray(p=0.2),
            ]
        else:  # moderate
            color_ops = [
                A.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.15, hue=0.05, p=0.5,
                ),
            ]

        self._aug = A.Compose([
            A.RandomResizedCrop(
                size=(100, 100), scale=(0.75, 1.0), ratio=(0.9, 1.1), p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            *color_ops,
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.GaussNoise(std_range=(0.02, 0.11), p=0.2),
            A.Normalize(mean=list(NORMALISE_MEAN), std=list(NORMALISE_STD)),
            ToTensorV2(),
        ])

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        return self._aug(image=img)["image"]


class _R18Aug:
    """
    Two independently augmented views for ResNet-18 SimCLR pre-training.

    No ToGray (H&E colour is discriminative for nuclei classes).
    Moderate colour jitter, tighter crop scale.
    """

    def __init__(self):
        self._aug = A.Compose([
            A.RandomResizedCrop(
                size=(100, 100), scale=(0.75, 1.0), ratio=(0.9, 1.1), p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.2, hue=0.05, p=0.7,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
            A.Normalize(mean=list(NORMALISE_MEAN), std=list(NORMALISE_STD)),
            ToTensorV2(),
        ])

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        return self._aug(image=img)["image"]


class _NormaliseOnly:
    """Normalise-only transform — used for k-means feature extraction."""

    def __init__(self):
        self._aug = A.Compose([
            A.Normalize(mean=list(NORMALISE_MEAN), std=list(NORMALISE_STD)),
            ToTensorV2(),
        ])

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        return self._aug(image=img)["image"]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _collect_npy(patch_dirs: list[str]) -> list[str]:
    """Recursively collect all .npy files from a list of directories."""
    paths = []
    for d in patch_dirs:
        paths.extend(sorted(Path(d).rglob("*.npy")))
    return [str(p) for p in paths]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SimCLRDataset(Dataset):
    """
    Unlabelled two-view dataset for SimCLR pre-training.

    Collects all .npy patches from patch_dirs (contrastive/ set only —
    never train/ or validation/ to prevent gradient leakage).
    Each __getitem__ returns (view1, view2): two independently augmented
    views of the same patch.
    """

    def __init__(self, patch_dirs: list[str], aug):
        self.paths = _collect_npy(patch_dirs)
        self._aug  = aug
        assert len(self.paths) > 0, f"No .npy files found in {patch_dirs}"

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        arr   = np.load(self.paths[idx])   # uint8 (100, 100, 3)
        view1 = self._aug(arr)
        view2 = self._aug(arr)
        return view1, view2


class PlainDataset(Dataset):
    """
    Single-view unlabelled dataset (normalise only, no augmentation).
    Used for k-means feature extraction in kmeans_init().
    """

    def __init__(self, patch_dirs: list[str]):
        self.paths = _collect_npy(patch_dirs)
        self._norm = _NormaliseOnly()
        assert len(self.paths) > 0, f"No .npy files found in {patch_dirs}"

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._norm(np.load(self.paths[idx]))
