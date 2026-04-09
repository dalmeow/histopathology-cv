"""
Data loading for Task 2a: Nuclei patch classification.

Two dataset classes:
  NucleiDataset — supervised train / validation splits
                  (patches stored in class sub-directories as .npy files)
  TestDataset   — flat test directory; parses class and source type from filenames

Augmentation pipelines:
  baseline  — hflip + vflip + ToTensor + Normalize  (torchvision); 2a Exp 1
  mild      — flips + rot90 + colour jitter only; 2b Exps 1, 2, 4 frozen-head fine-tuning
  moderate  — SimCLR-style single-view (RandomResizedCrop + flips + colour jitter + blur + noise);
              2b Exp 3 frozen-head fine-tuning (replaces Mixup as regulariser)
  improved  — rich albumentations pipeline (HED, colour jitter, affine, blur, dropout);
              used for 2a end-to-end training
  none      — normalise only (val / test)
"""

import os
import glob

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------------------------------------------------------
# Constants (no config.py dependency)
# ---------------------------------------------------------------------------

CLASS_MAP          = {"Tumor": 0, "Lymphocyte": 1, "Histiocyte": 2}
FILENAME_CLASS_MAP = {"tumor": 0, "lymphocyte": 1, "histiocyte": 2}
CLASS_NAMES        = ["Tumor", "Lymphocyte", "Histiocyte"]
NUM_CLASSES        = 3

NORMALISE_MEAN = (0.2237, 0.2237, 0.2237)
NORMALISE_STD  = (0.3384, 0.2318, 0.3410)

FLIP_PROB                = 0.5
HED_ALPHA                = 0.2
HED_BETA                 = 0.2
COLOUR_JITTER_BRIGHTNESS = 0.2
COLOUR_JITTER_CONTRAST   = 0.2
COLOUR_JITTER_SATURATION = 0.15
COLOUR_JITTER_HUE        = 0.05


# ===========================================================================
# Transforms — baseline (torchvision)
# ===========================================================================

def _make_train_transform_baseline() -> T.Compose:
    return T.Compose([
        T.RandomHorizontalFlip(p=FLIP_PROB),
        T.RandomVerticalFlip(p=FLIP_PROB),
        T.ToTensor(),
        T.Normalize(NORMALISE_MEAN, NORMALISE_STD),
    ])


def _make_val_test_transform() -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Normalize(NORMALISE_MEAN, NORMALISE_STD),
    ])


# ===========================================================================
# Transforms — improved (albumentations)
# ===========================================================================

class _HEDAug(A.ImageOnlyTransform):
    """
    HED stain augmentation (Tellez et al., 2019).
    RGB → OD → HED → per-channel multiplicative + additive jitter → RGB.
    """

    def __init__(self, alpha: float = HED_ALPHA, beta: float = HED_BETA, p: float = 0.5):
        super().__init__(p=p)
        self.alpha = alpha
        self.beta  = beta

    _RGB2HED = np.array([
        [ 1.87798274, -1.00767869, -0.55611582],
        [-0.06590806,  1.13473037, -0.1355218 ],
        [ 0.06099941, -0.10300637,  0.54971649],
    ])
    _HED2RGB = np.linalg.inv(_RGB2HED)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img_f = np.clip(img.astype(np.float32) / 255.0, 1e-6, 1.0)
        od    = -np.log(img_f)
        hed   = od @ self._RGB2HED.T
        alpha = np.random.uniform(1 - self.alpha, 1 + self.alpha, 3)
        beta  = np.random.uniform(-self.beta, self.beta, 3)
        hed   = hed * alpha + beta
        img_f2 = np.clip(np.exp(-(hed @ self._HED2RGB.T)), 0.0, 1.0)
        return (img_f2 * 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("alpha", "beta")


def _make_train_transform_mild() -> A.Compose:
    """
    Mild augmentation for 2b frozen-head fine-tuning (Exp 1, 2, 4).
    Flips + rot90 + colour jitter only — intentionally weaker than SimCLR pre-training
    to avoid disrupting frozen encoder representations.
    Values: brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05, p=0.5.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(
            brightness=COLOUR_JITTER_BRIGHTNESS,
            contrast=COLOUR_JITTER_CONTRAST,
            saturation=COLOUR_JITTER_SATURATION,
            hue=COLOUR_JITTER_HUE,
            p=0.5,
        ),
        A.Normalize(mean=list(NORMALISE_MEAN), std=list(NORMALISE_STD)),
        ToTensorV2(),
    ])


def _make_train_transform_moderate() -> A.Compose:
    """
    SimCLR-style single-view augmentation for 2b Exp 3 frozen-encoder fine-tuning.
    Replaces Mixup as the regularisation strategy: random crop + flips + colour jitter
    + blur + noise mirrors the contrastive pre-training view pipeline (single view).
    """
    return A.Compose([
        A.RandomResizedCrop(
            size=(100, 100), scale=(0.75, 1.0), ratio=(0.9, 1.1), p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(
            brightness=COLOUR_JITTER_BRIGHTNESS,
            contrast=COLOUR_JITTER_CONTRAST,
            saturation=COLOUR_JITTER_SATURATION,
            hue=COLOUR_JITTER_HUE,
            p=0.5,
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.GaussNoise(std_range=(0.02, 0.11), p=0.2),
        A.Normalize(mean=list(NORMALISE_MEAN), std=list(NORMALISE_STD)),
        ToTensorV2(),
    ])


def _make_train_transform_improved() -> A.Compose:
    """
    Albumentations pipeline for NucleiResNet / ResNet-18 training.

    Design rationale:
      - Full rotation (±180°): nuclei are orientation-invariant
      - Small scale (0.9–1.1): Lymphocyte/Histiocyte distinguished by size —
        aggressive rescaling would destroy that signal
      - GaussianBlur: mimics focus variation in whole-slide imaging
      - CoarseDropout: forces the model to use the whole patch
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-180, 180),
            p=0.6,
        ),
        _HEDAug(alpha=HED_ALPHA, beta=HED_BETA, p=0.5),
        A.ColorJitter(
            brightness=COLOUR_JITTER_BRIGHTNESS,
            contrast=COLOUR_JITTER_CONTRAST,
            saturation=COLOUR_JITTER_SATURATION,
            hue=COLOUR_JITTER_HUE,
            p=0.5,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(std_range=(0.02, 0.11), p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(4, 10),
            hole_width_range=(4, 10),
            p=0.2,
        ),
        A.Normalize(mean=list(NORMALISE_MEAN), std=list(NORMALISE_STD)),
        ToTensorV2(),
    ])


# ===========================================================================
# NucleiDataset
# ===========================================================================

class NucleiDataset(Dataset):
    """
    Loads 100×100 nucleus patches from a split directory:

        split_dir/
          Tumor/        ← class 0
          Lymphocyte/   ← class 1
          Histiocyte/   ← class 2

    Each file is a uint8 (100, 100, 3) .npy array produced by
    2_data/extract_patches.py.

    augment_level : "baseline" | "moderate" | "improved" | "none"
    """

    def __init__(self, split_dir: str, augment_level: str = "none"):
        self.samples: list[tuple[str, int]] = []

        for cls_name, label in CLASS_MAP.items():
            cls_dir  = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"[WARN] Class directory not found: {cls_dir}")
                continue
            for path in sorted(glob.glob(os.path.join(cls_dir, "*.npy"))):
                self.samples.append((path, label))

        if not self.samples:
            raise RuntimeError(
                f"No .npy files found under {split_dir}.\n"
                "Run 2_data/extract_patches.py first."
            )

        if augment_level == "baseline":
            self._transform = _make_train_transform_baseline()
            self._use_alb   = False
        elif augment_level == "mild":
            self._transform = _make_train_transform_mild()
            self._use_alb   = True
        elif augment_level == "moderate":
            self._transform = _make_train_transform_moderate()
            self._use_alb   = True
        elif augment_level == "improved":
            self._transform = _make_train_transform_improved()
            self._use_alb   = True
        else:
            self._transform = _make_val_test_transform()
            self._use_alb   = False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        arr = np.load(path)   # uint8 (100, 100, 3)
        if self._use_alb:
            tensor = self._transform(image=arr)["image"]
        else:
            tensor = self._transform(Image.fromarray(arr, mode="RGB"))
        return tensor, label


# ===========================================================================
# TestDataset
# ===========================================================================

class TestDataset(Dataset):
    """
    Loads the Task 2 test set from a flat directory of .npy files.

    Filename format:
        test_set_{primary|metastatic}_roi_{NNN}_nuclei_{class}_{uuid}.npy

    __getitem__ returns (tensor, label, source, filename).
    source is "primary" | "metastatic" | "unknown".
    """

    def __init__(self, test_dir: str):
        self._transform = _make_val_test_transform()
        npy_files = sorted(glob.glob(os.path.join(test_dir, "*.npy")))
        if not npy_files:
            raise RuntimeError(f"No .npy files found in {test_dir}.")

        self.samples: list[tuple[str, int, str]] = []
        skipped = 0
        for path in npy_files:
            fname = os.path.basename(path).lower()
            label = None
            for keyword, cls_int in FILENAME_CLASS_MAP.items():
                if f"nuclei_{keyword}" in fname:
                    label = cls_int
                    break
            if label is None:
                skipped += 1
                continue
            if "_primary_" in fname:
                source = "primary"
            elif "_metastatic_" in fname:
                source = "metastatic"
            else:
                source = "unknown"
            self.samples.append((path, label, source))

        if skipped:
            print(f"[WARN] TestDataset: skipped {skipped} files with unrecognised class.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str, str]:
        path, label, source = self.samples[idx]
        arr    = np.load(path)
        tensor = self._transform(Image.fromarray(arr, mode="RGB"))
        return tensor, label, source, os.path.basename(path)
