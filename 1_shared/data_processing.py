import json
import random
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_MAP = {
    "tissue_tumor":            0,
    "tissue_stroma":           1,
    "tissue_blood_vessel":     2,
    "tissue_epidermis":        2,
    "tissue_white_background": 2,
    "tissue_necrosis":         2,
}
DRAW_ORDER  = [2, 1, 0]
NUM_CLASSES = 3

# Normalisation — computed from training set
NORMALIZE = transforms.Normalize(
    mean=[0.6199, 0.4123, 0.6963],
    std =[0.1975, 0.1944, 0.1381],
)


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def geojson_to_mask(geo_path: Path, height: int, width: int) -> np.ndarray:
    """Rasterize tissue polygons into a (H, W) integer mask."""
    mask = np.full((height, width), fill_value=2, dtype=np.int32)

    with open(geo_path) as f:
        gj = json.load(f)

    by_label: dict[int, list] = {0: [], 1: [], 2: []}
    for feat in gj.get("features", []):
        raw_cls   = feat["properties"]["classification"]["name"]
        label     = CLASS_MAP.get(raw_cls, 2)
        geom      = feat["geometry"]
        geom_type = geom["type"]

        if geom_type == "Polygon":
            rings = [geom["coordinates"][0]]
        elif geom_type == "MultiPolygon":
            rings = [poly[0] for poly in geom["coordinates"]]
        else:
            continue

        for ring in rings:
            pts = np.array(ring, dtype=np.float32)[:, :2].astype(np.int32)
            by_label[label].append(pts)

    for label in DRAW_ORDER:
        for pts in by_label[label]:
            cv2.fillPoly(mask, [pts], color=int(label))

    return mask


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

_RGB_FROM_HED = np.array([[0.65, 0.70, 0.29],
                           [0.07, 0.99, 0.11],
                           [0.27, 0.57, 0.78]], dtype=np.float32)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED).astype(np.float32)


def _hed_jitter(img: np.ndarray) -> np.ndarray:
    """HED stain augmentation (Tellez et al. 2019). Input/output: uint8 RGB."""
    img = img.astype(np.float32)
    od  = -np.log((img + 1.0) / 256.0)
    hed = od @ _HED_FROM_RGB.T
    for c in range(3):
        hed[:, :, c] *= 1.0 + random.uniform(-0.2, 0.2)
        hed[:, :, c] +=       random.uniform(-0.2, 0.2)
    od_rec = hed @ _RGB_FROM_HED.T
    return np.clip(np.exp(-od_rec) * 256.0 - 1.0, 0, 255).astype(np.uint8)


def _brightness_jitter(img: np.ndarray) -> np.ndarray:
    """Brightness jitter ±10%. Input/output: uint8 RGB."""
    img = img.astype(np.float32) * random.uniform(0.9, 1.1)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TissueDataset(Dataset):
    """
    Loads full-resolution images, resizes to img_size × img_size.

    Args:
        split_dir:          path to the dataset split directory (train / validation / test)
        augment:            if True, applies geometric augmentation (flip + rot90)
        augment_hed:        if True (and augment=True), additionally applies HED stain
                            augmentation and brightness jitter
        img_size:           spatial size images are resized to
        other_threshold:    images where Other-class fraction >= this value are oversampled
                            (0.0 disables oversampling)
        other_oversample_k: number of extra copies added for each Other-rich image
    """

    def __init__(
        self,
        split_dir:          Path,
        augment:            bool  = False,
        augment_hed:        bool  = False,
        img_size:           int   = 512,
        other_threshold:    float = 0.0,
        other_oversample_k: int   = 0,
    ):
        self.augment     = augment
        self.augment_hed = augment_hed
        self.img_size    = img_size
        self.samples     = []

        img_dir    = Path(split_dir) / "image"
        tissue_dir = Path(split_dir) / "tissue"

        for img_path in sorted(img_dir.glob("*.tif")):
            geo_path = tissue_dir / f"{img_path.stem}_tissue.geojson"
            if geo_path.exists():
                self.samples.append((img_path, geo_path))

        # Physical oversampling: duplicate Other-rich images in the sample list.
        # Each duplicate gets independent random augmentations per epoch.
        if other_threshold > 0 and other_oversample_k > 0:
            extras = []
            for img_path, geo_path in self.samples:
                with tifffile.TiffFile(str(img_path)) as tif:
                    h, w = tif.pages[0].shape[0], tif.pages[0].shape[1]
                mask = geojson_to_mask(geo_path, h, w)
                if (mask == 2).mean() >= other_threshold:
                    extras.extend([(img_path, geo_path)] * other_oversample_k)
            self.samples.extend(extras)
            if extras:
                n_rich = len(extras) // other_oversample_k
                print(f"  Oversampled {n_rich} Other-rich images "
                      f"(×{other_oversample_k + 1}) → {len(self.samples)} total")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, geo_path = self.samples[idx]

        img  = tifffile.imread(str(img_path))[:, :, :3]
        h, w = img.shape[:2]
        mask = geojson_to_mask(geo_path, h, w)

        if h != self.img_size or w != self.img_size:
            img  = cv2.resize(img,  (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            # Geometric augmentation
            k = random.randint(0, 3)
            if k:
                img  = np.rot90(img,  k).copy()
                mask = np.rot90(mask, k).copy()
            if random.random() > 0.5:
                img  = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if random.random() > 0.5:
                img  = np.flipud(img).copy()
                mask = np.flipud(mask).copy()

            # Stain / colour augmentation
            if self.augment_hed:
                img = _hed_jitter(img)
                img = _brightness_jitter(img)

        img_tensor  = NORMALIZE(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        mask_tensor = torch.from_numpy(mask).long()
        return img_tensor, mask_tensor
