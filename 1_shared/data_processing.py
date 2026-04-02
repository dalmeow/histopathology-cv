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
DRAW_ORDER         = [2, 1, 0]
NUM_CLASSES        = 3
PATCH_SIZE         = 512

# Normalisation — computed from training set via compute_mean_std().
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
            pts = np.array(ring, dtype=np.float32)[:, :2]
            by_label[label].append(pts.astype(np.int32))

    for label in DRAW_ORDER:
        for pts in by_label[label]:
            cv2.fillPoly(mask, [pts], color=int(label))

    return mask


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def _sample_crop_center(h: int, w: int, patch: int) -> tuple[int, int]:
    """Sample a uniformly random (cy, cx) crop center within bounds."""
    half = patch // 2
    return random.randint(half, h - half - 1), random.randint(half, w - half - 1)


# HED colour-space matrices (Macenko/skimage convention).
# RGB → OD → HED decomposition for stain augmentation.
_RGB_FROM_HED = np.array([[0.65, 0.70, 0.29],
                           [0.07, 0.99, 0.11],
                           [0.27, 0.57, 0.78]], dtype=np.float32)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED).astype(np.float32)


def _hed_jitter(img: np.ndarray) -> np.ndarray:
    """HED stain augmentation on a uint8 RGB image (Tellez et al. 2019).

    Perturbs each stain channel with:
        h' = h * (1 + alpha) + beta,  alpha,beta ~ U(-0.2, 0.2)
    """
    img = img.astype(np.float32)
    od  = -np.log((img + 1.0) / 256.0)         # RGB → optical density (H, W, 3)
    hed = od @ _HED_FROM_RGB.T                  # OD → HED concentrations (H, W, 3)
    for c in range(3):
        hed[:, :, c] *= 1.0 + random.uniform(-0.2, 0.2)
        hed[:, :, c] +=       random.uniform(-0.2, 0.2)
    od_rec = hed @ _RGB_FROM_HED.T              # HED → OD → RGB
    img = np.exp(-od_rec) * 256.0 - 1.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _color_jitter(img: np.ndarray) -> np.ndarray:
    """Brightness jitter (±10%) on a uint8 RGB image."""
    img = img.astype(np.float32)
    img *= random.uniform(0.9, 1.1)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def tta_predict(
    model      : torch.nn.Module,
    img_path   : Path,
    geo_path   : Path,
    device     : torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Test-time augmentation over 8 transforms (4 rotations × hflip).
    Averages softmax probabilities after un-augmenting each prediction.
    Returns (pred, mask) both as (H, W) int64 tensors at original resolution.
    """
    img = tifffile.imread(str(img_path))[:, :, :3]
    H, W = img.shape[:2]

    # Resize to model input size
    img_resized = cv2.resize(img, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_LINEAR)

    prob_sum = torch.zeros(NUM_CLASSES, PATCH_SIZE, PATCH_SIZE)

    model.eval()
    with torch.no_grad():
        for k in range(4):           # 0°, 90°, 180°, 270°
            for flip in [False, True]:
                aug = np.rot90(img_resized, k).copy()
                if flip:
                    aug = np.fliplr(aug).copy()

                t = torch.from_numpy(aug).permute(2, 0, 1).float() / 255.0
                t = NORMALIZE(t).unsqueeze(0).to(device)
                probs = model(t).softmax(dim=1).squeeze(0).cpu()  # (C, H, W)

                # Un-augment: reverse flip then rotation on prob maps
                p = probs.numpy().transpose(1, 2, 0)  # (H, W, C)
                if flip:
                    p = np.fliplr(p)
                p = np.rot90(p, -k).copy()
                prob_sum += torch.from_numpy(p.transpose(2, 0, 1))

    pred = prob_sum.argmax(dim=0).long()                           # (H, W)
    mask = torch.from_numpy(geojson_to_mask(geo_path, H, W)).long()
    # Resize mask to match pred if needed
    if H != PATCH_SIZE or W != PATCH_SIZE:
        import torch.nn.functional as F
        mask_resized = cv2.resize(mask.numpy().astype(np.int32), (PATCH_SIZE, PATCH_SIZE),
                                  interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask_resized).long()
    return pred, mask


def sliding_window_predict(
    model       : torch.nn.Module,
    img_path    : Path,
    geo_path    : Path,
    device      : torch.device,
    patch_size  : int = 512,
    stride      : int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run sliding window inference on a full-resolution image.
    Returns (pred, mask) both as (H, W) int64 tensors.
    """
    img = tifffile.imread(str(img_path))[:, :, :3]
    H, W = img.shape[:2]

    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = NORMALIZE(img_tensor)   # (3, H, W)

    pred_sum = torch.zeros(NUM_CLASSES, H, W)
    count    = torch.zeros(H, W)

    model.eval()
    with torch.no_grad():
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patch = img_tensor[:, y:y + patch_size, x:x + patch_size].unsqueeze(0).to(device)
                probs = model(patch).softmax(dim=1).squeeze(0).cpu()
                pred_sum[:, y:y + patch_size, x:x + patch_size] += probs
                count[y:y + patch_size, x:x + patch_size] += 1

    pred = (pred_sum / count.unsqueeze(0).clamp(min=1)).argmax(dim=0)  # (H, W)
    mask = torch.from_numpy(geojson_to_mask(geo_path, H, W)).long()    # (H, W)
    return pred, mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TissueDataset(Dataset):
    """
    Yields 512×512 images resized from full-resolution 1024×1024 images.

    Training (augment=True):
        - Random 90° rotation, horizontal/vertical flip, color jitter

    Validation / test (augment=False):
        - No augmentation
    """

    def __init__(self, split_dir: Path, augment: bool = False, img_size: int = 512, return_raw: bool = False):
        self.augment    = augment
        self.img_size   = img_size
        self.return_raw = return_raw
        self.samples    = []

        img_dir    = split_dir / "image"
        tissue_dir = split_dir / "tissue"

        for img_path in sorted(img_dir.glob("*.tif")):
            geo_path = tissue_dir / f"{img_path.stem}_tissue.geojson"
            if geo_path.exists():
                self.samples.append((img_path, geo_path))

    def get_sample_weights(self, cache_path: Path = None, floor: float = 0.1) -> torch.Tensor:
        """Per-image sampling weights based on Other+Stroma pixel fraction.

        Weights are proportional to the fraction of minority-class pixels
        (Stroma=1, Other=2) in each full-resolution mask, floored at `floor`
        so every image remains reachable.  Results are cached to avoid
        re-rasterizing GeoJSON on every run.
        """
        if cache_path is not None and Path(cache_path).exists():
            print(f"Loaded sample weights from cache: {cache_path}")
            return torch.from_numpy(np.load(cache_path)).float()

        print(f"Computing sample weights for {len(self.samples)} images...")
        weights = []
        for img_path, geo_path in self.samples:
            img  = tifffile.imread(str(img_path))[:, :, :3]
            h, w = img.shape[:2]
            mask = geojson_to_mask(geo_path, h, w)
            minority_frac = np.sum((mask == 1) | (mask == 2)) / mask.size
            weights.append(max(minority_frac, floor))

        arr = np.array(weights, dtype=np.float32)
        if cache_path is not None:
            np.save(cache_path, arr)
            print(f"Cached sample weights to: {cache_path}")

        return torch.from_numpy(arr).float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, geo_path = self.samples[idx]

        # Load full-resolution image, drop alpha channel
        img = tifffile.imread(str(img_path))[:, :, :3]
        h, w = img.shape[:2]
        mask = geojson_to_mask(geo_path, h, w)

        # Resize to target size (skip if already correct)
        if h != self.img_size or w != self.img_size:
            img  = cv2.resize(img,  (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # Augmentation (train only)
        if self.augment:
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

            img = _hed_jitter(img)
            img = _color_jitter(img)

        # To tensor + normalise
        raw_tensor  = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor  = NORMALIZE(raw_tensor.clone())
        mask_tensor = torch.from_numpy(mask).long()

        if self.return_raw:
            return img_tensor, raw_tensor, mask_tensor
        return img_tensor, mask_tensor
