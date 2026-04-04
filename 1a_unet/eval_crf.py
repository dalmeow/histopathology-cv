"""
Standalone dense CRF post-processing evaluation for unet_v2_resid.

Usage:
    python eval_crf.py

Runs a parameter sweep over (sigma_spatial, sigma_colour, n_iters) and prints
a comparison table of per-class Dice vs the no-CRF baseline.

Requires: pydensecrf  (pip install pydensecrf)
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "1_shared"))

from arch import UNet
from data_processing import TissueDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_ROOT      = _HERE.parent.parent / "Coumputer_Vision_Mini_Project_Data" / "Dataset_Splits"
CHECKPOINT     = _HERE / "checkpoints" / "exp5_lovasz" / "last_model.pt"
CLASS_NAMES    = ["Tumor", "Stroma", "Other"]
NUM_CLASSES    = 3

# For unnormalising tensors back to uint8 RGB (needed by CRF)
_MEAN = np.array([0.6199, 0.4123, 0.6963], dtype=np.float32)
_STD  = np.array([0.1975, 0.1944, 0.1381], dtype=np.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")

# CRF sweep grid
SIGMA_SPATIAL = [3]
SIGMA_COLOUR  = [30]
N_ITERS       = [5]


# ---------------------------------------------------------------------------
# Dice helper
# ---------------------------------------------------------------------------

def dice_per_class(pred: np.ndarray, target: np.ndarray) -> list[float]:
    """Compute per-class Dice for a single (H, W) pair. Returns list of C floats."""
    scores = []
    for cls in range(NUM_CLASSES):
        p = pred   == cls
        t = target == cls
        denom = p.sum() + t.sum()
        scores.append(2 * (p & t).sum() / denom if denom > 0 else float("nan"))
    return scores


# ---------------------------------------------------------------------------
# CRF
# ---------------------------------------------------------------------------

def apply_crf(
    raw_rgb: np.ndarray,
    softmax_probs: np.ndarray,
    sigma_spatial: float,
    sigma_colour: float,
    n_iters: int,
) -> np.ndarray:
    """
    Apply dense CRF to refine softmax_probs using raw_rgb as appearance cue.

    Args:
        raw_rgb:       uint8 (H, W, 3) image
        softmax_probs: float32 (C, H, W) model softmax output
        sigma_spatial: spatial kernel bandwidth (Gaussian + bilateral)
        sigma_colour:  colour kernel bandwidth (bilateral)
        n_iters:       CRF inference iterations

    Returns:
        (H, W) int32 predicted label map
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        raise ImportError("pydensecrf not installed. Run: pip install pydensecrf")

    C, H, W = softmax_probs.shape
    assert raw_rgb.shape == (H, W, 3) and raw_rgb.dtype == np.uint8

    d = dcrf.DenseCRF2D(W, H, C)

    # Unary energy: -log(softmax), clipped for numerical safety
    U = unary_from_softmax(softmax_probs)   # (C, H*W)
    d.setUnaryEnergy(U)

    # Pairwise smoothness: spatial-only Gaussian
    d.addPairwiseGaussian(sxy=sigma_spatial, compat=3)

    # Pairwise appearance: bilateral (colour + spatial)
    d.addPairwiseBilateral(
        sxy=sigma_spatial,
        srgb=sigma_colour,
        rgbim=np.ascontiguousarray(raw_rgb),
        compat=10,
    )

    Q = d.inference(n_iters)                        # (C, H*W)
    return np.argmax(Q, axis=0).reshape(H, W).astype(np.int32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Load model ---
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    print(f"Device   : {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}\n")

    model = UNet(
        num_classes  = 3,
        base_filters = 64,
        norm_type    = "instance",
        use_residual = True,
        use_deep_sup = True,
        dropout_p    = 0.3,
    ).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    # --- Load test set ---
    test_ds = TissueDataset(DATA_ROOT / "test", augment=False, img_size=512)
    loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    print(f"Test set : {len(test_ds)} images\n")

    # --- Collect softmax probs + raw images + masks (single pass) ---
    all_probs  = []   # (N, C, H, W) float32 numpy
    all_raw    = []   # (N, H, W, 3) uint8 numpy
    all_masks  = []   # (N, H, W) int64 numpy

    with torch.no_grad():
        for img_t, mask_t in tqdm(loader, desc="Inference"):
            logits = model(img_t.to(DEVICE))                          # (1, C, H, W)
            probs  = logits.softmax(dim=1).squeeze(0).cpu().numpy()   # (C, H, W)
            # Unnormalise tensor → uint8 RGB for CRF appearance term
            raw = np.clip(
                img_t.squeeze(0).permute(1, 2, 0).numpy() * _STD + _MEAN,
                0, 1,
            )
            raw = (raw * 255).astype(np.uint8)                        # (H, W, 3)
            mask = mask_t.squeeze(0).numpy()                          # (H, W)

            all_probs.append(probs)
            all_raw.append(raw)
            all_masks.append(mask)

    N = len(all_masks)

    # --- Baseline (no CRF) ---
    baseline_per_class = [[] for _ in range(NUM_CLASSES)]
    for probs, mask in zip(all_probs, all_masks):
        pred = probs.argmax(axis=0).astype(np.int32)
        for cls, d in enumerate(dice_per_class(pred, mask)):
            if not np.isnan(d):
                baseline_per_class[cls].append(d)

    baseline_means = [np.mean(v) if v else float("nan") for v in baseline_per_class]
    baseline_avg   = float(np.nanmean(baseline_means))

    # --- CRF sweep ---
    sweep_configs = list(product(SIGMA_SPATIAL, SIGMA_COLOUR, N_ITERS))
    results = []   # (sxy, srgb, n_iters, [dice_per_class], avg_dice)

    for sxy, srgb, n_iters in sweep_configs:
        tag = f"sxy={sxy:2d}  srgb={srgb:2d}  iters={n_iters:2d}"
        per_class = [[] for _ in range(NUM_CLASSES)]

        for probs, raw, mask in tqdm(
            zip(all_probs, all_raw, all_masks),
            total=N,
            desc=tag,
            leave=False,
        ):
            pred = apply_crf(raw, probs, sxy, srgb, n_iters)
            for cls, d in enumerate(dice_per_class(pred, mask)):
                if not np.isnan(d):
                    per_class[cls].append(d)

        means = [np.mean(v) if v else float("nan") for v in per_class]
        avg   = float(np.nanmean(means))
        results.append((sxy, srgb, n_iters, means, avg))

    # --- Print results table ---
    col_w = 8
    header_cls = "  ".join(f"{n:>{col_w}s}" for n in CLASS_NAMES)
    print("\n" + "=" * 75)
    print(f"{'Config':<30s}  {header_cls}  {'Avg':>{col_w}s}")
    print("=" * 75)

    # Baseline row
    cls_str = "  ".join(f"{v:>{col_w}.4f}" for v in baseline_means)
    print(f"{'Baseline (no CRF)':<30s}  {cls_str}  {baseline_avg:>{col_w}.4f}")
    print("-" * 75)

    # Sort by avg Dice descending
    results.sort(key=lambda r: r[4], reverse=True)

    best_avg = results[0][4]
    for sxy, srgb, n_iters, means, avg in results:
        tag     = f"sxy={sxy}  srgb={srgb}  iters={n_iters}"
        cls_str = "  ".join(f"{v:>{col_w}.4f}" for v in means)
        marker  = "  <-- best" if avg == best_avg else ""
        print(f"{tag:<30s}  {cls_str}  {avg:>{col_w}.4f}{marker}")

    print("=" * 75)

    # Summary
    best = results[0]
    delta = best[4] - baseline_avg
    sign  = "+" if delta >= 0 else ""
    print(f"\nBest CRF config: sxy={best[0]}, srgb={best[1]}, iters={best[2]}")
    print(f"Avg Dice: baseline={baseline_avg:.4f}  CRF best={best[4]:.4f}  "
          f"delta={sign}{delta:.4f}")


if __name__ == "__main__":
    main()
