"""
Two report figures per target image.

  ablation_strip_XX.png  — Exp1→Exp5 predicted masks with per-image Dice.
  final_comparison_XX.png — Raw image | GT | Exp6 CRF prediction.

CRF best params read from results/exp6_crf/metrics.json.
Output saved to results/figures/.

Remove before submission.
"""

import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "1_shared"))

from arch import UNet
from data_processing import TissueDataset
from eval_crf import apply_crf
from run import EXPERIMENTS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT   = _HERE.parent.parent / "Coumputer_Vision_Mini_Project_Data" / "Dataset_Splits"
CKPT_DIR    = _HERE / "checkpoints"
RESULTS_DIR = _HERE / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

TARGET_INDICES = [13] # all test images: 00-09

CLASS_COLORS = np.array([[200, 0, 0], [0, 200, 0], [0, 0, 200]], dtype=np.uint8)
_MEAN = np.array([0.6199, 0.4123, 0.6963], dtype=np.float32)
_STD  = np.array([0.1975, 0.1944, 0.1381], dtype=np.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")

ABLATION_EXPS = [
    ("1",  "Baseline"),
    ("2",  "Domain"),
    ("3a", "Loss balance"),
    ("4",  "Architecture"),
    ("5",  "Focal + Lovász"),
]

LEGEND_PATCHES = [
    mpatches.Patch(color=np.array([200, 0,   0  ]) / 255, label="Tumor"),
    mpatches.Patch(color=np.array([0,   200, 0  ]) / 255, label="Stroma"),
    mpatches.Patch(color=np.array([0,   0,   200]) / 255, label="Other"),
]

plt.rcParams.update({
    "font.size":        14,
    "axes.titlesize":   15,
    "axes.labelsize":   14,
    "legend.fontsize":  13,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_dice(pred: np.ndarray, mask: np.ndarray) -> float:
    """Avg Dice across 3 classes for a single (H, W) prediction."""
    scores = []
    for c in range(3):
        p = pred == c
        t = mask == c
        denom = p.sum() + t.sum()
        if denom > 0:
            scores.append(2 * (p & t).sum() / denom)
    return float(np.mean(scores)) if scores else float("nan")


def _load_crf_params() -> dict:
    with open(RESULTS_DIR / "exp6_crf" / "metrics.json") as f:
        return json.load(f)["crf_best"]["crf_params"]


def _build_model(config: dict) -> torch.nn.Module:
    model = UNet(
        num_classes  = 3,
        base_filters = 64,
        norm_type    = config["norm_type"],
        use_residual = config["use_residual"],
        use_deep_sup = config["use_deep_sup"],
        dropout_p    = config["dropout_p"],
    ).to(DEVICE)
    ckpt = CKPT_DIR / config["name"] / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


def _predict(model, img_t):
    with torch.no_grad():
        logits = model(img_t.unsqueeze(0).to(DEVICE))
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)


def _softmax_probs(model, img_t):
    with torch.no_grad():
        return model(img_t.unsqueeze(0).to(DEVICE)).softmax(dim=1).squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Figure 1: Ablation strip  (Exp1 → Exp5 predictions)
# ---------------------------------------------------------------------------

def make_ablation_strip(test_ds, idx: int, crf_params: dict) -> None:
    img_t, mask_t = test_ds[idx]
    mask = mask_t.numpy()
    n = 2 + len(ABLATION_EXPS)  # GT + ablation exps + CRF

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4))

    # GT panel
    axes[0].imshow(CLASS_COLORS[mask])
    axes[0].set_title("Ground Truth", fontweight="bold")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Ablation panels — keep exp5 model for CRF
    exp5_model = None
    for ax, (key, short_name) in zip(axes[1:], ABLATION_EXPS):
        config = EXPERIMENTS[key]
        model  = _build_model(config)
        pred   = _predict(model, img_t)
        dice   = _image_dice(pred, mask)

        ax.imshow(CLASS_COLORS[pred])
        ax.set_title(short_name, fontweight="bold")
        ax.set_xlabel(f"Dice: {dice:.4f}")
        ax.set_xticks([])
        ax.set_yticks([])
        print(f"  {config['name']:<25s}  dice={dice:.4f}")

        if key == "5":
            exp5_model = model

    # CRF panel
    raw_u8   = (np.clip(img_t.permute(1, 2, 0).numpy() * _STD + _MEAN, 0, 1) * 255).astype(np.uint8)
    probs    = _softmax_probs(exp5_model, img_t)
    crf_pred = apply_crf(raw_u8, probs,
                         crf_params["sigma_spatial"],
                         crf_params["sigma_colour"],
                         crf_params["n_iters"])
    dice_crf = _image_dice(crf_pred, mask)

    axes[-1].imshow(CLASS_COLORS[crf_pred])
    axes[-1].set_title("CRF", fontweight="bold")
    axes[-1].set_xlabel(f"Dice: {dice_crf:.4f}")
    axes[-1].set_xticks([])
    axes[-1].set_yticks([])
    print(f"  {'exp6_crf':<25s}  dice={dice_crf:.4f}")

    fig.legend(handles=LEGEND_PATCHES, loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()

    out = FIGURES_DIR / f"unet_progression.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 2: Final comparison  (Image | GT | Exp6 CRF)
# ---------------------------------------------------------------------------

def make_final_comparison(test_ds, idx: int, crf_params: dict) -> None:
    img_t, mask_t = test_ds[idx]
    mask = mask_t.numpy()

    model  = _build_model(EXPERIMENTS["5"])
    probs  = _softmax_probs(model, img_t)
    raw_u8 = (np.clip(img_t.permute(1, 2, 0).numpy() * _STD + _MEAN, 0, 1) * 255).astype(np.uint8)

    crf_pred = apply_crf(raw_u8, probs,
                         crf_params["sigma_spatial"],
                         crf_params["sigma_colour"],
                         crf_params["n_iters"])
    dice_exp6 = _image_dice(crf_pred, mask)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    axes[0].imshow(raw_u8)
    axes[0].set_title("Image", fontweight="bold")

    axes[1].imshow(CLASS_COLORS[mask])
    axes[1].set_title("Ground Truth", fontweight="bold")

    axes[2].imshow(CLASS_COLORS[crf_pred])
    axes[2].set_title("Exp6 (+ CRF)", fontweight="bold")
    axes[2].set_xlabel(f"Dice: {dice_exp6:.4f}")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.legend(handles=LEGEND_PATCHES, loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()

    out = FIGURES_DIR / f"final_comparison_{idx:02d}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    crf_params = _load_crf_params()
    test_ds    = TissueDataset(DATA_ROOT / "test", augment=False, img_size=512)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for idx in TARGET_INDICES:
        print(f"\n--- Image {idx:02d} ---")
        print("Ablation strip…")
        make_ablation_strip(test_ds, idx, crf_params)
        print("Final comparison…")
        make_final_comparison(test_ds, idx, crf_params)


if __name__ == "__main__":
    main()
