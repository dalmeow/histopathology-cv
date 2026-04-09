"""
2×2 grid: AE pre-training strategy × finetune quality.

         Vanilla finetune  |  UNet best finetune
  ------------------------------------------------
  Full   exp1_baseline     |  exp2_mse_full
  Masked exp3_masked       |  exp4_masked_full

Per-image Dice shown under each panel.
Output saved to results/figures/.

Remove before submission.
"""

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "1_shared"))

from arch.autoencoder import Autoencoder
from data_processing import TissueDataset
from run import EXPERIMENTS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT   = _HERE.parent.parent / "Coumputer_Vision_Mini_Project_Data" / "Dataset_Splits"
CKPT_DIR    = _HERE / "checkpoints"
FIGURES_DIR = _HERE / "results" / "figures"

TARGET_INDICES = [13]

CLASS_COLORS = np.array([[200, 0, 0], [0, 200, 0], [0, 0, 200]], dtype=np.uint8)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")

plt.rcParams.update({
    "font.size":       14,
    "axes.titlesize":  15,
    "axes.labelsize":  14,
    "legend.fontsize": 13,
})

LEGEND_PATCHES = [
    mpatches.Patch(color=np.array([200, 0,   0  ]) / 255, label="Tumor"),
    mpatches.Patch(color=np.array([0,   200, 0  ]) / 255, label="Stroma"),
    mpatches.Patch(color=np.array([0,   0,   200]) / 255, label="Other"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_dice(pred: np.ndarray, mask: np.ndarray) -> float:
    scores = []
    for c in range(3):
        p = pred == c
        t = mask == c
        denom = p.sum() + t.sum()
        if denom > 0:
            scores.append(2 * (p & t).sum() / denom)
    return float(np.mean(scores)) if scores else float("nan")


def _build_model(config: dict) -> torch.nn.Module:
    model = Autoencoder(
        mode         = "finetune",
        base         = 64,
        dropout      = config["dropout_p"],
        use_residual = config["use_residual"],
        dimensions   = 3,
        norm_type    = config["norm_type"],
        use_deep_sup = config["use_deep_sup"],
    ).to(DEVICE)
    ckpt = CKPT_DIR / config["name"] / "ae_finetune_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


# Flat strip order: GT then experiments left-to-right
STRIP_PANELS = [
    (None, "Ground Truth"),
    ("1",  "Full + Vanilla"),
    ("2",  "Full + UNet best"),
    ("3",  "Masked + Vanilla"),
    ("4",  "Masked + UNet best"),
]

# ---------------------------------------------------------------------------
# Figure: 1×5 strip  (GT | exp1 | exp2 | exp3 | exp4)
# ---------------------------------------------------------------------------

def make_ae_grid(test_ds, idx: int) -> None:
    img_t, mask_t = test_ds[idx]
    mask = mask_t.numpy()

    n = len(STRIP_PANELS)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4))

    for ax, (key, label) in zip(axes, STRIP_PANELS):
        if key is None:
            ax.imshow(CLASS_COLORS[mask])
            ax.set_title(label, fontweight="bold")
        else:
            config = EXPERIMENTS[key]
            model  = _build_model(config)
            with torch.no_grad():
                pred = model(img_t.unsqueeze(0).to(DEVICE)).argmax(dim=1).squeeze(0).cpu().numpy()
            dice = _image_dice(pred, mask)
            ax.imshow(CLASS_COLORS[pred])
            ax.set_title(label, fontweight="bold")
            ax.set_xlabel(f"Dice: {dice:.4f}")
            print(f"  {config['name']:<25s}  dice={dice:.4f}")

        ax.set_xticks([])
        ax.set_yticks([])

    fig.legend(handles=LEGEND_PATCHES, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()

    out = FIGURES_DIR / f"ae_grid_{idx:02d}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    test_ds = TissueDataset(DATA_ROOT / "test", augment=False, img_size=512)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for idx in TARGET_INDICES:
        print(f"\n--- Image {idx:02d} ---")
        make_ae_grid(test_ds, idx)


if __name__ == "__main__":
    main()
