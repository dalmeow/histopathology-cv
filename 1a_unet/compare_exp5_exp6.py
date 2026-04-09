"""
Report figure: Exp5 vs Exp6 (CRF) comparison.
Saves one figure per image: GT | Exp5 prediction | Exp6 CRF prediction.

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

from arch import UNet
from data_processing import TissueDataset
from eval_crf import apply_crf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT  = _HERE.parent.parent / "Coumputer_Vision_Mini_Project_Data" / "Dataset_Splits"
CHECKPOINT = _HERE / "checkpoints" / "exp5_lovasz" / "best_model.pt"
SAVE_DIR   = _HERE / "results" / "exp6_crf"

TARGET_INDICES = [7]

# Best CRF params (from validation set grid search):
CRF_SXY    = 10
CRF_SRGB   = 50
CRF_NITERS = 10

CLASS_COLORS = np.array([[200, 0, 0], [0, 200, 0], [0, 0, 200]], dtype=np.uint8)
_MEAN = np.array([0.6199, 0.4123, 0.6963], dtype=np.float32)
_STD  = np.array([0.1975, 0.1944, 0.1381], dtype=np.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model = UNet(
        num_classes=3, base_filters=64, norm_type="instance",
        use_residual=True, use_deep_sup=True, dropout_p=0.3,
    ).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    test_ds = TissueDataset(DATA_ROOT / "test", augment=False, img_size=512)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for idx in TARGET_INDICES:
        img_t, mask_t = test_ds[idx]

        with torch.no_grad():
            probs = model(img_t.unsqueeze(0).to(DEVICE)).softmax(dim=1).squeeze(0).cpu().numpy()

        raw_uint8 = (np.clip(img_t.permute(1, 2, 0).numpy() * _STD + _MEAN, 0, 1) * 255).astype(np.uint8)

        exp5_pred = probs.argmax(axis=0).astype(np.int32)
        crf_pred  = apply_crf(raw_uint8, probs, CRF_SXY, CRF_SRGB, CRF_NITERS)

        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        axes[0].imshow(CLASS_COLORS[mask_t.numpy()])
        axes[1].imshow(CLASS_COLORS[exp5_pred])
        axes[2].imshow(CLASS_COLORS[crf_pred])
        axes[0].set_title("Ground Truth",   fontsize=10)
        axes[1].set_title("Focal + Lovász",  fontsize=10)
        axes[2].set_title("Exp6 (+ CRF)",   fontsize=10)
        for ax in axes:
            ax.axis("off")

        legend_patches = [
            mpatches.Patch(color=np.array([200, 0,   0  ]) / 255, label="Tumor"),
            mpatches.Patch(color=np.array([0,   200, 0  ]) / 255, label="Stroma"),
            mpatches.Patch(color=np.array([0,   0,   200]) / 255, label="Other"),
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=3,
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))

        plt.tight_layout()
        out = SAVE_DIR / f"compare_img{idx:02d}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
