import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "1_shared"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from arch import UNet
from data_processing import TissueDataset

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
DATA_ROOT       = _HERE.parent.parent / "Coumputer_Vision_Mini_Project_Data" / "Dataset_Splits"
CHECKPOINTS_DIR = _HERE / "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")

CLASS_NAMES  = ["Tumor", "Stroma", "Other"]
CLASS_COLORS = np.array([[200, 0, 0], [0, 200, 0], [0, 0, 200]], dtype=np.uint8)
MEAN = np.array([0.6199, 0.4123, 0.6963])
STD  = np.array([0.1975, 0.1944, 0.1381])


def mask_to_rgb(mask_tensor: torch.Tensor) -> np.ndarray:
    return CLASS_COLORS[mask_tensor.numpy()]


def unnorm(t: torch.Tensor) -> np.ndarray:
    return np.clip(t.permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: torch.nn.Module, device: torch.device) -> TissueDataset:
    test_ds     = TissueDataset(DATA_ROOT / "test", augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    per_class_dice: dict[int, list] = {c: [] for c in range(3)}
    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Test"):
            preds = model(images.to(device)).argmax(dim=1).cpu().view(-1)
            masks = masks.view(-1)
            for cls in range(3):
                pred_c       = preds == cls
                target_c     = masks == cls
                intersection = (pred_c & target_c).sum().item()
                denom        = pred_c.sum().item() + target_c.sum().item()
                if denom > 0:
                    per_class_dice[cls].append(2 * intersection / denom)

    print("\nTest results:")
    mean_dices = []
    for cls, name in enumerate(CLASS_NAMES):
        d = float(np.mean(per_class_dice[cls])) if per_class_dice[cls] else 0.0
        mean_dices.append(d)
        print(f"  {name:<10s}  Dice = {d:.4f}")
    print(f"  {'Avg':<10s}  Dice = {np.mean(mean_dices):.4f}")

    return test_ds


# ---------------------------------------------------------------------------
# Qualitative visualisation
# ---------------------------------------------------------------------------

def qualitative(model: torch.nn.Module, test_ds: TissueDataset, device: torch.device,
                run_dir: Path, n_show: int = 4) -> None:
    fig, axes = plt.subplots(n_show, 3, figsize=(10, 3.5 * n_show))
    fig.suptitle("Image  |  Ground Truth  |  Prediction", fontsize=12)

    model.eval()
    for i in range(n_show):
        img_t, mask_t = test_ds[i]
        with torch.no_grad():
            pred = model(img_t.unsqueeze(0).to(device)).argmax(dim=1).squeeze().cpu()

        axes[i][0].imshow(unnorm(img_t));       axes[i][0].axis("off")
        axes[i][1].imshow(mask_to_rgb(mask_t)); axes[i][1].axis("off")
        axes[i][2].imshow(mask_to_rgb(pred));   axes[i][2].axis("off")
        if i == 0:
            axes[i][0].set_title("Image")
            axes[i][1].set_title("Ground Truth")
            axes[i][2].set_title("Prediction")

    plt.tight_layout()
    out_path = run_dir / "qualitative_results.png"
    plt.savefig(out_path, dpi=100)
    print(f"Saved qualitative results to {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    run_dir   = CHECKPOINTS_DIR / args.run_name
    best_path = run_dir / "unet_tissue_best.pt"

    if not best_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {best_path}")

    print(f"Device: {DEVICE}")
    print(f"Loading model from {best_path}")

    model = UNet(dimensions=3, base=64).to(DEVICE)
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()

    test_ds = evaluate(model, DEVICE)
    qualitative(model, test_ds, DEVICE, run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UNet on test set")
    parser.add_argument("--run-name", type=str, required=True,
                        help="Subdirectory name under checkpoints/ to load from")
    args = parser.parse_args()
    main(args)
