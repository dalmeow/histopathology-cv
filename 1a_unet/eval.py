import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "1_shared"))

from arch import UNet
from data_processing import TissueDataset


DATA_ROOT   = _HERE.parent.parent / "Coumputer_Vision_Mini_Project_Data" / "Dataset_Splits"
CKPT_DIR    = _HERE / "checkpoints"
RESULTS_DIR = _HERE / "results"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else
                           "mps"  if torch.backends.mps.is_available() else "cpu")

CLASS_NAMES  = ["Tumor", "Stroma", "Other"]
CLASS_COLORS = np.array([[200, 0, 0], [0, 200, 0], [0, 0, 200]], dtype=np.uint8)
MEAN = np.array([0.6199, 0.4123, 0.6963])
STD  = np.array([0.1975, 0.1944, 0.1381])

N_QUALITATIVE = 10


# Helpers

def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[mask]


def _unnorm(t: torch.Tensor) -> np.ndarray:
    return np.clip(t.permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)


def _build_model(config: dict) -> torch.nn.Module:
    return UNet(
        num_classes  = 3,
        base_filters = 64,
        norm_type    = config["norm_type"],
        use_residual = config["use_residual"],
        use_deep_sup = config["use_deep_sup"],
        dropout_p    = config["dropout_p"],
    ).to(DEVICE)


# Quantitative evaluation

def _compute_metrics(model: torch.nn.Module, loader: DataLoader) -> dict:
    """
    Compute per-class Dice and IoU as mean-of-per-image scores,
    skipping images where a class is absent in both pred and GT.
    Returns a dict with keys: tumor_dice, stroma_dice, other_dice, avg_dice,
                               tumor_iou,  stroma_iou,  other_iou,  avg_iou.
    """
    per_class_dice = {c: [] for c in range(3)}
    per_class_iou  = {c: [] for c in range(3)}

    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="  evaluating", leave=False):
            preds = model(images.to(DEVICE)).argmax(dim=1).cpu()
            for b in range(preds.shape[0]):
                p = preds[b].view(-1)
                t = masks[b].view(-1)
                for c in range(3):
                    pred_c   = p == c
                    target_c = t == c
                    tp    = (pred_c & target_c).sum().item()
                    fp    = (pred_c & ~target_c).sum().item()
                    fn    = (~pred_c & target_c).sum().item()
                    denom_dice = pred_c.sum().item() + target_c.sum().item()
                    denom_iou  = tp + fp + fn
                    if denom_dice > 0:
                        per_class_dice[c].append(2 * tp / denom_dice)
                    if denom_iou > 0:
                        per_class_iou[c].append(tp / denom_iou)

    dice_scores = [float(np.mean(per_class_dice[c])) if per_class_dice[c] else 0.0
                   for c in range(3)]
    iou_scores  = [float(np.mean(per_class_iou[c]))  if per_class_iou[c]  else 0.0
                   for c in range(3)]

    return {
        "tumor_dice":  dice_scores[0],
        "stroma_dice": dice_scores[1],
        "other_dice":  dice_scores[2],
        "avg_dice":    float(np.mean(dice_scores)),
        "tumor_iou":   iou_scores[0],
        "stroma_iou":  iou_scores[1],
        "other_iou":   iou_scores[2],
        "avg_iou":     float(np.mean(iou_scores)),
    }


def _print_metrics(metrics: dict, label: str) -> None:
    print(f"\n  [{label}]")
    print(f"  {'Class':<10s}  {'Dice':>8s}  {'IoU':>8s}")
    print(f"  {'-'*30}")
    for name, key in zip(CLASS_NAMES, ["tumor", "stroma", "other"]):
        print(f"  {name:<10s}  {metrics[key+'_dice']:>8.4f}  {metrics[key+'_iou']:>8.4f}")
    print(f"  {'Avg':<10s}  {metrics['avg_dice']:>8.4f}  {metrics['avg_iou']:>8.4f}")


# Qualitative evaluation

def _qualitative(
    model:    torch.nn.Module,
    dataset:  TissueDataset,
    save_dir: Path,
    label:    str,
) -> None:
    """Save a grid of N_QUALITATIVE samples (image | GT | prediction)."""
    indices = np.linspace(0, len(dataset) - 1, N_QUALITATIVE, dtype=int)

    fig, axes = plt.subplots(N_QUALITATIVE, 3, figsize=(10, 3.5 * N_QUALITATIVE))
    fig.suptitle(f"{label}  —  Image  |  Ground Truth  |  Prediction", fontsize=11)

    model.eval()
    for row, idx in enumerate(indices):
        img_t, mask_t = dataset[idx]
        with torch.no_grad():
            pred = model(img_t.unsqueeze(0).to(DEVICE)).argmax(dim=1).squeeze().cpu()

        axes[row][0].imshow(_unnorm(img_t))
        axes[row][1].imshow(_mask_to_rgb(mask_t.numpy()))
        axes[row][2].imshow(_mask_to_rgb(pred.numpy()))
        for ax in axes[row]:
            ax.axis("off")

    axes[0][0].set_title("Image",        fontsize=9)
    axes[0][1].set_title("Ground Truth", fontsize=9)
    axes[0][2].set_title("Prediction",   fontsize=9)

    plt.tight_layout()
    out = save_dir / f"qualitative_{label}.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved qualitative grid → {out}")


# Summary CSV

_SUMMARY_FIELDS = [
    "name",
    "best_avg_dice", "best_tumor_dice", "best_stroma_dice", "best_other_dice",
    "best_avg_iou",
    "last_avg_dice", "last_tumor_dice", "last_stroma_dice", "last_other_dice",
    "last_avg_iou",
]


def _update_summary(name: str, best: dict, last: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV.exists()

    row = {
        "name": name,
        **{f"best_{k}": v for k, v in best.items()},
        **{f"last_{k}": v for k, v in last.items()},
    }
    # Keep only the summary fields
    row = {k: row[k] for k in _SUMMARY_FIELDS if k in row}

    with open(SUMMARY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v
                         for k, v in row.items()})

    print(f"\n  Summary updated → {SUMMARY_CSV}")


# Main entry point

def evaluate(config: dict) -> None:
    """
    Evaluate both best_model.pt and last_model.pt for the given experiment.
    Saves per-experiment metrics.json, qualitative grids, and appends to
    the shared summary.csv.
    """
    name     = config["name"]
    run_dir  = CKPT_DIR    / name
    save_dir = RESULTS_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    test_ds     = TissueDataset(DATA_ROOT / "test", augment=False, img_size=config["img_size"])
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"],
                             shuffle=False, num_workers=2)

    print(f"\n{'='*55}")
    print(f"Evaluating: {name}  ({len(test_ds)} test images)")
    print(f"{'='*55}")

    all_metrics = {}
    for label in ("best", "last"):
        ckpt = run_dir / f"{label}_model.pt"
        if not ckpt.exists():
            print(f"  [{label}] checkpoint not found at {ckpt} — skipping")
            all_metrics[label] = {}
            continue

        model = _build_model(config)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()

        metrics = _compute_metrics(model, test_loader)
        _print_metrics(metrics, label)
        _qualitative(model, test_ds, save_dir, label)
        all_metrics[label] = metrics

    # Save metrics.json
    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Metrics saved → {metrics_path}")

    # Append to summary.csv
    if all_metrics.get("best") and all_metrics.get("last"):
        _update_summary(name, all_metrics["best"], all_metrics["last"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Experiment name (matches checkpoints/ subdir)")
    args = parser.parse_args()

    # Minimal config for standalone eval — must match the training config for this run
    # Prefer calling evaluate() from run.py where the full config is available
    raise SystemExit(
        "Run eval via run.py to ensure the correct config is used.\n"
        "If you need standalone eval, call evaluate(config) directly with the full config dict."
    )
