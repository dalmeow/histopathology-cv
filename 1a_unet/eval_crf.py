"""
Dense CRF post-processing evaluation — Experiment 6.

Exposes evaluate_crf(config) which:
  1. Runs a single inference pass to cache softmax probs, raw images, GT masks.
  2. Sweeps CRF params and selects the best configuration by avg Dice.
  3. Saves sweep_results.json, qualitative grid (Image|GT|Baseline|CRF),
     metrics.json, and appends to the shared summary.csv.

Requires: pydensecrf  (pip install pydensecrf)
"""

import csv
import json
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "1_shared"))

from unet import UNet
from data_processing import TissueDataset


DATA_ROOT   = _HERE.parent.parent / "Coumputer_Vision_Mini_Project_Data" / "Dataset_Splits"
CKPT_DIR    = _HERE / "checkpoints"
RESULTS_DIR = _HERE / "results"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")

CLASS_NAMES  = ["Tumor", "Stroma", "Other"]
CLASS_COLORS = np.array([[200, 0, 0], [0, 200, 0], [0, 0, 200]], dtype=np.uint8)
NUM_CLASSES  = 3

_MEAN = np.array([0.6199, 0.4123, 0.6963], dtype=np.float32)
_STD  = np.array([0.1975, 0.1944, 0.1381], dtype=np.float32)

N_QUALITATIVE = 10

_SUMMARY_FIELDS = [
    "name",
    "best_avg_dice", "best_tumor_dice", "best_stroma_dice", "best_other_dice",
    "best_avg_iou",
    "last_avg_dice", "last_tumor_dice", "last_stroma_dice", "last_other_dice",
    "last_avg_iou",
]


# Helpers

def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[mask]


def _metrics_from_preds(preds: list, masks: list) -> dict:
    """Compute per-class Dice and IoU (mean-of-per-image) from (H,W) int arrays."""
    per_class_dice = {c: [] for c in range(NUM_CLASSES)}
    per_class_iou  = {c: [] for c in range(NUM_CLASSES)}

    for pred, mask in zip(preds, masks):
        p = pred.ravel()
        t = mask.ravel()
        for c in range(NUM_CLASSES):
            pred_c   = p == c
            target_c = t == c
            tp = int((pred_c & target_c).sum())
            fp = int((pred_c & ~target_c).sum())
            fn = int((~pred_c & target_c).sum())
            denom_dice = pred_c.sum() + target_c.sum()
            denom_iou  = tp + fp + fn
            if denom_dice > 0:
                per_class_dice[c].append(2 * tp / denom_dice)
            if denom_iou > 0:
                per_class_iou[c].append(tp / denom_iou)

    dice = [float(np.mean(per_class_dice[c])) if per_class_dice[c] else 0.0
            for c in range(NUM_CLASSES)]
    iou  = [float(np.mean(per_class_iou[c]))  if per_class_iou[c]  else 0.0
            for c in range(NUM_CLASSES)]

    return {
        "tumor_dice":  dice[0], "stroma_dice": dice[1], "other_dice": dice[2],
        "avg_dice":    float(np.mean(dice)),
        "tumor_iou":   iou[0],  "stroma_iou":  iou[1],  "other_iou":  iou[2],
        "avg_iou":     float(np.mean(iou)),
    }


def _print_metrics(metrics: dict, label: str) -> None:
    print(f"\n  [{label}]")
    print(f"  {'Class':<10s}  {'Dice':>8s}  {'IoU':>8s}")
    print(f"  {'-'*30}")
    for name, key in zip(CLASS_NAMES, ["tumor", "stroma", "other"]):
        print(f"  {name:<10s}  {metrics[key+'_dice']:>8.4f}  {metrics[key+'_iou']:>8.4f}")
    print(f"  {'Avg':<10s}  {metrics['avg_dice']:>8.4f}  {metrics['avg_iou']:>8.4f}")


# CRF

def apply_crf(
    raw_rgb: np.ndarray,
    softmax_probs: np.ndarray,
    sigma_spatial: float,
    sigma_colour: float,
    n_iters: int,
) -> np.ndarray:
    """Apply dense CRF; returns (H, W) int32 label map."""
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        raise ImportError("pydensecrf not installed. Run: pip install pydensecrf")

    C, H, W = softmax_probs.shape
    d = dcrf.DenseCRF2D(W, H, C)
    d.setUnaryEnergy(unary_from_softmax(softmax_probs))
    d.addPairwiseGaussian(sxy=sigma_spatial, compat=3)
    d.addPairwiseBilateral(
        sxy=sigma_spatial,
        srgb=sigma_colour,
        rgbim=np.ascontiguousarray(raw_rgb),
        compat=10,
    )
    Q = d.inference(n_iters)
    return np.argmax(Q, axis=0).reshape(H, W).astype(np.int32)


def _run_crf_on_dataset(all_raw, all_probs, sxy, srgb, n_iters, desc="") -> list:
    return [
        apply_crf(raw, probs, sxy, srgb, n_iters)
        for raw, probs in tqdm(zip(all_raw, all_probs), total=len(all_raw),
                               desc=f"  {desc}", leave=False)
    ]


# Qualitative grid  (4 cols: Image | GT | Baseline | CRF)

def _qualitative_crf(
    all_raw:        list,
    all_masks:      list,
    baseline_preds: list,
    crf_preds:      list,
    save_dir:       Path,
    label:          str,
) -> None:
    indices = np.linspace(0, len(all_raw) - 1, N_QUALITATIVE, dtype=int)

    fig, axes = plt.subplots(N_QUALITATIVE, 4, figsize=(14, 3.5 * N_QUALITATIVE))
    fig.suptitle(f"{label}  —  Image  |  Ground Truth  |  Baseline  |  CRF", fontsize=11)

    for row, idx in enumerate(indices):
        axes[row][0].imshow(all_raw[idx])
        axes[row][1].imshow(_mask_to_rgb(all_masks[idx]))
        axes[row][2].imshow(_mask_to_rgb(baseline_preds[idx]))
        axes[row][3].imshow(_mask_to_rgb(crf_preds[idx]))
        for ax in axes[row]:
            ax.axis("off")

    for col, title in enumerate(["Image", "Ground Truth", "Baseline", "CRF"]):
        axes[0][col].set_title(title, fontsize=9)

    plt.tight_layout()
    out = save_dir / f"qualitative_{label}.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved qualitative grid → {out}")


# Summary CSV  (same format as eval.py)

def _update_summary(name: str, best: dict, last: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV.exists()
    row = {"name": name,
           **{f"best_{k}": v for k, v in best.items()},
           **{f"last_{k}": v for k, v in last.items()}}
    row = {k: row[k] for k in _SUMMARY_FIELDS if k in row}
    with open(SUMMARY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v
                         for k, v in row.items()})
    print(f"\n  Summary updated → {SUMMARY_CSV}")


# Main entry point

def evaluate_crf(config: dict) -> None:
    """
    Sweep CRF params on the best checkpoint from source_exp, then run full
    eval (metrics.json, qualitative grid, summary.csv) with the best params.

    In summary.csv: "best" = CRF best result, "last" = no-CRF baseline.
    """
    name       = config["name"]
    source_exp = config["source_exp"]
    ckpt       = CKPT_DIR / source_exp / "best_model.pt"
    save_dir   = RESULTS_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"\n{'='*55}")
    print(f"Experiment: {name}")
    print(f"Source    : {source_exp}  (best_model.pt)")
    print(f"Device    : {DEVICE}")
    print(f"{'='*55}")

    # --- Load model ---
    model = UNet(
        num_classes  = NUM_CLASSES,
        base_filters = 64,
        norm_type    = config["norm_type"],
        use_residual = config["use_residual"],
        use_deep_sup = config["use_deep_sup"],
        dropout_p    = config["dropout_p"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    def _collect(split: str, desc: str):
        ds     = TissueDataset(DATA_ROOT / split, augment=False, img_size=config["img_size"])
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
        print(f"\n{split.capitalize()} set : {len(ds)} images")
        probs_list, raw_list, mask_list = [], [], []
        with torch.no_grad():
            for img_t, mask_t in tqdm(loader, desc=f"  {desc}"):
                probs = model(img_t.to(DEVICE)).softmax(dim=1).squeeze(0).cpu().numpy()
                raw   = np.clip(img_t.squeeze(0).permute(1, 2, 0).numpy() * _STD + _MEAN, 0, 1)
                raw   = (raw * 255).astype(np.uint8)
                probs_list.append(probs)
                raw_list.append(raw)
                mask_list.append(mask_t.squeeze(0).numpy())
        return probs_list, raw_list, mask_list

    # --- Phase 1: sweep on validation set ---
    val_probs, val_raw, val_masks = _collect("validation", "val inference")

    val_baseline_preds   = [p.argmax(axis=0).astype(np.int32) for p in val_probs]
    val_baseline_metrics = _metrics_from_preds(val_baseline_preds, val_masks)
    _print_metrics(val_baseline_metrics, "Val baseline (no CRF)")

    sweep_grid = list(product(
        config["crf_sigma_spatial"],
        config["crf_sigma_colour"],
        config["crf_n_iters"],
    ))
    print(f"\n  Sweeping {len(sweep_grid)} CRF configurations on validation set…")
    sweep_results = []

    for sxy, srgb, n_iters in sweep_grid:
        tag   = f"sxy={sxy}  srgb={srgb}  iters={n_iters}"
        preds = _run_crf_on_dataset(val_raw, val_probs, sxy, srgb, n_iters, desc=tag)
        m     = _metrics_from_preds(preds, val_masks)
        sweep_results.append({
            "sigma_spatial": sxy, "sigma_colour": srgb, "n_iters": n_iters, **m
        })

    sweep_results.sort(key=lambda r: r["avg_dice"], reverse=True)

    # --- Print sweep table (val) ---
    col_w = 8
    print("\n" + "=" * 80)
    print(f"  Sweep results (validation set)")
    print(f"{'Config':<30s}  " +
          "  ".join(f"{n:>{col_w}s}" for n in CLASS_NAMES) +
          f"  {'Avg':>{col_w}s}")
    print("=" * 80)
    cls_str = "  ".join(f"{val_baseline_metrics[k+'_dice']:>{col_w}.4f}"
                        for k in ["tumor", "stroma", "other"])
    print(f"{'Baseline (no CRF)':<30s}  {cls_str}  {val_baseline_metrics['avg_dice']:>{col_w}.4f}")
    print("-" * 80)
    best_val_avg = sweep_results[0]["avg_dice"]
    for r in sweep_results:
        tag     = f"sxy={r['sigma_spatial']}  srgb={r['sigma_colour']}  iters={r['n_iters']}"
        cls_str = "  ".join(f"{r[k+'_dice']:>{col_w}.4f}" for k in ["tumor", "stroma", "other"])
        marker  = "  <-- best" if r["avg_dice"] == best_val_avg else ""
        print(f"{tag:<30s}  {cls_str}  {r['avg_dice']:>{col_w}.4f}{marker}")
    print("=" * 80)

    best = sweep_results[0]
    print(f"\n  Best config (val): sxy={best['sigma_spatial']}, "
          f"srgb={best['sigma_colour']}, iters={best['n_iters']}")

    # --- Save sweep results ---
    sweep_path = save_dir / "sweep_results.json"
    with open(sweep_path, "w") as f:
        json.dump({"val_baseline": val_baseline_metrics, "sweep": sweep_results}, f, indent=2)
    print(f"\n  Sweep results saved → {sweep_path}")

    # --- Phase 2: final eval on test set with best params ---
    test_probs, test_raw, test_masks = _collect("test", "test inference")

    test_baseline_preds   = [p.argmax(axis=0).astype(np.int32) for p in test_probs]
    test_baseline_metrics = _metrics_from_preds(test_baseline_preds, test_masks)
    _print_metrics(test_baseline_metrics, "Test baseline (no CRF)")

    best_crf_preds = _run_crf_on_dataset(
        test_raw, test_probs,
        best["sigma_spatial"], best["sigma_colour"], best["n_iters"],
        desc="final CRF eval (test)",
    )
    best_crf_metrics_test = _metrics_from_preds(best_crf_preds, test_masks)
    _print_metrics(best_crf_metrics_test, "Test CRF best")

    delta = best_crf_metrics_test["avg_dice"] - test_baseline_metrics["avg_dice"]
    print(f"\n  Avg Dice: test baseline={test_baseline_metrics['avg_dice']:.4f}  "
          f"CRF={best_crf_metrics_test['avg_dice']:.4f}  delta={delta:+.4f}")

    # Qualitative grid (test set)
    _qualitative_crf(test_raw, test_masks, test_baseline_preds, best_crf_preds, save_dir, name)

    # metrics.json
    best_crf_metrics_test["crf_params"] = {
        "sigma_spatial": best["sigma_spatial"],
        "sigma_colour":  best["sigma_colour"],
        "n_iters":       best["n_iters"],
    }
    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "crf_best":      best_crf_metrics_test,
            "baseline":      test_baseline_metrics,
            "val_sweep_best": {k: v for k, v in best.items()
                               if k not in ("sigma_spatial", "sigma_colour", "n_iters")},
        }, f, indent=2)
    print(f"\n  Metrics saved → {metrics_path}")

    # summary.csv: "best" = test CRF result, "last" = test baseline (no-CRF)
    _update_summary(name, best_crf_metrics_test, test_baseline_metrics)


if __name__ == "__main__":
    raise SystemExit(
        "Run via run.py:  python run.py --exp 6\n"
        "Or call evaluate_crf(config) directly with the full config dict."
    )
