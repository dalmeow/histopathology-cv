"""
Evaluation for Task 2a: Nuclei Classification.

Entry point: evaluate(config) — called by run.py, not directly.

Loads the best checkpoint for an experiment, runs TTA inference on the
test set, and saves metrics (JSON + TXT) and a confusion matrix PNG.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from dataset import TestDataset, CLASS_NAMES, NUM_CLASSES
from model   import SimpleClassifier, NucleiResNet, ResNet18Encoder

DATA_ROOT = _HERE.parent.parent / "Coumputer_Vision_Mini_Project_Data"
TEST_DIR  = str(DATA_ROOT / "Task2_Test_Set")
CKPT_DIR  = _HERE / "checkpoints"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str) -> nn.Module:
    ckpt        = torch.load(checkpoint_path, map_location=DEVICE)
    model_class = ckpt.get("model_class", "SimpleClassifier")
    use_eca     = ckpt.get("config", {}).get("use_eca", False)

    if model_class == "ResNet18Encoder":
        model = ResNet18Encoder()
    elif model_class == "NucleiResNet":
        model = NucleiResNet(use_eca=use_eca)
    else:
        model = SimpleClassifier()

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model


# ---------------------------------------------------------------------------
# Test Time Augmentation
# ---------------------------------------------------------------------------

def _tta_views(images: torch.Tensor) -> list[torch.Tensor]:
    """8 views: hflip(F/T) × vflip(F/T) × rot90(k=0,1)."""
    views = []
    for hflip in (False, True):
        for vflip in (False, True):
            img = images
            if hflip:
                img = torch.flip(img, dims=[3])
            if vflip:
                img = torch.flip(img, dims=[2])
            views.append(img)
            views.append(torch.rot90(img, k=1, dims=[2, 3]))
    return views   # 8 views


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_inference(model: nn.Module, loader: DataLoader) -> dict:
    """TTA inference over the test loader. Returns preds/targets/sources."""
    all_preds, all_targets, all_sources = [], [], []

    for images, labels, sources, _ in loader:
        images = images.to(DEVICE, non_blocking=True)
        views  = _tta_views(images)
        probs  = torch.stack(
            [F.softmax(model(v), dim=1) for v in views], dim=0
        ).mean(dim=0)                             # (B, C)
        preds  = probs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(labels.tolist())
        all_sources.extend(sources)

    return {"preds": all_preds, "targets": all_targets, "sources": all_sources}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(preds: list, targets: list) -> dict:
    labels    = list(range(NUM_CLASSES))
    accuracy  = sum(p == t for p, t in zip(preds, targets)) / len(targets)
    precision = precision_score(targets, preds, labels=labels, average=None, zero_division=0).tolist()
    recall    = recall_score(   targets, preds, labels=labels, average=None, zero_division=0).tolist()
    f1        = f1_score(       targets, preds, labels=labels, average=None, zero_division=0).tolist()
    return {
        "accuracy":  round(accuracy, 6),
        "macro_f1":  round(float(np.mean(f1)), 6),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": round(precision[i], 6),
                "recall":    round(recall[i],    6),
                "f1":        round(f1[i],        6),
            }
            for i in range(NUM_CLASSES)
        },
    }


def _source_breakdown(preds: list, targets: list, sources: list) -> dict:
    """Per-source (primary / metastatic) accuracy."""
    out = {}
    for src in ("primary", "metastatic"):
        idx = [i for i, s in enumerate(sources) if s == src]
        if not idx:
            continue
        acc = sum(preds[i] == targets[i] for i in idx) / len(idx)
        out[src] = {"n_samples": len(idx), "accuracy": round(acc, 6)}
    return out


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def _save_confusion_matrix(preds: list, targets: list, path: str, title: str = "") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[WARN] matplotlib/seaborn not available — skipping confusion matrix.")
        return

    cm      = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cmap="Blues", vmin=0, vmax=1, ax=ax)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j + 0.5, i + 0.72, f"(n={cm[i,j]})",
                    ha="center", va="center", fontsize=7, color="black")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(title or "Confusion Matrix", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix : {path}")


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def _print_results(tag: str, metrics: dict, source_breakdown: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Results: {tag}")
    print(f"{'='*60}")
    print(f"  Overall accuracy : {metrics['accuracy']:.4f}")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    print()
    print(f"  {'Class':15s}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}")
    print(f"  {'-'*48}")
    for cls in CLASS_NAMES:
        pc = metrics["per_class"][cls]
        print(f"  {cls:15s}  {pc['precision']:>9.4f}  {pc['recall']:>9.4f}  {pc['f1']:>9.4f}")
    if source_breakdown:
        print()
        for src, info in source_breakdown.items():
            print(f"  {src:12s}: n={info['n_samples']:4d}  acc={info['accuracy']:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def evaluate(config: dict) -> None:
    """
    Evaluate the best checkpoint for an experiment on the test set.

    Required config keys: name, ckpt_dir, test_dir, batch_size
    """
    exp_name  = config["name"]
    ckpt_path = CKPT_DIR / exp_name / "best.pth"
    out_dir   = CKPT_DIR / exp_name

    if not ckpt_path.exists():
        print(f"[WARN] Checkpoint not found: {ckpt_path} — skipping evaluation.")
        return

    print(f"\n{'='*60}")
    print(f"  Evaluating: {exp_name}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    model = _load_model(str(ckpt_path))

    ckpt    = torch.load(str(ckpt_path), map_location=DEVICE)
    saved_epoch = ckpt.get("epoch", "?")
    saved_acc   = ckpt.get("val_accuracy", "?")
    print(f"  Model class  : {model.__class__.__name__}")
    print(f"  Saved epoch  : {saved_epoch}")
    print(f"  Saved val acc: {saved_acc}")

    # ---- Test set ----
    test_ds = TestDataset(TEST_DIR)
    loader  = DataLoader(test_ds, batch_size=config["batch_size"],
                         shuffle=False, num_workers=2, pin_memory=True)
    print(f"  Test samples : {len(test_ds)}")
    print("  TTA enabled  : 8 views (hflip × vflip × rot90)")

    result   = _run_inference(model, loader)
    preds    = result["preds"]
    targets  = result["targets"]
    sources  = result["sources"]

    metrics  = _compute_metrics(preds, targets)
    src_bkdn = _source_breakdown(preds, targets, sources)

    _print_results(exp_name, metrics, src_bkdn)
    print()
    print(classification_report(targets, preds, target_names=CLASS_NAMES,
                                 digits=4, zero_division=0))

    # ---- Save confusion matrix ----
    cm_path = str(out_dir / "confusion_matrix_test.png")
    _save_confusion_matrix(preds, targets, cm_path,
                           title=f"Confusion Matrix — {exp_name}")

    # ---- Save JSON ----
    output = {
        "exp_name":        exp_name,
        "checkpoint":      str(ckpt_path),
        "model_class":     model.__class__.__name__,
        "saved_epoch":     saved_epoch,
        "saved_val_acc":   saved_acc,
        "n_samples":       len(targets),
        "metrics":         metrics,
        "source_breakdown": src_bkdn,
    }
    json_path = str(out_dir / "test_metrics.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Metrics JSON : {json_path}")

    # ---- Save TXT ----
    txt_path = str(out_dir / "test_metrics.txt")
    with open(txt_path, "w") as f:
        f.write(f"Experiment       : {exp_name}\n")
        f.write(f"Model class      : {model.__class__.__name__}\n")
        f.write(f"Saved epoch      : {saved_epoch}\n")
        f.write(f"Saved val acc    : {saved_acc}\n")
        f.write(f"Test samples     : {len(targets)}\n\n")
        f.write(f"Overall accuracy : {metrics['accuracy']:.6f}\n")
        f.write(f"Macro F1         : {metrics['macro_f1']:.6f}\n\n")
        f.write(f"{'Class':15s}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}\n")
        f.write(f"{'-'*52}\n")
        for cls in CLASS_NAMES:
            pc = metrics["per_class"][cls]
            f.write(f"{cls:15s}  {pc['precision']:>10.6f}  "
                    f"{pc['recall']:>10.6f}  {pc['f1']:>10.6f}\n")
        if src_bkdn:
            f.write("\nSource breakdown:\n")
            for src, info in src_bkdn.items():
                f.write(f"  {src:12s}: n={info['n_samples']:4d}  "
                        f"accuracy={info['accuracy']:.6f}\n")
        f.write("\n")
        f.write(classification_report(targets, preds, target_names=CLASS_NAMES,
                                       digits=4, zero_division=0))
    print(f"  Metrics TXT  : {txt_path}")
