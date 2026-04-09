"""
Supervised training for Task 2a: Nuclei Classification.

Entry point: train(config) — called by run.py, not directly.

Behaviour matches the original task2_submission/train.py for both
the "baseline" and "improved" approach paths, but without argparse,
contrastive-specific code, or curriculum blur.
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from dataset import NucleiDataset, CLASS_NAMES, NUM_CLASSES
from model   import SimpleClassifier, NucleiResNet, ResNet18Encoder

DATA_ROOT = _HERE.parent.parent / "Coumputer_Vision_Mini_Project_Data"
TRAIN_DIR = str(DATA_ROOT / "task2_patches" / "train")
VAL_DIR   = str(DATA_ROOT / "task2_patches" / "validation")
CKPT_DIR  = _HERE / "checkpoints"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)

LOG_INTERVAL = 20   # print batch-level progress every N batches


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _per_class_metrics(preds: list, targets: list) -> tuple:
    """Return (precision, recall) arrays of length NUM_CLASSES."""
    labels    = list(range(NUM_CLASSES))
    precision = precision_score(targets, preds, labels=labels, average=None, zero_division=0)
    recall    = recall_score(   targets, preds, labels=labels, average=None, zero_division=0)
    return precision, recall


def _fmt(m: dict) -> str:
    per_cls = "  ".join(
        f"{CLASS_NAMES[i]}={m['precision'][i]:.3f}/{m['recall'][i]:.3f}"
        for i in range(NUM_CLASSES)
    )
    return f"loss={m['loss']:.4f}  acc={m['accuracy']:.4f}  [{per_cls}]"


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

def _mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha:  float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Beta(alpha, alpha) mixup. Returns (mixed_images, soft_labels).
    soft_labels shape: (B, NUM_CLASSES).
    """
    lam = float(np.random.beta(alpha, alpha))
    B   = images.size(0)
    idx = torch.randperm(B, device=images.device)
    mixed = lam * images + (1.0 - lam) * images[idx]
    y_a   = torch.zeros(B, NUM_CLASSES, device=labels.device).scatter_(1, labels.unsqueeze(1), 1.0)
    y_b   = torch.zeros(B, NUM_CLASSES, device=labels.device).scatter_(1, labels[idx].unsqueeze(1), 1.0)
    return mixed, lam * y_a + (1.0 - lam) * y_b


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def _train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimiser: torch.optim.Optimizer,
    scheduler,            # cosine (per-batch) or None
    epoch:     int,
    use_mixup: bool,
    mixup_alpha: float,
    use_cosine_sched: bool,
) -> dict:
    import torch.nn.functional as F

    model.train()
    total_loss, n_batches = 0.0, 0
    all_preds, all_targets = [], []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if use_mixup:
            images, soft_labels = _mixup_batch(images, labels, mixup_alpha)
            logits   = model(images)
            log_prob = F.log_softmax(logits, dim=1)
            loss     = -(soft_labels * log_prob).sum(dim=1).mean()
        else:
            logits = model(images)
            loss   = criterion(logits, labels)

        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        # CosineAnnealingWarmRestarts is stepped per batch
        if use_cosine_sched and scheduler is not None:
            scheduler.step(epoch - 1 + batch_idx / len(loader))

        total_loss  += loss.item()
        n_batches   += 1
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().tolist())

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            acc = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_preds)
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                  f"loss={loss.item():.4f}  acc={acc:.4f}")

    accuracy       = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_preds)
    precision, recall = _per_class_metrics(all_preds, all_targets)
    return {"loss": total_loss / n_batches, "accuracy": accuracy,
            "precision": precision, "recall": recall}


@torch.no_grad()
def _validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.CrossEntropyLoss,
) -> dict:
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_targets = [], []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        n_batches  += 1
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    accuracy       = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_preds)
    precision, recall = _per_class_metrics(all_preds, all_targets)
    return {"loss": total_loss / n_batches, "accuracy": accuracy,
            "precision": precision, "recall": recall}


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

def _csv_fields() -> list[str]:
    base = ["epoch", "phase", "loss", "accuracy"]
    for c in CLASS_NAMES:
        base += [f"prec_{c}", f"rec_{c}"]
    return base


def _log_row(writer, epoch: int, phase: str, m: dict) -> None:
    row = {"epoch": epoch, "phase": phase,
           "loss": round(m["loss"], 6), "accuracy": round(m["accuracy"], 6)}
    for i, c in enumerate(CLASS_NAMES):
        row[f"prec_{c}"] = round(float(m["precision"][i]), 6)
        row[f"rec_{c}"]  = round(float(m["recall"][i]),    6)
    writer.writerow(row)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train(config: dict) -> None:
    """
    Train a single experiment end-to-end.

    Required config keys:
      name, model, use_eca, augment_level, mixup_alpha,
      train_dir, val_dir, ckpt_dir,
      batch_size, num_epochs, lr, weight_decay,
      scheduler, lr_patience, lr_factor,      (if scheduler == "plateau")
      cosine_t0, cosine_t_mult, cosine_eta_min, (if scheduler == "cosine")
      early_stop, seed
    """
    _set_seed(config["seed"])

    exp_name  = config["name"]
    ckpt_dir  = CKPT_DIR / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pth"
    csv_path  = ckpt_dir / "training_log.csv"

    print(f"\n{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Device     : {DEVICE}")
    print(f"  Model      : {config['model']}  eca={config.get('use_eca', False)}")
    print(f"  Augment    : {config['augment_level']}")
    print(f"  Mixup      : {config['mixup_alpha']}")
    print(f"  Scheduler  : {config['scheduler']}")
    print(f"{'='*60}")

    # ---- Datasets ----
    train_ds = NucleiDataset(TRAIN_DIR, augment_level=config["augment_level"])
    val_ds   = NucleiDataset(VAL_DIR,   augment_level="none")
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)

    # ---- Model ----
    m = config["model"]
    if m == "simple":
        model = SimpleClassifier().to(DEVICE)
    elif m == "resnet18":
        model = ResNet18Encoder().to(DEVICE)
    else:
        model = NucleiResNet(use_eca=config.get("use_eca", False)).to(DEVICE)

    n_params = model.count_parameters()
    print(f"Parameters : {n_params:,}")

    # ---- Loss ----
    criterion = nn.CrossEntropyLoss()

    # ---- Optimiser ----
    OptimCls = torch.optim.Adam if config["model"] == "simple" else torch.optim.AdamW
    optimiser = OptimCls(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    # ---- Scheduler ----
    use_cosine = config["scheduler"] == "cosine"
    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimiser,
            T_0=config["cosine_t0"],
            T_mult=config["cosine_t_mult"],
            eta_min=config["cosine_eta_min"],
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="max",
            patience=config["lr_patience"],
            factor=config["lr_factor"],
        )

    # ---- CSV ----
    csv_fh = open(str(csv_path), "w", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=_csv_fields())
    writer.writeheader()

    # ---- Training loop ----
    use_mixup    = config["mixup_alpha"] > 0.0
    best_val_acc = -1.0
    no_improve   = 0

    epoch_bar = tqdm(range(1, config["num_epochs"] + 1), desc=exp_name)
    for epoch in epoch_bar:
        train_m = _train_epoch(
            model, train_loader, criterion, optimiser, scheduler, epoch,
            use_mixup=use_mixup, mixup_alpha=config["mixup_alpha"],
            use_cosine_sched=use_cosine,
        )
        val_m = _validate(model, val_loader, criterion)

        lr = optimiser.param_groups[0]["lr"]
        epoch_bar.set_postfix(
            val_acc=f"{val_m['accuracy']:.4f}",
            val_loss=f"{val_m['loss']:.4f}",
            lr=f"{lr:.1e}",
        )

        _log_row(writer, epoch, "train", train_m)
        _log_row(writer, epoch, "val",   val_m)
        csv_fh.flush()

        # ReduceLROnPlateau is stepped per epoch on val accuracy
        if not use_cosine:
            scheduler.step(val_m["accuracy"])

        # Checkpoint
        val_acc = val_m["accuracy"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save({
                "epoch":            epoch,
                "exp_name":         exp_name,
                "model_class":      model.__class__.__name__,
                "model_state_dict": model.state_dict(),
                "val_accuracy":     best_val_acc,
                "config": {
                    "use_eca": config.get("use_eca", False),
                },
            }, str(best_path))
        else:
            no_improve += 1
            if no_improve >= config["early_stop"]:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    csv_fh.close()
    print(f"\nTraining complete.")
    print(f"  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Checkpoint        : {best_path}")
    print(f"  Training log      : {csv_path}")
