"""
Supervised head fine-tuning for Task 2b.

Two entry points, called by run.py:
  kmeans_init(config) — k-means head initialisation (exp 2 + 3 only)
  finetune(config)    — frozen-encoder supervised head training

Checkpoint layout:
  checkpoints/pretrain/{pretrain_name}/best.pth        — encoder from pretrain.py
  checkpoints/pretrain/{pretrain_name}/kmeans_head.pth — encoder + init head
  checkpoints/{exp_name}/best.pth                      — best finetune model
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "2a_classifier"))

from dataset  import PlainDataset                          # 2b dataset (unlabelled)
from model    import NucleiResNet, ResNet18Encoder         # encoders
from dataset  import NucleiDataset, CLASS_NAMES, NUM_CLASSES  # noqa: F811 (from 2a via sys.path)

DATA_ROOT       = _HERE.parent.parent / "Coumputer_Vision_Mini_Project_Data"
TRAIN_DIR       = str(DATA_ROOT / "task2_patches" / "train")
VAL_DIR         = str(DATA_ROOT / "task2_patches" / "validation")
CONTRASTIVE_DIR = str(DATA_ROOT / "task2_patches" / "contrastive")
PATCH_DIRS      = [CONTRASTIVE_DIR]   # k-means features from contrastive set only
PRETRAIN_DIR    = _HERE / "checkpoints" / "pretrain"
CKPT_DIR        = _HERE / "checkpoints"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)

LOG_INTERVAL = 20


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
# Model helpers
# ---------------------------------------------------------------------------

def _build_encoder(config: dict) -> nn.Module:
    if config["backbone"] == "resnet18":
        return ResNet18Encoder().to(DEVICE)
    return NucleiResNet(use_eca=config.get("use_eca", True)).to(DEVICE)


def _build_head(fdim: int) -> nn.Sequential:
    """MLP head: Linear(fdim→fdim//2) → ReLU → Dropout(0.3) → Linear(fdim//2→3)."""
    return nn.Sequential(
        nn.Linear(fdim, fdim // 2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(fdim // 2, NUM_CLASSES),
    )


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_features(encoder: nn.Module, loader: DataLoader) -> np.ndarray:
    """Extract features from an unlabelled loader (single tensor per item)."""
    encoder.eval()
    feats = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        feats.append(encoder.get_features(batch.to(DEVICE)).cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def _extract_labelled_features(
    encoder: nn.Module, loader: DataLoader
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (features, labels) from a labelled loader."""
    encoder.eval()
    feats, labels = [], []
    for images, lbls in loader:
        feats.append(encoder.get_features(images.to(DEVICE)).cpu().numpy())
        labels.append(lbls.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


# ---------------------------------------------------------------------------
# K-means head initialisation
# ---------------------------------------------------------------------------

def kmeans_init(config: dict) -> None:
    """
    K-means head initialisation for NucleiResNet encoders (exp 2 + 3).

    Steps:
      1. Load frozen SimCLR encoder from pretrain checkpoint.
      2. Extract features from ALL patches (unlabelled).
      3. K-means (k=3) on those features — no labels used.
      4. Map cluster IDs → class labels via majority vote on training set.
      5. PCA-init head[0] (Linear fdim→fdim//2): top-fdim//2 principal components.
      6. Init head[3] (Linear fdim//2→3): L2-normalised projected class centroids.
      7. Save new checkpoint with initialised encoder + head.
    """
    _set_seed(config["seed"])

    pretrain_name = config["pretrain_name"]
    encoder_ckpt  = PRETRAIN_DIR / pretrain_name / "best.pth"
    out_path      = PRETRAIN_DIR / pretrain_name / "kmeans_head.pth"

    print(f"\n{'='*60}")
    print(f"  K-means Init : {pretrain_name}")
    print(f"  Encoder ckpt : {encoder_ckpt}")
    print(f"{'='*60}")

    # Step 1: Load encoder
    state   = torch.load(str(encoder_ckpt), map_location=DEVICE)
    encoder = _build_encoder(config)
    encoder.load_state_dict(state["encoder_state_dict"], strict=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    fdim    = encoder.feature_dim
    hid_dim = fdim // 2

    # Step 2: Extract unlabelled features (all patches)
    print("\nStep 2: Extracting unlabelled features …")
    plain_ds     = PlainDataset(PATCH_DIRS)
    plain_loader = DataLoader(plain_ds, batch_size=256, shuffle=False, num_workers=2)
    contra_feats = _extract_features(encoder, plain_loader)
    print(f"  Features shape : {contra_feats.shape}")

    # Step 3: K-means (k=3)
    print("\nStep 3: Running K-means (k=3) …")
    contra_64 = contra_feats.astype(np.float64)
    km = KMeans(n_clusters=NUM_CLASSES, random_state=config["seed"], n_init=10, max_iter=300)
    km.fit(contra_64)
    centroids   = km.cluster_centers_.astype(np.float64)   # (3, fdim)
    cluster_ids = km.labels_
    for k in range(NUM_CLASSES):
        n = (cluster_ids == k).sum()
        print(f"  Cluster {k}: {n:,} patches ({n / len(cluster_ids) * 100:.1f}%)")

    # Step 4: Map clusters → class labels via training set majority vote
    print("\nStep 4: Mapping clusters → class labels …")
    train_ds     = NucleiDataset(TRAIN_DIR, augment_level="none")
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=2)
    train_feats, train_labels = _extract_labelled_features(encoder, train_loader)

    train_cluster_ids = km.predict(train_feats.astype(np.float64))
    cluster_to_class  = {}
    for k in range(NUM_CLASSES):
        mask = train_cluster_ids == k
        if mask.sum() == 0:
            cluster_to_class[k] = k
            continue
        counts = np.bincount(train_labels[mask], minlength=NUM_CLASSES)
        cluster_to_class[k] = int(counts.argmax())
        print(f"  Cluster {k} → class {cluster_to_class[k]} "
              f"({CLASS_NAMES[cluster_to_class[k]]})  votes: {counts.tolist()}")

    if len(set(cluster_to_class.values())) < NUM_CLASSES:
        print("  [WARN] Duplicate mapping — falling back to identity (0→0, 1→1, 2→2).")
        cluster_to_class = {i: i for i in range(NUM_CLASSES)}

    # Reorder centroids so centroid[i] corresponds to class i
    ordered_centroids = np.zeros((NUM_CLASSES, fdim), dtype=np.float32)
    for cluster_id, class_id in cluster_to_class.items():
        ordered_centroids[class_id] = centroids[cluster_id]

    # Step 5: PCA init for head[0] (Linear fdim→hid_dim)
    print(f"\nStep 5: PCA init for head[0] ({fdim}→{hid_dim}) …")
    pca = PCA(n_components=hid_dim, random_state=config["seed"])
    pca.fit(contra_64)
    pca_components = pca.components_.astype(np.float32)   # (hid_dim, fdim)
    explained      = pca.explained_variance_ratio_.sum()
    print(f"  Top {hid_dim} PCs explain {explained * 100:.1f}% of variance")

    # Step 6: Init head[3] (Linear hid_dim→3) from projected centroids
    print(f"\nStep 6: Init head[3] ({hid_dim}→{NUM_CLASSES}) from projected centroids …")
    centroid_proj = ordered_centroids @ pca_components.T    # (3, hid_dim)
    norms         = np.linalg.norm(centroid_proj, axis=1, keepdims=True) + 1e-8
    centroid_proj_normed = (centroid_proj / norms).astype(np.float32)

    # Step 7: Build initialised model and save
    print("\nStep 7: Building initialised model …")
    model = _build_encoder(config)
    model.load_state_dict(state["encoder_state_dict"], strict=False)
    model.head = _build_head(fdim).to(DEVICE)

    with torch.no_grad():
        model.head[0].weight.copy_(torch.from_numpy(pca_components))
        model.head[0].bias.zero_()
        model.head[3].weight.copy_(torch.from_numpy(centroid_proj_normed))
        model.head[3].bias.zero_()

    n_head = sum(p.numel() for p in model.head.parameters())
    print(f"  Head trainable params : {n_head:,}")

    torch.save({
        "epoch":              0,
        "pretrain_name":      pretrain_name,
        "encoder_state_dict": model.state_dict(),
        "loss":               state.get("loss"),
        "kmeans_init":        True,
        "cluster_to_class":   cluster_to_class,
        "pca_variance_explained": float(explained),
        "config":             state.get("config", {}),
    }, str(out_path))
    print(f"\n  K-means init checkpoint : {out_path}")


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _per_class_metrics(preds: list, targets: list) -> tuple:
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
    images: torch.Tensor, labels: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    lam = float(np.random.beta(alpha, alpha))
    B   = images.size(0)
    idx = torch.randperm(B, device=images.device)
    mixed = lam * images + (1.0 - lam) * images[idx]
    y_a   = torch.zeros(B, NUM_CLASSES, device=labels.device).scatter_(1, labels.unsqueeze(1), 1.0)
    y_b   = torch.zeros(B, NUM_CLASSES, device=labels.device).scatter_(1, labels[idx].unsqueeze(1), 1.0)
    return mixed, lam * y_a + (1.0 - lam) * y_b


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------

def _train_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    optimiser:   torch.optim.Optimizer,
    epoch:       int,
    use_mixup:   bool,
    mixup_alpha: float,
) -> dict:
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

        total_loss += loss.item()
        n_batches  += 1
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
def _validate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> dict:
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

def finetune(config: dict) -> None:
    """
    Frozen-encoder supervised head training.

    Required config keys:
      name, pretrain_name, backbone, use_eca, head_init, mixup_alpha,
      lr, weight_decay, lr_patience, lr_factor,
      batch_size, num_epochs, early_stop, seed
    """
    _set_seed(config["seed"])

    exp_name      = config["name"]
    pretrain_name = config["pretrain_name"]
    ckpt_dir      = CKPT_DIR / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pth"
    csv_path  = ckpt_dir / "training_log.csv"

    print(f"\n{'='*60}")
    print(f"  Finetune : {exp_name}")
    print(f"  Pretrain : {pretrain_name}")
    print(f"  Head init: {config['head_init']}")
    print(f"  Mixup    : {config['mixup_alpha']}")
    print(f"  Device   : {DEVICE}")
    print(f"{'='*60}")

    # ---- Load pretrain checkpoint ----
    if config["head_init"] == "kmeans":
        encoder_ckpt = PRETRAIN_DIR / pretrain_name / "kmeans_head.pth"
    else:
        encoder_ckpt = PRETRAIN_DIR / pretrain_name / "best.pth"

    assert encoder_ckpt.exists(), f"Pretrain checkpoint not found: {encoder_ckpt}"
    state = torch.load(str(encoder_ckpt), map_location=DEVICE)

    # ---- Build model with replaced head ----
    model = _build_encoder(config)
    fdim  = model.feature_dim
    model.head = _build_head(fdim).to(DEVICE)

    # strict=False: encoder keys load; head keys load from kmeans_head or are
    # skipped (size mismatch with default Linear(256,3)) keeping random init.
    model.load_state_dict(state["encoder_state_dict"], strict=False)
    model = model.to(DEVICE)

    # ---- Freeze encoder (all params except head) ----
    for name, param in model.named_parameters():
        if not name.startswith("head"):
            param.requires_grad = False

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {n_total:,}")
    print(f"Trainable (head) : {n_trainable:,}")

    # ---- Datasets ----
    train_ds = NucleiDataset(TRAIN_DIR, augment_level=config.get("augment_level", "improved"))
    val_ds   = NucleiDataset(VAL_DIR,   augment_level="none")
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)

    # ---- Loss, optimiser, scheduler ----
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
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
            model, train_loader, criterion, optimiser, epoch,
            use_mixup=use_mixup, mixup_alpha=config["mixup_alpha"],
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

        scheduler.step(val_m["accuracy"])

        val_acc = val_m["accuracy"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save({
                "epoch":            epoch,
                "exp_name":         exp_name,
                "backbone":         config["backbone"],
                "model_state_dict": model.state_dict(),
                "val_accuracy":     best_val_acc,
                "config": {
                    "backbone": config["backbone"],
                    "use_eca":  config.get("use_eca", True),
                    "fdim":     fdim,
                },
            }, str(best_path))
        else:
            no_improve += 1
            if no_improve >= config["early_stop"]:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    csv_fh.close()
    print(f"\nFinetuning complete.")
    print(f"  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Checkpoint        : {best_path}")
    print(f"  Training log      : {csv_path}")
