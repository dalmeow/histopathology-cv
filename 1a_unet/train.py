import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "1_shared"))

from arch import UNet
from data_processing import TissueDataset
from losses import build_criterion


DATA_ROOT = _HERE.parent.parent / "Coumputer_Vision_Mini_Project_Data" / "Dataset_Splits"
CKPT_DIR  = _HERE / "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")

CLASS_NAMES = ["Tumor", "Stroma", "Other"]


# Reproducibility

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# Metrics

def compute_dice_per_class(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3,
) -> list[float]:
    """
    Per-image Dice averaged over the batch.
    Skips images where a class is absent in both pred and target so that
    absent classes don't drag the mean toward zero.
    """
    if preds.dim() == 2:
        preds   = preds.unsqueeze(0)
        targets = targets.unsqueeze(0)

    totals = [0.0] * num_classes
    counts = [0]   * num_classes

    for b in range(preds.shape[0]):
        p = preds[b].view(-1)
        t = targets[b].view(-1)
        for c in range(num_classes):
            pred_c   = p == c
            target_c = t == c
            denom = pred_c.sum().item() + target_c.sum().item()
            if denom > 0:
                totals[c] += 2 * (pred_c & target_c).sum().item() / denom
                counts[c] += 1

    return [totals[c] / counts[c] if counts[c] > 0 else 0.0
            for c in range(num_classes)]


# Epoch runner

def run_epoch(model, loader, criterion, optimizer=None):
    """
    Run one train or validation epoch.
    Pass optimizer=None for validation.
    Returns (mean_loss, mean_dice_per_class).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss  = 0.0
    total_dice  = [0.0] * 3
    n_batches   = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, masks in loader:
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)

            output = model(images)

            if isinstance(output, tuple):
                # Deep supervision: main + aux1 + aux2 + aux3 (all at input resolution)
                main, aux1, aux2, aux3 = output
                loss = (criterion(main, masks)
                      + 0.5   * criterion(aux1, masks)
                      + 0.25  * criterion(aux2, masks)
                      + 0.125 * criterion(aux3, masks))
                logits = main
            else:
                loss   = criterion(output, masks)
                logits = output

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1).cpu()
            masks_cpu = masks.cpu()

            total_loss += loss.item()
            for c, d in enumerate(compute_dice_per_class(preds, masks_cpu)):
                total_dice[c] += d
            n_batches += 1

    mean_dice = [d / n_batches for d in total_dice]
    return total_loss / n_batches, mean_dice


# Train

def train(config: dict) -> None:
    set_seed(config["seed"])

    run_dir = CKPT_DIR / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDevice : {DEVICE}")
    print(f"Run    : {config['name']}")
    print(f"Config : {config}\n")

    # Datasets
    train_ds = TissueDataset(
        DATA_ROOT / "train",
        augment=True,
        augment_hed=config["augment_hed"],
        img_size=config["img_size"],
        other_threshold=config.get("other_threshold", 0.0),
        other_oversample_k=config.get("other_oversample_k", 0),
    )
    val_ds = TissueDataset(
        DATA_ROOT / "validation",
        augment=False,
        img_size=config["img_size"],
    )
    print(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"],
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # Model
    model = UNet(
        num_classes  = 3,
        base_filters = 64,
        norm_type    = config["norm_type"],
        use_residual = config["use_residual"],
        use_deep_sup = config["use_deep_sup"],
        dropout_p    = config["dropout_p"],
    ).to(DEVICE)

    # Loss
    criterion = build_criterion(
        loss_type         = config["loss_type"],
        use_class_weights = config["use_class_weights"],
        loss_lambda       = config["loss_lambda"],
        device            = DEVICE,
    )

    # Optimiser + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )

    # WandB
    wandb.init(
        project = "tissue-segmentation",
        name    = config["name"],
        config  = {**config, "device": str(DEVICE)},
    )
    wandb.watch(model, log="gradients", log_freq=50)

    best_val_loss     = float("inf")
    epochs_no_improve = 0
    best_path         = run_dir / "best_model.pt"
    last_path         = run_dir / "last_model.pt"

    epoch_bar = tqdm(range(config["epochs"]), desc=config["name"])
    for epoch in epoch_bar:
        train_loss, train_dice = run_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_dice   = run_epoch(model, val_loader,   criterion)

        val_avg_dice = float(np.mean(val_dice))
        scheduler.step(val_loss)

        epoch_bar.set_postfix(
            val_loss=f"{val_loss:.4f}",
            val_dice=f"{val_dice[0]:.3f}/{val_dice[1]:.3f}/{val_dice[2]:.3f}",
        )

        wandb.log({
            "epoch":           epoch + 1,
            "lr":              optimizer.param_groups[0]["lr"],
            "train/loss":      train_loss,
            "val/loss":        val_loss,
            "val/avg_dice":    val_avg_dice,
            **{f"train/dice_{n}": train_dice[i] for i, n in enumerate(CLASS_NAMES)},
            **{f"val/dice_{n}":   val_dice[i]   for i, n in enumerate(CLASS_NAMES)},
        })

        # Best checkpoint
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"]    = epoch + 1
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Always save last checkpoint
    torch.save(model.state_dict(), last_path)
    wandb.finish()
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {run_dir}")
