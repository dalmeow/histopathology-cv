"""Autoencoder fine-tuning: frozen encoder + trainable SegDecoder."""
import random
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "1_shared"))

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from arch import Autoencoder
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

def _dice_per_class(preds: torch.Tensor, targets: torch.Tensor) -> list[float]:
    """Per-image Dice, skipping absent classes. Returns list of 3 floats."""
    totals, counts = [0.0] * 3, [0] * 3
    for b in range(preds.shape[0]):
        p, t = preds[b].view(-1), targets[b].view(-1)
        for c in range(3):
            pred_c, tgt_c = p == c, t == c
            denom = pred_c.sum().item() + tgt_c.sum().item()
            if denom > 0:
                totals[c] += 2 * (pred_c & tgt_c).sum().item() / denom
                counts[c] += 1
    return [totals[c] / counts[c] if counts[c] > 0 else 0.0 for c in range(3)]


# Entry point

def finetune(config: dict) -> None:
    """Fine-tune the SegDecoder on top of a frozen pre-trained encoder."""
    set_seed(config["seed"])

    exp_name  = config["name"]
    run_dir   = CKPT_DIR / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "ae_finetune_best.pt"
    last_path = run_dir / "ae_finetune_last.pt"

    print(f"\nDevice    : {DEVICE}")
    print(f"Finetune  : {exp_name}")
    print(f"Config    : {config}")

    # --- Build model ---
    model = Autoencoder(mode="finetune", base=64,
                        use_residual=config["use_residual"],
                        use_deep_sup=config["use_deep_sup"],
                        norm_type=config["norm_type"],
                        dropout=config["dropout_p"]).to(DEVICE)

    # Load encoder from pretrain checkpoint
    pretrain_ckpt = CKPT_DIR / config["pretrain_name"] / "ae_best.pt"
    if not pretrain_ckpt.exists():
        raise FileNotFoundError(f"Pre-train checkpoint not found: {pretrain_ckpt}")
    state    = torch.load(pretrain_ckpt, map_location=DEVICE)
    enc_keys = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(enc_keys)
    print(f"Loaded encoder from {pretrain_ckpt}")

    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    print("Encoder frozen.")

    # Data
    train_ds = TissueDataset(DATA_ROOT / "train",      augment=True,
                             augment_hed=config["augment_hed"], img_size=config["img_size"])
    val_ds   = TissueDataset(DATA_ROOT / "validation", augment=False, img_size=config["img_size"])
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)

    # Optimiser & loss (SegDecoder params only)
    optimizer = torch.optim.AdamW(model.seg_decoder.parameters(),
                                  lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10)
    criterion = build_criterion(config["loss_type"], config["use_class_weights"],
                                config["loss_lambda"], DEVICE)

    wandb.init(
        project = "tissue-segmentation",
        name    = exp_name,
        config  = {
            "stage":            "finetune",
            "pretrain_name":    config["pretrain_name"],
            "pretrain_masked":  config["pretrain_masked"],
            "norm_type":        config["norm_type"],
            "use_residual":     config["use_residual"],
            "use_deep_sup":     config["use_deep_sup"],
            "dropout_p":        config["dropout_p"],
            "augment_hed":      config["augment_hed"],
            "loss_type":        config["loss_type"],
            "use_class_weights":config["use_class_weights"],
            "loss_lambda":      config["loss_lambda"],
            "img_size":         config["img_size"],
            "batch_size":       config["batch_size"],
            "epochs":           config["finetune_epochs"],
            "lr":               config["lr"],
            "seed":             config["seed"],
        },
    )

    best_val_loss     = float("inf")
    epochs_no_improve = 0

    def _loss_with_deep_sup(outputs, masks):
        out, aux1, aux2, aux3 = outputs
        sz = masks.shape[-2:]
        up = lambda t: F.interpolate(t, size=sz, mode="bilinear", align_corners=False)
        return (      criterion(out,       masks)
                + 0.5   * criterion(up(aux1), masks)
                + 0.25  * criterion(up(aux2), masks)
                + 0.125 * criterion(up(aux3), masks))

    epoch_bar = tqdm(range(config["finetune_epochs"]), desc=exp_name)
    for epoch in epoch_bar:
        # Train
        model.train()
        train_loss, train_dice, n = 0.0, [0.0] * 3, 0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            if isinstance(outputs, tuple):   # deep supervision on
                loss  = _loss_with_deep_sup(outputs, masks)
                preds = outputs[0].argmax(dim=1).cpu()
            else:
                loss  = criterion(outputs, masks)
                preds = outputs.argmax(dim=1).cpu()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            for c, d in enumerate(_dice_per_class(preds, masks.cpu())):
                train_dice[c] += d
            n += 1

        train_loss /= n
        train_dice  = [d / n for d in train_dice]

        # Val
        model.eval()
        val_loss, val_dice, n = 0.0, [0.0] * 3, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                logits = model(images)       # plain tensor during eval
                val_loss += criterion(logits, masks).item()
                preds    = logits.argmax(dim=1).cpu()
                for c, d in enumerate(_dice_per_class(preds, masks.cpu())):
                    val_dice[c] += d
                n += 1

        val_loss /= n
        val_dice  = [d / n for d in val_dice]
        val_avg_dice = float(np.mean(val_dice))

        scheduler.step(val_loss)
        epoch_bar.set_postfix(
            val_loss=f"{val_loss:.4f}",
            val_dice=f"{val_dice[0]:.3f}/{val_dice[1]:.3f}/{val_dice[2]:.3f}",
        )
        wandb.log({
            "epoch":        epoch + 1,
            "train/loss":   train_loss,
            "val/loss":     val_loss,
            "val/avg_dice": val_avg_dice,
            "lr":           optimizer.param_groups[0]["lr"],
            **{f"train/dice_{n}": train_dice[i] for i, n in enumerate(CLASS_NAMES)},
            **{f"val/dice_{n}":   val_dice[i]   for i, n in enumerate(CLASS_NAMES)},
        })

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

    torch.save(model.state_dict(), last_path)
    wandb.finish()
    print(f"\nFine-tuning done. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints → {run_dir}")
