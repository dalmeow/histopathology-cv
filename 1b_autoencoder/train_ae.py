"""Autoencoder pre-training.

Phase 1 (default):  vanilla MSE on masked regions.
Phase 2 (--masked): masked MAE-style — zero out 75% of input patches,
                    MSE + 0.5*(1-SSIM) on masked regions only.
"""
import argparse
import random
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "1_shared"))

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from arch import Autoencoder
from data_processing import TissueDataset

DATA_ROOT = _HERE.parent.parent / "Coumputer_Vision_Mini_Project_Data" / "Dataset_Splits"
RUNS_DIR  = _HERE / "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")

MASK_RATIO      = 0.75
MASK_PATCH_SIZE = 80
SSIM_WEIGHT     = 0.5


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def make_patch_mask(batch_size: int, img_size: int, patch_size: int, ratio: float, device: torch.device) -> torch.Tensor:
    """Returns (B, 1, H, W) binary mask where 1 = masked (zeroed in input, used in loss)."""
    n_patches = max(1, round(ratio * img_size * img_size / (patch_size * patch_size)))
    mask = torch.zeros(batch_size, 1, img_size, img_size, device=device)
    for b in range(batch_size):
        for _ in range(n_patches):
            y = random.randint(0, img_size - patch_size)
            x = random.randint(0, img_size - patch_size)
            mask[b, :, y:y + patch_size, x:x + patch_size] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def pretrain_loss(pred: torch.Tensor, raw: torch.Tensor, mask: torch.Tensor, use_ssim: bool) -> torch.Tensor:
    """MSE on masked pixels, optionally + 0.5*(1-SSIM) on masked patches."""
    # Masked MSE
    mse = ((pred - raw) ** 2 * mask).sum() / (mask.sum() * pred.shape[1]).clamp(min=1)

    if not use_ssim:
        return mse

    # SSIM on masked patches — extract patches from pred and raw, batch them
    try:
        from pytorch_msssim import ssim
    except ImportError:
        return mse

    B, C, H, W = pred.shape
    patch_size  = MASK_PATCH_SIZE
    patches_pred, patches_raw = [], []

    for b in range(B):
        m = mask[b, 0]  # (H, W)
        # Find unique patch top-left corners that are fully masked
        ys = range(0, H - patch_size + 1, patch_size)
        xs = range(0, W - patch_size + 1, patch_size)
        for y in ys:
            for x in xs:
                if m[y:y + patch_size, x:x + patch_size].mean() > 0.5:
                    patches_pred.append(pred[b, :, y:y + patch_size, x:x + patch_size])
                    patches_raw.append(raw[b,  :, y:y + patch_size, x:x + patch_size])

    if not patches_pred:
        return mse

    pp = torch.stack(patches_pred)  # (N, 3, patch, patch)
    pr = torch.stack(patches_raw)
    ssim_val = ssim(pp, pr, data_range=1.0, size_average=True)
    return mse + SSIM_WEIGHT * (1.0 - ssim_val)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(args):
    model_dir  = RUNS_DIR / args.run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path  = model_dir / "ae_best.pt"
    last_path  = model_dir / "ae_last.pt"

    print(f"Device: {DEVICE}")
    print(f"Run: {args.run_name}  |  masked: {args.masked}")

    train_dataset = TissueDataset(DATA_ROOT / "train", augment=True, img_size=args.img_size, return_raw=True)
    val_dataset   = TissueDataset(DATA_ROOT / "validation", augment=False, img_size=args.img_size, return_raw=True)
    print(f"Train: {len(train_dataset)}  |  Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model     = Autoencoder(mode="pretrain", base=64, use_residual=args.use_residual).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    wandb.init(
        project="tissue-segmentation",
        name=args.run_name,
        config={
            "phase":         "pretrain",
            "masked":        args.masked,
            "mask_ratio":    MASK_RATIO if args.masked else None,
            "mask_patch":    MASK_PATCH_SIZE if args.masked else None,
            "ssim_weight":   SSIM_WEIGHT if args.masked else None,
            "architecture":  "Autoencoder",
            "use_residual":  args.use_residual,
            "img_size":      args.img_size,
            "batch_size":    args.batch_size,
            "epochs":        args.epochs,
            "lr":            args.lr,
            "weight_decay":  args.weight_decay,
        },
    )

    best_val_loss     = float("inf")
    epochs_no_improve = 0

    epoch_bar = tqdm(range(args.epochs), desc=args.run_name)
    for epoch in epoch_bar:
        # --- train ---
        model.train()
        train_loss, n = 0.0, 0
        for images, raw, _ in train_loader:
            images, raw = images.to(DEVICE), raw.to(DEVICE)

            if args.masked:
                mask = make_patch_mask(images.shape[0], args.img_size, MASK_PATCH_SIZE, MASK_RATIO, DEVICE)
                inp  = images * (1.0 - mask)
            else:
                mask = torch.ones(images.shape[0], 1, args.img_size, args.img_size, device=DEVICE)
                inp  = images

            pred = model(inp)
            loss = pretrain_loss(pred, raw, mask, use_ssim=args.masked)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item(); n += 1

        train_loss /= n

        # --- val ---
        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for images, raw, _ in val_loader:
                images, raw = images.to(DEVICE), raw.to(DEVICE)

                if args.masked:
                    mask = make_patch_mask(images.shape[0], args.img_size, MASK_PATCH_SIZE, MASK_RATIO, DEVICE)
                    inp  = images * (1.0 - mask)
                else:
                    mask = torch.ones(images.shape[0], 1, args.img_size, args.img_size, device=DEVICE)
                    inp  = images

                pred = model(inp)
                loss = pretrain_loss(pred, raw, mask, use_ssim=args.masked)
                val_loss += loss.item(); n += 1

        val_loss /= n
        scheduler.step(val_loss)

        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")
        wandb.log({"epoch": epoch+1, "train/loss": train_loss, "val/loss": val_loss,
                   "lr": optimizer.param_groups[0]["lr"]})

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"]    = epoch + 1
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    torch.save(model.state_dict(), last_path)
    wandb.finish()
    print(f"Pre-training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name",            type=str,   required=True)
    parser.add_argument("--masked",              action="store_true")
    parser.add_argument("--use-residual",        action="store_true")
    parser.add_argument("--img-size",            type=int,   default=512)
    parser.add_argument("--epochs",              type=int,   default=100)
    parser.add_argument("--batch-size",          type=int,   default=8)
    parser.add_argument("--lr",                  type=float, default=1e-4)
    parser.add_argument("--weight-decay",        type=float, default=1e-2)
    parser.add_argument("--early-stop-patience", type=int,   default=25)
    args = parser.parse_args()
    train(args)
