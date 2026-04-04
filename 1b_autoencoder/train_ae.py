"""Autoencoder pre-training (reconstruction objective).

Vanilla MSE (masked=False): reconstruct full image from full input.
Masked MAE  (masked=True):  zero out 75% of patches, reconstruct them
                             with MSE + 0.5*(1-SSIM) on masked regions.
"""
import random
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "1_shared"))

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from arch import Autoencoder
from data_processing import TissueDataset

DATA_ROOT = _HERE.parent.parent / "Coumputer_Vision_Mini_Project_Data" / "Dataset_Splits"
CKPT_DIR  = _HERE / "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")

MASK_RATIO      = 0.75
MASK_PATCH_SIZE = 40    # scaled for 512×512 (same coverage as 80px at 1024×1024)
SSIM_WEIGHT     = 0.5

# For unnormalising tensors → [0, 1] raw RGB (reconstruction target)
_MEAN = torch.tensor([0.6199, 0.4123, 0.6963]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.1975, 0.1944, 0.1381]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def make_patch_mask(
    batch_size: int, img_size: int, patch_size: int, ratio: float, device: torch.device
) -> torch.Tensor:
    """Returns (B, 1, H, W) binary mask where 1 = masked region."""
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

def pretrain_loss(
    pred: torch.Tensor, raw: torch.Tensor, mask: torch.Tensor, use_ssim: bool
) -> torch.Tensor:
    """MSE (+ optional SSIM) on masked pixels only."""
    mse = ((pred - raw) ** 2 * mask).sum() / (mask.sum() * pred.shape[1]).clamp(min=1)

    if not use_ssim:
        return mse

    try:
        from pytorch_msssim import ssim
    except ImportError:
        return mse

    B, C, H, W = pred.shape
    patch_size  = MASK_PATCH_SIZE
    patches_pred, patches_raw = [], []

    for b in range(B):
        m  = mask[b, 0]
        ys = range(0, H - patch_size + 1, patch_size)
        xs = range(0, W - patch_size + 1, patch_size)
        for y in ys:
            for x in xs:
                if m[y:y + patch_size, x:x + patch_size].mean() > 0.5:
                    patches_pred.append(pred[b, :, y:y + patch_size, x:x + patch_size])
                    patches_raw.append(raw[b,  :, y:y + patch_size, x:x + patch_size])

    if not patches_pred:
        return mse

    pp       = torch.stack(patches_pred)
    pr       = torch.stack(patches_raw)
    ssim_val = ssim(pp, pr, data_range=1.0, size_average=True)
    return mse + SSIM_WEIGHT * (1.0 - ssim_val)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train_pretrain(config: dict) -> None:
    """Run autoencoder pre-training for the given experiment config."""
    set_seed(config["seed"])

    run_name = config["pretrain_name"]
    masked   = config["pretrain_masked"]
    img_size = config["img_size"]

    run_dir   = CKPT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "ae_best.pt"
    last_path = run_dir / "ae_last.pt"

    print(f"\nDevice      : {DEVICE}")
    print(f"Pre-training: {run_name}  |  masked: {masked}")

    train_ds = TissueDataset(DATA_ROOT / "train",      augment=True,
                             augment_hed=config["augment_hed"], img_size=img_size)
    val_ds   = TissueDataset(DATA_ROOT / "validation", augment=False, img_size=img_size)
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)

    model     = Autoencoder(mode="pretrain", base=64,
                            norm_type=config["norm_type"],
                            dropout=config["dropout_p"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10)

    wandb.init(
        project = "tissue-segmentation",
        name    = run_name,
        config  = {
            "stage":       "pretrain",
            "norm_type":   config["norm_type"],
            "masked":      masked,
            "mask_ratio":  MASK_RATIO      if masked else None,
            "mask_patch":  MASK_PATCH_SIZE if masked else None,
            "ssim_weight": SSIM_WEIGHT     if masked else None,
            "img_size":    img_size,
            "batch_size":  config["batch_size"],
            "epochs":      config["pretrain_epochs"],
            "lr":          config["lr"],
            "seed":        config["seed"],
        },
    )

    _mean = _MEAN.to(DEVICE)
    _std  = _STD.to(DEVICE)

    best_val_loss     = float("inf")
    epochs_no_improve = 0

    epoch_bar = tqdm(range(config["pretrain_epochs"]), desc=run_name)
    for epoch in epoch_bar:
        # --- train ---
        model.train()
        train_loss, n = 0.0, 0
        for images, _ in train_loader:
            images = images.to(DEVICE)
            raw    = (images * _std + _mean).clamp(0, 1)

            if masked:
                patch_mask = make_patch_mask(images.shape[0], img_size, MASK_PATCH_SIZE, MASK_RATIO, DEVICE)
                inp        = images * (1.0 - patch_mask)
            else:
                patch_mask = torch.ones(images.shape[0], 1, img_size, img_size, device=DEVICE)
                inp        = images

            pred = model(inp)
            loss = pretrain_loss(pred, raw, patch_mask, use_ssim=masked)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item(); n += 1

        train_loss /= n

        # --- val ---
        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(DEVICE)
                raw    = (images * _std + _mean).clamp(0, 1)

                if masked:
                    patch_mask = make_patch_mask(images.shape[0], img_size, MASK_PATCH_SIZE, MASK_RATIO, DEVICE)
                    inp        = images * (1.0 - patch_mask)
                else:
                    patch_mask = torch.ones(images.shape[0], 1, img_size, img_size, device=DEVICE)
                    inp        = images

                pred      = model(inp)
                val_loss += pretrain_loss(pred, raw, patch_mask, use_ssim=masked).item()
                n        += 1

        val_loss /= n
        scheduler.step(val_loss)
        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train/loss": train_loss, "val/loss": val_loss,
                   "lr": optimizer.param_groups[0]["lr"]})

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
    print(f"Pre-training done. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints → {run_dir}")
