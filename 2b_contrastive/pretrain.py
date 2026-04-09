"""
SimCLR pre-training for Task 2b.

Entry point: pretrain(config) — called by run.py, not directly.

Saves best encoder checkpoint to:
  checkpoints/pretrain/{config["pretrain_name"]}/best.pth

Pre-training is shared across experiments with the same pretrain_name:
  nuclresnet_strong  — exp 1 + 2
  nuclresnet_moderate — exp 3
  resnet18            — exp 4
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from dataset import SimCLRDataset, _SimCLRAug, _R18Aug
from model   import NucleiResNet, ResNet18Encoder, SimCLRProjectionHead, NTXentLoss

DATA_ROOT       = _HERE.parent.parent / "Coumputer_Vision_Mini_Project_Data"
CONTRASTIVE_DIR = str(DATA_ROOT / "task2_patches" / "contrastive")
PATCH_DIRS      = [CONTRASTIVE_DIR]   # dedicated leakage-safe set; never train/ or val/
PRETRAIN_DIR    = _HERE / "checkpoints" / "pretrain"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)


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
# Entry point
# ---------------------------------------------------------------------------

def pretrain(config: dict) -> None:
    """
    SimCLR pre-training loop.

    Required config keys:
      pretrain_name, backbone, color_strength, use_eca,
      pretrain_batch_size, pretrain_epochs, pretrain_warmup,
      pretrain_lr, pretrain_wd, pretrain_temp, pretrain_proj_dim,
      pretrain_early_stop, seed
    """
    _set_seed(config["seed"])

    pretrain_name = config["pretrain_name"]
    ckpt_dir      = PRETRAIN_DIR / pretrain_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pth"
    csv_path  = ckpt_dir / "pretrain_log.csv"

    print(f"\n{'='*60}")
    print(f"  SimCLR Pre-training : {pretrain_name}")
    print(f"  Backbone            : {config['backbone']}")
    if config["backbone"] != "resnet18":
        print(f"  Color strength      : {config['color_strength']}")
    print(f"  Device              : {DEVICE}")
    print(f"{'='*60}")

    # ---- Augmentation & dataset ----
    if config["backbone"] == "resnet18":
        aug = _R18Aug()
    else:
        aug = _SimCLRAug(color_strength=config["color_strength"])

    ds = SimCLRDataset(PATCH_DIRS, aug=aug)
    print(f"Contrastive patches : {len(ds):,}")

    loader = DataLoader(
        ds,
        batch_size=config["pretrain_batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Model ----
    if config["backbone"] == "resnet18":
        encoder = ResNet18Encoder().to(DEVICE)
    else:
        encoder = NucleiResNet(use_eca=config.get("use_eca", True)).to(DEVICE)

    proj_head = SimCLRProjectionHead(
        in_dim=encoder.feature_dim,
        hid_dim=256,
        out_dim=config["pretrain_proj_dim"],
    ).to(DEVICE)

    print(f"Encoder params      : {encoder.count_parameters():,}")
    print(f"Proj head params    : {sum(p.numel() for p in proj_head.parameters()):,}")

    # ---- Loss & optimiser ----
    criterion = NTXentLoss(temperature=config["pretrain_temp"])
    optimiser = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=config["pretrain_lr"],
        weight_decay=config["pretrain_wd"],
    )

    warmup_epochs = config["pretrain_warmup"]
    warmup_sched  = torch.optim.lr_scheduler.LinearLR(
        optimiser, start_factor=0.01, total_iters=warmup_epochs,
    )
    cosine_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=config["pretrain_epochs"] - warmup_epochs,
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimiser,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs],
    )

    print(f"\nStarting pre-training for up to {config['pretrain_epochs']} epochs …")
    print(f"  Warmup  : {warmup_epochs} epochs (linear 0.01→1.0)")
    print(f"  Cosine  : {config['pretrain_epochs'] - warmup_epochs} epochs → 1e-6\n")

    # ---- CSV logger ----
    csv_fh = open(str(csv_path), "w", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=["epoch", "loss", "lr"])
    writer.writeheader()

    # ---- Training loop ----
    best_loss  = float("inf")
    no_improve = 0
    patience   = config["pretrain_early_stop"]

    epoch_bar = tqdm(range(1, config["pretrain_epochs"] + 1), desc=pretrain_name)
    for epoch in epoch_bar:
        encoder.train()
        proj_head.train()
        total_loss, n_batches = 0.0, 0

        for view1, view2 in loader:
            view1 = view1.to(DEVICE, non_blocking=True)
            view2 = view2.to(DEVICE, non_blocking=True)

            h1 = encoder.get_features(view1)
            h2 = encoder.get_features(view2)
            z1 = proj_head(h1)
            z2 = proj_head(h2)
            loss = criterion(z1, z2)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(proj_head.parameters()),
                max_norm=1.0,
            )
            optimiser.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        avg_loss = total_loss / n_batches
        lr       = optimiser.param_groups[0]["lr"]

        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.1e}")
        writer.writerow({"epoch": epoch, "loss": round(avg_loss, 6), "lr": round(lr, 8)})
        csv_fh.flush()

        if avg_loss < best_loss:
            best_loss  = avg_loss
            no_improve = 0
            torch.save({
                "epoch":              epoch,
                "pretrain_name":      pretrain_name,
                "encoder_state_dict": encoder.state_dict(),
                "proj_state_dict":    proj_head.state_dict(),
                "loss":               best_loss,
                "config": {
                    "backbone":    config["backbone"],
                    "use_eca":     config.get("use_eca", True),
                    "temperature": config["pretrain_temp"],
                    "proj_dim":    config["pretrain_proj_dim"],
                },
            }, str(best_path))
        else:
            no_improve += 1
            if no_improve >= patience:
                tqdm.write(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs)."
                )
                break

    csv_fh.close()
    print(f"\nPre-training complete.")
    print(f"  Best NT-Xent loss : {best_loss:.4f}")
    print(f"  Checkpoint        : {best_path}")
    print(f"  Training log      : {csv_path}")
