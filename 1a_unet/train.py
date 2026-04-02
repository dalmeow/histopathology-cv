import argparse
import sys
from pathlib import Path

# Allow imports from arch/ (this dir) and 1_shared/ (shared data + losses)
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "1_shared"))

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from arch import UNet
from data_processing import TissueDataset
from losses import FocalLoss, DiceLoss, build_primary_criterion, CLASS_WEIGHTS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = _HERE.parent.parent / "Coumputer_Vision_Mini_Project_Data" / "Dataset_Splits"
RUNS_DIR  = _HERE / "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 3):
    """Mean IoU over present classes."""
    ious = []
    preds   = preds.view(-1)
    targets = targets.view(-1)
    for cls in range(num_classes):
        pred_c   = preds   == cls
        target_c = targets == cls
        intersection = (pred_c & target_c).sum().item()
        union        = (pred_c | target_c).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return sum(ious) / len(ious) if ious else 0.0


def compute_dice_per_class(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 3):
    """Per-image Dice averaged over the batch. Skips images where a class is absent
    in both pred and target so absent classes don't drag the mean to zero."""
    if preds.dim() == 2:
        preds   = preds.unsqueeze(0)
        targets = targets.unsqueeze(0)
    totals = [0.0] * num_classes
    counts = [0]   * num_classes
    for b in range(preds.shape[0]):
        p = preds[b].view(-1)
        t = targets[b].view(-1)
        for cls in range(num_classes):
            pred_c   = p == cls
            target_c = t == cls
            denom = pred_c.sum().item() + target_c.sum().item()
            if denom > 0:
                totals[cls] += 2 * (pred_c & target_c).sum().item() / denom
                counts[cls] += 1
    return [totals[c] / counts[c] if counts[c] > 0 else 0.0 for c in range(num_classes)]


def run_epoch(model, loader, criterion_primary, criterion_dice, loss_ratio, optimizer=None):
    """Run one train or val epoch. Pass optimizer=None for val.

    loss = criterion_primary(outputs, masks) + loss_ratio * DiceLoss(outputs, masks)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_iou, n_batches = 0.0, 0.0, 0
    total_dice = [0.0, 0.0, 0.0]

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, masks in loader:
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)
            outputs = model(images)

            def seg_loss(o, t):
                return criterion_primary(o, t) + loss_ratio * criterion_dice(o, t)

            if isinstance(outputs, tuple):
                out_final, out_aux1, out_aux2, out_aux3 = outputs
                target_size = masks.shape[-2:]
                def up(t): return F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
                loss = (seg_loss(out_final, masks)
                      + 0.5   * seg_loss(up(out_aux1), masks)
                      + 0.25  * seg_loss(up(out_aux2), masks)
                      + 0.125 * seg_loss(up(out_aux3), masks))
                outputs = out_final
            else:
                loss = seg_loss(outputs, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(dim=1)
            preds_cpu, masks_cpu = preds.cpu(), masks.cpu()
            total_loss += loss.item()
            total_iou  += compute_iou(preds_cpu, masks_cpu)
            for c, d in enumerate(compute_dice_per_class(preds_cpu, masks_cpu)):
                total_dice[c] += d
            n_batches  += 1

    mean_dice = [d / n_batches for d in total_dice]
    return total_loss / n_batches, total_iou / n_batches, mean_dice


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(args):
    model_dir  = RUNS_DIR / args.run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "unet_tissue.pt"
    best_path  = model_dir / "unet_tissue_best.pt"

    print(f"Device: {DEVICE}")
    print(f"Run dir: {model_dir}")
    print(f"Loss: {args.loss}  |  loss_ratio (dice weight): {args.loss_ratio}  |  focal gamma: {args.focal_gamma}  |  class weights: {not args.no_class_weights}")

    img_size      = getattr(args, "img_size", 512)
    train_dataset = TissueDataset(DATA_ROOT / "train",      augment=True,  img_size=img_size)
    val_dataset   = TissueDataset(DATA_ROOT / "validation", augment=False, img_size=img_size)
    print(f"Train: {len(train_dataset)} patches  |  Val: {len(val_dataset)} patches")

    sample_weights = train_dataset.get_sample_weights(
        cache_path=DATA_ROOT / "train" / "sample_weights.npy"
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,  num_workers=2, pin_memory=True)

    model = UNet(dimensions=3, base=64, use_residual=getattr(args, "use_residual", False)).to(DEVICE)

    if args.resume and model_path.exists():
        print(f"Resuming from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    optimizer         = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler         = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    criterion_primary = build_primary_criterion(args.loss, args.focal_gamma, not args.no_class_weights, DEVICE)
    criterion_dice    = DiceLoss(num_classes=3)

    loss_label = (
        f"focal(gamma={args.focal_gamma}) + {args.loss_ratio}*Dice"
        if args.loss == "focal"
        else f"weighted_CE + {args.loss_ratio}*Dice"
    )

    wandb.init(
        project="tissue-segmentation",
        name=args.run_name,

        config={
            "architecture":  "UNet-Residual" if getattr(args, "use_residual", False) else "UNet",
            "unet_base":     64,
            "batch_size":    args.batch_size,
            "epochs":        args.epochs,
            "optimizer":     "AdamW",
            "lr":            args.lr,
            "weight_decay":  args.weight_decay,
            "loss":          loss_label,
            "loss_type":     args.loss,
            "loss_ratio":    args.loss_ratio,
            "focal_gamma":   args.focal_gamma,
            "class_weights": CLASS_WEIGHTS.tolist() if not args.no_class_weights else None,
            "scheduler":     "ReduceLROnPlateau(mode=max, factor=0.5, patience=10)",
            "patch_size":    img_size,
            "num_classes":   3,
            "augmentation":  "hflip+vflip+rot90+brightness(±10%)+HED_stain",
            "sampling":      "weighted_random_sampler(Other+Stroma_frac,floor=0.1)",
        },
    )
    wandb.watch(model, log="gradients", log_freq=50)

    best_val_iou      = 0.0
    epochs_no_improve = 0

    CLASS_NAMES = ["Tumor", "Stroma", "Other"]

    epoch_bar = tqdm(range(args.epochs), desc=args.run_name)
    for epoch in epoch_bar:
        train_loss, train_iou, train_dice = run_epoch(model, train_loader, criterion_primary, criterion_dice, args.loss_ratio, optimizer)
        val_loss,   val_iou,   val_dice   = run_epoch(model, val_loader,   criterion_primary, criterion_dice, args.loss_ratio)

        scheduler.step(val_iou)
        epoch_bar.set_postfix(val_iou=f"{val_iou:.4f}", val_dice=f"{val_dice[0]:.3f}/{val_dice[1]:.3f}/{val_dice[2]:.3f}")

        wandb.log({
            "epoch":      epoch + 1,
            "train/loss": train_loss,
            "train/mIoU": train_iou,
            "val/loss":   val_loss,
            "val/mIoU":   val_iou,
            "lr":         optimizer.param_groups[0]["lr"],
            **{f"train/dice_{name}": train_dice[i] for i, name in enumerate(CLASS_NAMES)},
            **{f"val/dice_{name}":   val_dice[i]   for i, name in enumerate(CLASS_NAMES)},
        })

        if val_iou > best_val_iou:
            best_val_iou      = val_iou
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            wandb.run.summary["best_val_mIoU"] = best_val_iou
            wandb.run.summary["best_epoch"]    = epoch + 1
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), model_path)

    torch.save(model.state_dict(), model_path)
    wandb.finish()
    print(f"\nTraining complete. Best val mIoU: {best_val_iou:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet for tissue segmentation")
    parser.add_argument("--run-name",      type=str,   default="unet_v1_focal")
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch-size",    type=int,   default=4)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--weight-decay",  type=float, default=1e-2)
    parser.add_argument("--momentum",      type=float, default=0.9)
    parser.add_argument("--save-interval",       type=int,   default=10)
    parser.add_argument("--early-stop-patience", type=int,   default=25)
    parser.add_argument("--resume",              action="store_true")
    parser.add_argument("--loss",        type=str,   default="focal", choices=["focal", "ce"])
    parser.add_argument("--loss-ratio",  type=float, default=2.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()
    train(args)
