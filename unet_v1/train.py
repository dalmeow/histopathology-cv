import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data_processing import TissueDataset
from unet import UNet

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path(__file__).parent.parent.parent / "Coumputer_Vision_Mini_Project_Data" / "Dataset_Splits"
RUNS_DIR  = Path(__file__).parent / "model"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")

# Class weights derived from inverse pixel frequency (train set):
#   Tumor 69.7%  Stroma 25.1%  Other 5.3%
CLASS_WEIGHTS = torch.tensor([1/0.697, 1/0.251, 1/0.053])
CLASS_WEIGHTS = CLASS_WEIGHTS.sqrt()                  # soften: ~3.5x ratio instead of 13x
CLASS_WEIGHTS = CLASS_WEIGHTS / CLASS_WEIGHTS.sum()   # normalise to sum to 1


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Multiclass focal loss with optional per-class weights."""
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight  # passed to F.cross_entropy for class balancing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # log-softmax for numerical stability
        log_p  = F.log_softmax(logits, dim=1)                  # (B, C, H, W)
        ce     = F.nll_loss(log_p, targets, weight=self.weight, reduction="none")  # (B, H, W)
        # gather p_t = softmax probability of the true class
        p_t    = torch.exp(-ce)
        focal  = (1 - p_t) ** self.gamma * ce
        return focal.mean()


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs      = logits.softmax(dim=1)                              # (B, C, H, W)
        targets_oh = nn.functional.one_hot(targets, self.num_classes)   # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()            # (B, C, H, W)

        # Reduce over H, W only → (B, C); average over images then classes
        intersection = (probs * targets_oh).sum(dim=(2, 3))            # (B, C)
        cardinality  = probs.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3))
        dice_per_img = (2 * intersection + self.eps) / (cardinality + self.eps)
        return 1 - dice_per_img.mean()


def build_primary_criterion(loss_type: str, gamma: float, use_class_weights: bool, device: torch.device) -> nn.Module:
    """Return the primary (per-pixel) loss criterion."""
    weights = CLASS_WEIGHTS.to(device) if use_class_weights else None
    if loss_type == "focal":
        return FocalLoss(gamma=gamma, weight=weights)
    elif loss_type == "ce":
        return nn.CrossEntropyLoss(weight=weights)
    else:
        raise ValueError(f"Unknown loss type: {loss_type!r}. Choose 'focal' or 'ce'.")


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
        pbar = tqdm(loader, desc="train" if is_train else "val  ", leave=False)
        for images, masks in pbar:
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
            pbar.set_postfix(loss=f"{loss.item():.4f}")

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

    train_dataset = TissueDataset(DATA_ROOT / "train",      augment=True)
    val_dataset   = TissueDataset(DATA_ROOT / "validation", augment=False)
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

    model = UNet(dimensions=3, base=64).to(DEVICE)

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
            "architecture":  "UNet",
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
            "patch_size":    512,
            "num_classes":   3,
            "augmentation":  "hflip+vflip+rot90+brightness(±10%)+HED_stain",
            "sampling":      "weighted_random_sampler(Other+Stroma_frac,floor=0.1)",
        },
    )
    wandb.watch(model, log="gradients", log_freq=50)

    best_val_iou      = 0.0
    epochs_no_improve = 0

    CLASS_NAMES = ["Tumor", "Stroma", "Other"]

    for epoch in range(args.epochs):
        train_loss, train_iou, train_dice = run_epoch(model, train_loader, criterion_primary, criterion_dice, args.loss_ratio, optimizer)
        val_loss,   val_iou,   val_dice   = run_epoch(model, val_loader,   criterion_primary, criterion_dice, args.loss_ratio)

        print(
            f"Epoch {epoch+1:3d}/{args.epochs}"
            f"  train loss={train_loss:.4f}  train mIoU={train_iou:.4f}"
            f"  val loss={val_loss:.4f}  val mIoU={val_iou:.4f}"
            f"  val Dice=({val_dice[0]:.3f}/{val_dice[1]:.3f}/{val_dice[2]:.3f})"
        )

        scheduler.step(val_iou)

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
            print(f"  → New best val mIoU {best_val_iou:.4f} — saved to {best_path}")
            wandb.run.summary["best_val_mIoU"] = best_val_iou
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(f"  → Early stopping triggered after {epoch+1} epochs")
                break

        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), model_path)
            print(f"  → Checkpoint saved to {model_path}")

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
    # Loss configuration
    parser.add_argument("--loss",        type=str,   default="focal",
                        choices=["focal", "ce"],
                        help="Primary loss function (default: focal)")
    parser.add_argument("--loss-ratio",  type=float, default=2.0,
                        help="Weight multiplier on DiceLoss (default: 2.0 → primary + 2*Dice)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss focusing parameter gamma (default: 2.0)")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable class weighting in the primary loss (default: weights enabled)")
    args = parser.parse_args()
    train(args)
