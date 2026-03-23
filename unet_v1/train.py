import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch import nn
from torch.utils.data import DataLoader
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
# Dice loss
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = logits.softmax(dim=1)                          # (B, C, H, W)
        targets_oh = nn.functional.one_hot(targets, self.num_classes)  # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()     # (B, C, H, W)

        dims = (0, 2, 3)   # reduce over batch, H, W — keep class dim
        intersection = (probs * targets_oh).sum(dim=dims)
        cardinality  = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
        dice_per_cls = (2 * intersection + self.eps) / (cardinality + self.eps)
        return 1 - dice_per_cls.mean()


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


def run_epoch(model, loader, criterion_ce, criterion_dice, optimizer=None):
    """Run one train or val epoch. Pass optimizer=None for val."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_iou, n_batches = 0.0, 0.0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc="train" if is_train else "val  ", leave=False)
        for images, masks in pbar:
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)
            outputs = model(images)

            if isinstance(outputs, tuple):
                out_final, out_aux1, out_aux2, out_aux3 = outputs
                target_size = masks.shape[-2:]
                def up(t): return F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
                def seg_loss(o): return criterion_ce(o, masks) + 2 * criterion_dice(o, masks)
                loss = (seg_loss(out_final)
                      + 0.5   * seg_loss(up(out_aux1))
                      + 0.25  * seg_loss(up(out_aux2))
                      + 0.125 * seg_loss(up(out_aux3)))
                outputs = out_final
            else:
                loss = criterion_ce(outputs, masks) + 2 * criterion_dice(outputs, masks)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds = outputs.argmax(dim=1)
            total_loss += loss.item()
            total_iou  += compute_iou(preds.cpu(), masks.cpu())
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / n_batches, total_iou / n_batches


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

    train_dataset = TissueDataset(DATA_ROOT / "train",      augment=True)
    val_dataset   = TissueDataset(DATA_ROOT / "validation", augment=False)
    print(f"Train: {len(train_dataset)} patches  |  Val: {len(val_dataset)} patches")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(dimensions=3, base=96).to(DEVICE)

    if args.resume and model_path.exists():
        print(f"Resuming from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    optimizer      = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler      = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    criterion_ce   = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))
    criterion_dice = DiceLoss(num_classes=3)

    wandb.init(
        project="tissue-segmentation",
        name=args.run_name,
        config={
            "architecture":  "UNet",
            "batch_size":    args.batch_size,
            "epochs":        args.epochs,
            "optimizer":     "AdamW",
            "lr":            args.lr,
            "weight_decay":  args.weight_decay,
            "loss":          "weighted_CE + 2*Dice",
            "class_weights": CLASS_WEIGHTS.tolist(),
            "scheduler":     "ReduceLROnPlateau(mode=max, factor=0.5, patience=10)",
            "patch_size":    512,
            "num_classes":   3,
            "augmentation":  "hflip+vflip+rot90+color_jitter",
            "sampling":      "4_fixed_crops",
        },
    )
    wandb.watch(model, log="gradients", log_freq=50)

    best_val_iou    = 0.0
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        train_loss, train_iou = run_epoch(model, train_loader, criterion_ce, criterion_dice, optimizer)
        val_loss,   val_iou   = run_epoch(model, val_loader,   criterion_ce, criterion_dice)

        print(
            f"Epoch {epoch+1:3d}/{args.epochs}"
            f"  train loss={train_loss:.4f}  train mIoU={train_iou:.4f}"
            f"  val loss={val_loss:.4f}  val mIoU={val_iou:.4f}"
        )

        scheduler.step(val_iou)

        wandb.log({
            "epoch":      epoch + 1,
            "train/loss": train_loss,
            "train/mIoU": train_iou,
            "val/loss":   val_loss,
            "val/mIoU":   val_iou,
            "lr":         optimizer.param_groups[0]["lr"],
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
    parser.add_argument("--run-name",      type=str,   default="unet_v1_data_v2")
    parser.add_argument("--epochs",        type=int,   default=60)
    parser.add_argument("--batch-size",    type=int,   default=4)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--weight-decay",  type=float, default=1e-8)
    parser.add_argument("--momentum",      type=float, default=0.9)
    parser.add_argument("--save-interval",       type=int,   default=10)
    parser.add_argument("--early-stop-patience", type=int,   default=20)
    parser.add_argument("--resume",              action="store_true")
    args = parser.parse_args()
    train(args)
