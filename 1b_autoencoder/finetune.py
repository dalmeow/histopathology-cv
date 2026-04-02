"""Fine-tuning: frozen encoder + trainable SegDecoder."""
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "1_shared"))

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from arch import Autoencoder
from data_processing import TissueDataset
from losses import DiceLoss, build_primary_criterion, CLASS_WEIGHTS

DATA_ROOT = _HERE.parent.parent / "Coumputer_Vision_Mini_Project_Data" / "Dataset_Splits"
RUNS_DIR  = _HERE / "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Reuse metric helpers from 1a_unet/train.py
# ---------------------------------------------------------------------------
def compute_iou(preds, targets, num_classes=3):
    ious = []
    preds, targets = preds.view(-1), targets.view(-1)
    for cls in range(num_classes):
        pred_c, target_c = preds == cls, targets == cls
        intersection = (pred_c & target_c).sum().item()
        union        = (pred_c | target_c).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return sum(ious) / len(ious) if ious else 0.0


def compute_dice_per_class(preds, targets, num_classes=3):
    if preds.dim() == 2:
        preds, targets = preds.unsqueeze(0), targets.unsqueeze(0)
    totals, counts = [0.0] * num_classes, [0] * num_classes
    for b in range(preds.shape[0]):
        p, t = preds[b].view(-1), targets[b].view(-1)
        for cls in range(num_classes):
            pred_c, target_c = p == cls, t == cls
            denom = pred_c.sum().item() + target_c.sum().item()
            if denom > 0:
                totals[cls] += 2 * (pred_c & target_c).sum().item() / denom
                counts[cls] += 1
    return [totals[c] / counts[c] if counts[c] > 0 else 0.0 for c in range(num_classes)]


# ---------------------------------------------------------------------------
# Fine-tune
# ---------------------------------------------------------------------------

def finetune(args):
    model_dir = RUNS_DIR / args.run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "ae_finetune_best.pt"
    last_path = model_dir / "ae_finetune_last.pt"

    print(f"Device: {DEVICE}")
    print(f"Run: {args.run_name}")

    # Build model in finetune mode
    model = Autoencoder(mode="finetune", base=64, use_residual=args.use_residual).to(DEVICE)

    # Load pretrained encoder weights
    pretrain_ckpt = RUNS_DIR / args.pretrain_run / "ae_best.pt"
    if not pretrain_ckpt.exists():
        raise FileNotFoundError(f"No pretrain checkpoint found at {pretrain_ckpt}")
    state = torch.load(pretrain_ckpt, map_location=DEVICE)
    # Extract only encoder keys
    enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(enc_state)
    print(f"Loaded encoder weights from {pretrain_ckpt}")

    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    print("Encoder frozen.")

    # Optionally init SegDecoder from UNet checkpoint
    if args.unet_dec_init:
        unet_ckpt = _HERE.parent / "1a_unet" / "checkpoints" / args.unet_dec_run / "unet_tissue_best.pt"
        unet_state = torch.load(unet_ckpt, map_location=DEVICE)
        dec_keys   = {k: v for k, v in unet_state.items()
                      if k.startswith(("up1.", "up2.", "up3.", "up4.", "last_conv.", "aux1.", "aux2.", "aux3."))}
        missing, unexpected = model.seg_decoder.load_state_dict(dec_keys, strict=False)
        print(f"Loaded SegDecoder from UNet checkpoint ({unet_ckpt.name}). Missing: {missing}")

    img_size      = getattr(args, "img_size", 512)
    train_dataset = TissueDataset(DATA_ROOT / "train",      augment=True,  img_size=img_size)
    val_dataset   = TissueDataset(DATA_ROOT / "validation", augment=False, img_size=img_size)
    print(f"Train: {len(train_dataset)}  |  Val: {len(val_dataset)}")

    sample_weights = train_dataset.get_sample_weights(
        cache_path=DATA_ROOT / "train" / "sample_weights.npy"
    )
    sampler      = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,    num_workers=2, pin_memory=True)

    # Only optimise SegDecoder parameters
    optimizer         = optim.AdamW(model.seg_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler         = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    criterion_primary = build_primary_criterion("focal", 2.0, False, DEVICE)
    criterion_dice    = DiceLoss(num_classes=3)

    wandb.init(
        project="tissue-segmentation",
        name=args.run_name,
        config={
            "phase":          "finetune",
            "pretrain_run":   args.pretrain_run,
            "unet_dec_init":  args.unet_dec_init,
            "use_residual":   args.use_residual,
            "img_size":       img_size,
            "batch_size":     args.batch_size,
            "epochs":         args.epochs,
            "lr":             args.lr,
            "weight_decay":   args.weight_decay,
            "loss":           "focal + 2.0*Dice",
        },
    )

    best_val_iou      = 0.0
    epochs_no_improve = 0
    CLASS_NAMES       = ["Tumor", "Stroma", "Other"]

    def seg_loss(o, t):
        return criterion_primary(o, t) + 2.0 * criterion_dice(o, t)

    epoch_bar = tqdm(range(args.epochs), desc=args.run_name)
    for epoch in epoch_bar:
        # --- train ---
        model.train()
        train_loss, train_iou, train_dice, n = 0.0, 0.0, [0.0]*3, 0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            out_final, out_aux1, out_aux2, out_aux3 = outputs
            target_size = masks.shape[-2:]
            def up(t): return F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
            loss = (seg_loss(out_final, masks)
                  + 0.5   * seg_loss(up(out_aux1), masks)
                  + 0.25  * seg_loss(up(out_aux2), masks)
                  + 0.125 * seg_loss(up(out_aux3), masks))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            preds = out_final.argmax(dim=1).cpu()
            train_loss += loss.item()
            train_iou  += compute_iou(preds, masks.cpu())
            for c, d in enumerate(compute_dice_per_class(preds, masks.cpu())):
                train_dice[c] += d
            n += 1
        train_loss /= n; train_iou /= n; train_dice = [d/n for d in train_dice]

        # --- val ---
        model.eval()
        val_loss, val_iou, val_dice, n = 0.0, 0.0, [0.0]*3, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                out_final = model(images)
                if isinstance(out_final, tuple):
                    out_final = out_final[0]
                loss  = seg_loss(out_final, masks)
                preds = out_final.argmax(dim=1).cpu()
                val_loss += loss.item()
                val_iou  += compute_iou(preds, masks.cpu())
                for c, d in enumerate(compute_dice_per_class(preds, masks.cpu())):
                    val_dice[c] += d
                n += 1
        val_loss /= n; val_iou /= n; val_dice = [d/n for d in val_dice]

        scheduler.step(val_iou)
        epoch_bar.set_postfix(val_iou=f"{val_iou:.4f}", val_dice=f"{val_dice[0]:.3f}/{val_dice[1]:.3f}/{val_dice[2]:.3f}")
        wandb.log({
            "epoch":      epoch + 1,
            "train/loss": train_loss, "train/mIoU": train_iou,
            "val/loss":   val_loss,   "val/mIoU":   val_iou,
            "lr":         optimizer.param_groups[0]["lr"],
            **{f"train/dice_{n}": train_dice[i] for i, n in enumerate(CLASS_NAMES)},
            **{f"val/dice_{n}":   val_dice[i]   for i, n in enumerate(CLASS_NAMES)},
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

    torch.save(model.state_dict(), last_path)
    wandb.finish()
    print(f"Fine-tuning complete. Best val mIoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name",            type=str,   required=True)
    parser.add_argument("--pretrain-run",        type=str,   required=True,
                        help="name of the pretrain checkpoint dir to load encoder from")
    parser.add_argument("--unet-dec-init",       action="store_true",
                        help="init SegDecoder weights from the best UNet checkpoint")
    parser.add_argument("--unet-dec-run",        type=str,   default="unet_v1_perdice",
                        help="which UNet run to copy decoder weights from")
    parser.add_argument("--use-residual",        action="store_true")
    parser.add_argument("--img-size",            type=int,   default=512)
    parser.add_argument("--epochs",              type=int,   default=100)
    parser.add_argument("--batch-size",          type=int,   default=8)
    parser.add_argument("--lr",                  type=float, default=1e-4)
    parser.add_argument("--weight-decay",        type=float, default=1e-2)
    parser.add_argument("--early-stop-patience", type=int,   default=25)
    args = parser.parse_args()
    finetune(args)
