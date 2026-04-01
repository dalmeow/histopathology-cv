"""UNet v1 — Tissue Segmentation — local training & evaluation script."""

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
UNET_DIR  = Path(__file__).parent
DATA_ROOT = UNET_DIR.parent.parent / 'Coumputer_Vision_Mini_Project_Data' / 'Dataset_Splits'
RUN_NAME  = 'unet_v1_perdice'
MODEL_DIR = UNET_DIR / 'model' / RUN_NAME
BEST_PATH = MODEL_DIR / 'unet_tissue_best.pt'

sys.path.insert(0, str(UNET_DIR))

from unet import UNet
from data_processing import TissueDataset

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = ['Tumor', 'Stroma', 'Other']
CLASS_COLORS = np.array([
    [200,   0,   0],   # Tumor  — red
    [  0, 200,   0],   # Stroma — green
    [  0,   0, 200],   # Other  — blue
], dtype=np.uint8)
MEAN = np.array([0.6199, 0.4123, 0.6963])
STD  = np.array([0.1975, 0.1944, 0.1381])


def mask_to_rgb(mask_tensor):
    return CLASS_COLORS[mask_tensor.numpy()]


def unnorm(t):
    return np.clip(t.permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    subprocess.run([
        sys.executable, str(UNET_DIR / 'train.py'),
        '--run-name',            RUN_NAME,
        '--epochs',              '100',
        '--batch-size',          '32',
        '--lr',                  '1e-4',
        '--weight-decay',        '1e-2',
        '--save-interval',       '10',
        '--early-stop-patience', '25',
        '--loss',                'focal',
        '--loss-ratio',          '2.0',
        '--focal-gamma',         '2.0',
        '--no-class-weights',
    ], check=True)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, device):
    test_ds     = TissueDataset(DATA_ROOT / 'test', augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    per_class_dice = {c: [] for c in range(3)}
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc='Test'):
            preds = model(images.to(device)).argmax(dim=1).cpu().view(-1)
            masks = masks.view(-1)
            for cls in range(3):
                pred_c       = preds == cls
                target_c     = masks == cls
                intersection = (pred_c & target_c).sum().item()
                denom        = pred_c.sum().item() + target_c.sum().item()
                if denom > 0:
                    per_class_dice[cls].append(2 * intersection / denom)

    print('\nTest results:')
    mean_dices = []
    for cls, name in enumerate(CLASS_NAMES):
        d = np.mean(per_class_dice[cls]) if per_class_dice[cls] else 0.0
        mean_dices.append(d)
        print(f'  {name:<10s}  Dice = {d:.4f}')
    print(f'  {"Avg":<10s}  Dice = {np.mean(mean_dices):.4f}')

    return test_ds


# ── Qualitative results ───────────────────────────────────────────────────────
def qualitative(model, test_ds, device, n_show=4):
    fig, axes = plt.subplots(n_show, 3, figsize=(10, 3.5 * n_show))
    fig.suptitle('Image  |  Ground Truth  |  Prediction', fontsize=12)

    for i in range(n_show):
        img_t, mask_t = test_ds[i]
        with torch.no_grad():
            pred = model(img_t.unsqueeze(0).to(device)).argmax(dim=1).squeeze().cpu()

        axes[i][0].imshow(unnorm(img_t))
        axes[i][0].axis('off')
        axes[i][1].imshow(mask_to_rgb(mask_t))
        axes[i][1].axis('off')
        axes[i][2].imshow(mask_to_rgb(pred))
        axes[i][2].axis('off')
        if i == 0:
            axes[i][0].set_title('Image')
            axes[i][1].set_title('Ground Truth')
            axes[i][2].set_title('Prediction')

    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'qualitative_results.png', dpi=100)
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    wandb.login(key="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")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('unet_v1 dir :', UNET_DIR)
    print('data root   :', DATA_ROOT, '| exists:', DATA_ROOT.exists())
    print('device      :', device)
    if device.type == 'cuda':
        print('GPU  :', torch.cuda.get_device_name(0))
        print('VRAM :', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')

    train()

    model = UNet(dimensions=3, base=64).to(device)
    model.load_state_dict(torch.load(BEST_PATH, map_location=device))
    model.eval()

    test_ds = evaluate(model, device)
    qualitative(model, test_ds, device)
