# UNet v1 — Experiment Log

## Results Summary

| Run | Tumor | Stroma | Other | Avg Dice | Eval Method |
|-----|-------|--------|-------|----------|-------------|
| baseline | 0.822 | 0.264 | 0.003 | 0.363 | center crop |
| data_v2 | 0.869 | 0.006 | 0.000 | 0.292 | center crop |
| loss_v2 | 0.736 | 0.122 | 0.033 | 0.297 | center crop |
| loss_v3 | 0.874 | 0.128 | 0.005 | 0.336 | center crop |
| loss_v3 | 0.838 | 0.157 | 0.005 | 0.333 | sliding window |
| 4crops + CE+Dice(1x) | 0.809 | 0.318 | 0.008 | 0.378 | sliding window |
| 4crops + CE+Dice(2x) | 0.826 | 0.296 | 0.017 | **0.380** | sliding window |
| 4crops + CE+Dice(2x) resumed 120ep | 0.819 | 0.294 | 0.007 | 0.373 | sliding window |
| resize + RMSprop + scheduler | 0.817 | 0.250 | 0.007 | 0.358 | direct |
| resize + AdamW + scheduler | 0.826 | 0.342 | 0.003 | **0.390** | direct |
| resize + AdamW + TTA | 0.830 | 0.336 | 0.003 | 0.390 | TTA |
| deep supervision + instance norm | 0.847 | 0.303 | 0.031 | 0.394 | direct |
| + dropout (p=0.3) | 0.852 | 0.289 | 0.041 | 0.394 | direct |
| + wider (96 base, ~70M params) | 0.840 | 0.299 | 0.077 | **0.405** | direct |

---

## Experiment Details

### baseline (`unet_v1`)
- Resize 1024×1024 → 512×512
- ImageNet normalisation
- Augmentation: H-flip + V-flip
- Loss: CrossEntropyLoss (unweighted)
- Optimiser: RMSprop (lr=1e-4, momentum=0.9, wd=1e-8)
- Epochs: 60

### data_v2
- **New:** Random 512×512 patch crop from native 1024×1024 (no resize)
- **New:** Dataset-specific normalisation (mean=[0.620, 0.412, 0.696], std=[0.198, 0.194, 0.138])
- **New:** Augmentation: H-flip + V-flip + random 90° rotation + colour jitter
- Loss/optimiser unchanged
- Epochs: 60
- *Model collapsed to predicting Tumor everywhere — loss weighting needed*

### loss_v2
- Data pipeline: same as data_v2
- **New:** Weighted CE (inverse frequency, 13× ratio Other/Tumor) + Dice loss (1×)
- Epochs: 60
- *Weights too aggressive — Tumor dropped, overcorrected toward rare classes*

### loss_v3
- **New:** Softened class weights via sqrt (3.5× ratio instead of 13×)
- **New:** Switched eval to sliding window (3×3 grid, stride=256)
- Epochs: 60

### 4crops + CE+Dice(1x)
- **New:** All 4 non-overlapping 512×512 crops per image instead of random single crop
- Loss: weighted CE + Dice(1×)
- *4 crops gave biggest Stroma jump — model sees full image every epoch*

### 4crops + CE+Dice(2x)
- **New:** Dice loss weighted 2× (loss = CE + 2×Dice)
- Epochs: 60 — *first run to beat baseline (0.380)*
- Resume to 120 epochs degraded performance — model overfit, LR too high

### resize + RMSprop + scheduler
- **New:** Reverted to full-image resize (1024→512)
- **New:** ReduceLROnPlateau scheduler (factor=0.5, patience=10)
- Optimiser: RMSprop
- Epochs: 100
- *Resize worse than 4 crops with RMSprop*

### resize + AdamW + scheduler
- **New:** AdamW optimiser (lr=1e-4, wd=1e-2)
- Resize + scheduler unchanged
- Epochs: 100
- *Best run — AdamW improved Stroma significantly (0.342)*

### resize + AdamW + TTA
- Same model as above (no retraining)
- **New:** Test-time augmentation: 8 transforms (4 rotations × hflip), average softmax probs
- *TTA gave no improvement — model predictions already stable*

### deep supervision + instance norm
- **New:** Auxiliary loss heads at up1, up2, up3 decoder stages (weights: 1.0, 0.5, 0.25, 0.125)
- **New:** BatchNorm → InstanceNorm throughout (more robust to stain variation)
- All else same as resize + AdamW + scheduler
- *Other class improved significantly (0.003 → 0.031); train mIoU reached 0.80, best val mIoU 0.59*

### + dropout (p=0.3)
- **New:** Dropout2d (p=0.3) at bottleneck and first two decoder stages
- Addressing train/val gap (train mIoU 0.80 vs test Dice 0.39)
- *Other improved further (0.041) but Stroma dropped; avg unchanged at 0.394*

### + wider (96 base, ~70M params)
- **New:** Filter counts scaled from 64 base to 96 base (64→128→256→512→1024 to 96→192→384→768→1536)
- ~70M parameters vs ~31M previously
- *Clear improvement in Other (0.077) — first run above 0.40 avg Dice*
