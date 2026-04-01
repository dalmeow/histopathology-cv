# UNet — Tissue Segmentation

5-level UNet with InstanceNorm, deep supervision, and dropout for 3-class tissue segmentation (Tumor / Stroma / Other).

## Results

| Run | Tumor | Stroma | Other | Avg Dice |
|-----|-------|--------|-------|----------|
| Best (`unet_v1_perdice`) | 0.815 | 0.352 | 0.081 | **0.416** |

See [experiments.md](experiments.md) for the full log of 27 runs.

## Usage

Edit the config constants at the top of `run.py` (run name, hyperparameters), then:

```bash
# From this directory (unet/)
python run.py train    # train only
python run.py eval     # evaluate best checkpoint on test set
python run.py both     # train then eval
```

For Colab, open `colab_run.ipynb` — it mounts Drive, installs deps, logs into W&B, and calls `run.py`.

## Files

| File | Purpose |
|------|---------|
| `run.py` | Entry point — best config baked in, calls train/eval |
| `train.py` | Training loop with W&B logging and early stopping |
| `eval.py` | Test set evaluation and qualitative visualisation |
| `arch/unet.py` | Model architecture |
| `colab_run.ipynb` | Colab entry point |
| `experiments.md` | Full experiment log |
| `checkpoints/` | Saved model weights (gitignored) |

## Architecture

```
Input (3, 512, 512)
  ↓ DoubleConv(3→64)                          [skip x1]
  ↓ DownLayer(64→128)                         [skip x2]
  ↓ DownLayer(128→256)                        [skip x3]
  ↓ DownLayer(256→512)                        [skip x4]
  ↓ Dropout + DownLayer(512→1024)             [bottleneck]
  ↓ Dropout + UpLayer(1024→512)  → aux head
  ↓ Dropout + UpLayer(512→256)   → aux head
  ↓ UpLayer(256→128)             → aux head
  ↓ UpLayer(128→64)
  ↓ Conv2d(64→3)                              [output]
```

- **Norm:** InstanceNorm2d throughout
- **Deep supervision:** auxiliary loss heads at up1/up2/up3 (weights 0.5 / 0.25 / 0.125)
- **Dropout2d** (p=0.3) at bottleneck and first two decoder stages
- ~31M parameters (64 base)

## Best Config

| Hyperparameter | Value |
|----------------|-------|
| Loss | Focal (γ=2, unweighted) + 2×per-image Dice |
| Optimiser | AdamW (lr=1e-4, wd=1e-2) |
| Scheduler | ReduceLROnPlateau (mode=max, factor=0.5, patience=10) |
| Sampling | WeightedRandomSampler (minority pixel fraction, floor=0.1) |
| Augmentation | rot90 + hflip + vflip + HED stain jitter + brightness ±10% |
| Epochs | 100 (early stop patience=25) |
| Batch size | 4 |
