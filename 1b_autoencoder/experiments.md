# Autoencoder — Experiment Log

## Results Summary

| Phase | Run | Tumor | Stroma | Other | Avg Dice | Notes |
|-------|-----|-------|--------|-------|----------|-------|
| finetune | ae_v1_rand (512, no residual) | 0.738 | 0.302 | 0.080 | 0.373 | vanilla MSE pretrain, random decoder — preliminary |
| finetune | ae_v1_rand (1024, no residual) | 0.838 | 0.282 | 0.043 | 0.388 | vanilla MSE pretrain, random decoder |
| finetune | ae_v2_masked (1024, no residual) | 0.802 | 0.323 | 0.090 | **0.405** | masked MAE pretrain (75%, 80px patches), random decoder |

---

## Baseline (UNet reference)
- UNet v1 avg Dice: **0.416** (512, no residual)
- UNet v2 avg Dice: **0.461** (512, residual decoder)

---

## Experiment Details

### ae_v1_rand — 1024×1024 (`ae_v1_rand`)
- **Encoder init:** pretrained autoencoder (`ae_v1_pretrain`, vanilla MSE)
- **Decoder init:** random
- **Pretrain:** vanilla MSE reconstruction, 1024×1024, 100 epochs — early stopped at epoch 90 (best epoch 65)
- **Finetune:** frozen encoder, SegDecoder trained from scratch, focal unweighted (γ=2) + 2×Dice, weighted sampler, AdamW (lr=1e-4, wd=1e-2), 100 epochs — ran full 100 epochs (best epoch 99)
- *Avg Dice 0.388 — below UNet v1 baseline (0.416); better than 512px preliminary (0.373). Stroma (0.282) and Other (0.043) still weak — encoder pretrained on reconstruction hasn't learned segmentation-useful features*

### ae_v2_masked — masked MAE pretrain (`ae_v2_masked`)
- **Encoder init:** pretrained autoencoder (`ae_v2_masked_pretrain`, masked MAE-style)
- **Decoder init:** random
- **Pretrain:** masked reconstruction, 1024×1024, 75% of 80px patches zeroed in input, loss = MSE + 0.5×(1−SSIM) on masked regions only, 100 epochs — ran full 100 epochs (best epoch 97)
- **Finetune:** same as ae_v1_rand — focal unweighted (γ=2) + 2×Dice, weighted sampler, AdamW (lr=1e-4, wd=1e-2), 100 epochs — ran full 100 epochs (best epoch 100)
- *Avg Dice 0.405 — best AE result yet, above UNet v1 baseline (0.416 borderline). Other class nearly doubled vs ae_v1_rand (0.090 vs 0.043); Stroma also up (0.323 vs 0.282). Masked pretraining forces encoder to learn structural features more useful for segmentation*

### ae_v1_rand (preliminary — 512×512, no residual)
- **Encoder init:** random
- **Decoder init:** random
- **Pretrain:** vanilla MSE reconstruction, 512×512
- **Finetune:** frozen encoder, SegDecoder trained from scratch, focal + 2×Dice, weighted sampler, AdamW
- **Image size:** 512 (old config — preliminary run only)
- Early stopped at epoch 29 (patience=10)
- *Avg Dice 0.373 — below UNet v1 baseline (0.416). Expected: this is a sanity check run with suboptimal config (512, no residual). Full experiments will use 1024×1024 with v3 UNet settings.*
