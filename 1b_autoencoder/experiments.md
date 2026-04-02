# Autoencoder — Experiment Log

## Results Summary

| Phase | Run | Tumor | Stroma | Other | Avg Dice | Notes |
|-------|-----|-------|--------|-------|----------|-------|
| finetune | ae_v1_rand (512, no residual) | 0.738 | 0.302 | 0.080 | 0.373 | vanilla MSE pretrain, random decoder |

---

## Baseline (UNet reference)
- UNet v1 avg Dice: **0.416** (512, no residual)
- UNet v2 avg Dice: **0.461** (512, residual decoder)

---

## Experiment Details

### ae_v1_rand (preliminary — 512×512, no residual)
- **Encoder init:** random
- **Decoder init:** random
- **Pretrain:** vanilla MSE reconstruction, 512×512
- **Finetune:** frozen encoder, SegDecoder trained from scratch, focal + 2×Dice, weighted sampler, AdamW
- **Image size:** 512 (old config — preliminary run only)
- Early stopped at epoch 29 (patience=10)
- *Avg Dice 0.373 — below UNet v1 baseline (0.416). Expected: this is a sanity check run with suboptimal config (512, no residual). Full experiments will use 1024×1024 with v3 UNet settings.*
