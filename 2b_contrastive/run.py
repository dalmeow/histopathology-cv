"""
Task 2b — Contrastive pre-training runner.

Four additive experiments:
  Exp 1: SimCLR NucleiResNet (strong aug) + random-init MLP head + Mixup
  Exp 2: + K-means head initialisation
  Exp 3: moderate SimCLR aug + K-means init, − Mixup
  Exp 4: ResNet-18 (ImageNet) SimCLR + random-init MLP head, − Mixup

  Exp 1 → 2: add K-means head init
  Exp 2 → 3: swap strong → moderate SimCLR aug, drop Mixup
  Exp 3 → 4: swap backbone NucleiResNet → ResNet-18

Pre-training is shared across experiments with the same pretrain_name:
  nuclresnet_strong    — exp 1 + 2  (one pretrain run, two finetune runs)
  nuclresnet_moderate  — exp 3
  resnet18             — exp 4

Run one experiment:
    python run.py --exp 1
    python run.py --exp 3

Pre-training and fine-tuning are each skipped if the checkpoint already exists.
Run extract_patches.py in 2_data/ before training.
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from pretrain import pretrain, PRETRAIN_DIR
from finetune import finetune, kmeans_init, CKPT_DIR
from eval     import evaluate

# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------
# Each experiment is fully self-contained — no inheritance.
# Additive ablation: each exp adds exactly one change over the previous.

EXPERIMENTS = {
    "1": dict(
        name           = "exp1_simclr_mlp",
        # ---- Pretrain ----
        pretrain_name  = "nuclresnet_strong",   # shared with exp 2
        backbone       = "nuclresnet",
        color_strength = "strong",
        use_eca        = True,
        # ---- Head ----
        head_init      = "random",
        augment_level  = "mild",
        mixup_alpha    = 0.3,
        # ---- Pretrain hparams ----
        pretrain_batch_size = 256,
        pretrain_epochs     = 500,
        pretrain_warmup     = 10,
        pretrain_lr         = 3e-4,
        pretrain_wd         = 1e-4,
        pretrain_temp       = 0.3,
        pretrain_proj_dim   = 128,
        pretrain_early_stop = 30,
        # ---- Finetune hparams ----
        lr           = 1e-3,
        weight_decay = 5e-4,
        lr_patience  = 8,
        lr_factor    = 0.5,
        batch_size   = 64,
        num_epochs   = 100,
        early_stop   = 15,
        seed         = 42,
    ),
    "2": dict(
        name           = "exp2_kmeans_init",
        # ---- Pretrain: same checkpoint as exp 1 ----
        pretrain_name  = "nuclresnet_strong",
        backbone       = "nuclresnet",
        color_strength = "strong",
        use_eca        = True,
        # ---- Head: + K-means initialisation ----
        head_init      = "kmeans",
        augment_level  = "mild",
        mixup_alpha    = 0.3,
        # ---- Pretrain hparams: identical to exp 1 ----
        pretrain_batch_size = 256,
        pretrain_epochs     = 500,
        pretrain_warmup     = 10,
        pretrain_lr         = 3e-4,
        pretrain_wd         = 1e-4,
        pretrain_temp       = 0.3,
        pretrain_proj_dim   = 128,
        pretrain_early_stop = 30,
        # ---- Finetune hparams: identical to exp 1 ----
        lr           = 1e-3,
        weight_decay = 5e-4,
        lr_patience  = 8,
        lr_factor    = 0.5,
        batch_size   = 64,
        num_epochs   = 100,
        early_stop   = 15,
        seed         = 42,
    ),
    "3": dict(
        name           = "exp3_moderate_aug",
        # ---- Pretrain: moderate SimCLR aug, own checkpoint ----
        pretrain_name  = "nuclresnet_moderate",
        backbone       = "nuclresnet",
        color_strength = "moderate",
        use_eca        = True,
        # ---- Head: K-means init, − Mixup, moderate (SimCLR-style) finetune aug ----
        head_init      = "kmeans",
        augment_level  = "moderate",
        mixup_alpha    = 0.0,
        # ---- Pretrain hparams: same as exp 1/2 (different aug via color_strength) ----
        pretrain_batch_size = 256,
        pretrain_epochs     = 500,
        pretrain_warmup     = 10,
        pretrain_lr         = 3e-4,
        pretrain_wd         = 1e-4,
        pretrain_temp       = 0.3,
        pretrain_proj_dim   = 128,
        pretrain_early_stop = 30,
        # ---- Finetune hparams: same (minus Mixup) ----
        lr           = 1e-3,
        weight_decay = 5e-4,
        lr_patience  = 8,
        lr_factor    = 0.5,
        batch_size   = 64,
        num_epochs   = 100,
        early_stop   = 15,
        seed         = 42,
    ),
    "4": dict(
        name           = "exp4_resnet18",
        # ---- Pretrain: ResNet-18 (ImageNet init) with _R18Aug ----
        pretrain_name  = "resnet18",
        backbone       = "resnet18",
        color_strength = "strong",   # selects _R18Aug (backbone check takes precedence)
        use_eca        = False,      # N/A for ResNet-18
        # ---- Head: random init, − Mixup ----
        head_init      = "random",
        augment_level  = "mild",
        mixup_alpha    = 0.0,
        # ---- Pretrain hparams: same schedule as NucleiResNet exps ----
        pretrain_batch_size = 256,
        pretrain_epochs     = 150,
        pretrain_warmup     = 10,
        pretrain_lr         = 1e-3,
        pretrain_wd         = 1e-4,
        pretrain_temp       = 0.3,
        pretrain_proj_dim   = 128,
        pretrain_early_stop = 30,
        # ---- Finetune hparams: same (minus Mixup) ----
        lr           = 1e-3,
        weight_decay = 5e-4,
        lr_patience  = 8,
        lr_factor    = 0.5,
        batch_size   = 64,
        num_epochs   = 100,
        early_stop   = 15,
        seed         = 42,
    ),
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single Task 2b experiment (pretrain + finetune + eval)."
    )
    parser.add_argument(
        "--exp", type=str, required=True, choices=list(EXPERIMENTS.keys()),
        help="Experiment number to run (1–4).",
    )
    args = parser.parse_args()

    config = EXPERIMENTS[args.exp]
    print(f"\nExperiment {args.exp}: {config['name']}")
    print(f"  Pretrain checkpoint : {config['pretrain_name']}")
    print(f"  Head init           : {config['head_init']}")
    print(f"  Mixup alpha         : {config['mixup_alpha']}")

    pretrain_ckpt = PRETRAIN_DIR / config["pretrain_name"] / "best.pth"
    finetune_ckpt = CKPT_DIR     / config["name"]          / "best.pth"

    # Stage 1: SimCLR pre-training (shared across experiments with same pretrain_name)
    if pretrain_ckpt.exists():
        print(f"\nPretrain checkpoint found ({pretrain_ckpt}) — skipping pre-training.")
    else:
        pretrain(config)

    # Stage 2: K-means head initialisation (exp 2 + 3 only)
    if config["head_init"] == "kmeans":
        kmeans_ckpt = PRETRAIN_DIR / config["pretrain_name"] / "kmeans_head.pth"
        if kmeans_ckpt.exists():
            print(f"\nK-means checkpoint found ({kmeans_ckpt}) — skipping k-means init.")
        else:
            kmeans_init(config)

    # Stage 3: Supervised head fine-tuning
    if finetune_ckpt.exists():
        print(f"\nFinetune checkpoint found ({finetune_ckpt}) — skipping fine-tuning.")
    else:
        finetune(config)

    # Stage 4: Evaluation (always runs)
    evaluate(config)


if __name__ == "__main__":
    main()
