"""
Task 2a — End-to-end nuclei classification runner.

Five additive experiments:
  Exp 1: SimpleClassifier (naive baseline)
  Exp 2: NucleiResNet base
  Exp 3: + ECA attention
  Exp 4: + Mixup
  Exp 5: ResNet-18 (ImageNet pretrained)

Run one experiment:
    python run.py --exp 1
    python run.py --exp 3

Training is skipped if the best checkpoint already exists.
Run extract_patches.py in 2_data/ before training.
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from train import train
from eval  import evaluate

# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------
# Each experiment is fully self-contained — no inheritance.
# Additive ablation: each exp adds exactly one change over the previous.
#
#   Exp 1 → Exp 2: model simple → nuclresnet  (architecture upgrade)
#   Exp 2 → Exp 3: use_eca False → True        (channel attention)
#   Exp 3 → Exp 4: mixup_alpha 0 → 0.3         (training regularisation)
#   Exp 4 → Exp 5: model nuclresnet → resnet18  (ImageNet pretraining)

EXPERIMENTS = {
    "1": dict(
        name          = "exp1_simple",
        # Model
        model         = "simple",
        use_eca       = False,          # N/A for SimpleClassifier
        # Training
        augment_level = "baseline",
        mixup_alpha   = 0.0,
        # Optimiser
        lr            = 1e-3,
        weight_decay  = 1e-4,
        # Scheduler: ReduceLROnPlateau (stepped per epoch on val accuracy)
        scheduler     = "plateau",
        lr_patience   = 8,
        lr_factor     = 0.5,
        # Loop
        batch_size    = 64,
        num_epochs    = 100,
        early_stop    = 15,
        seed          = 42,
    ),
    "2": dict(
        name          = "exp2_nuclresnet_base",
        # Model
        model         = "nuclresnet",
        use_eca       = False,
        # Training
        augment_level = "improved",
        mixup_alpha   = 0.0,
        # Optimiser
        lr            = 1e-3,
        weight_decay  = 5e-4,
        # Scheduler: CosineAnnealingWarmRestarts (stepped per batch)
        scheduler     = "cosine",
        cosine_t0     = 20,
        cosine_t_mult = 2,
        cosine_eta_min= 1e-6,
        # Loop
        batch_size    = 64,
        num_epochs    = 100,
        early_stop    = 15,
        seed          = 42,
    ),
    "3": dict(
        name          = "exp3_eca",
        # Model: + ECA attention in residual blocks
        model         = "nuclresnet",
        use_eca       = True,
        # Training: identical to exp2
        augment_level = "improved",
        mixup_alpha   = 0.0,
        # Optimiser: identical to exp2
        lr            = 1e-3,
        weight_decay  = 5e-4,
        # Scheduler: identical to exp2
        scheduler     = "cosine",
        cosine_t0     = 20,
        cosine_t_mult = 2,
        cosine_eta_min= 1e-6,
        # Loop: identical to exp2
        batch_size    = 64,
        num_epochs    = 100,
        early_stop    = 15,
        seed          = 42,
    ),
    "4": dict(
        name          = "exp4_mixup",
        # Model: identical to exp3
        model         = "nuclresnet",
        use_eca       = True,
        # Training: + Mixup (Beta(0.3, 0.3) label mixing)
        augment_level = "improved",
        mixup_alpha   = 0.3,
        # Optimiser: identical to exp3
        lr            = 1e-3,
        weight_decay  = 5e-4,
        # Scheduler: identical to exp3
        scheduler     = "cosine",
        cosine_t0     = 20,
        cosine_t_mult = 2,
        cosine_eta_min= 1e-6,
        # Loop: identical to exp3
        batch_size    = 64,
        num_epochs    = 100,
        early_stop    = 15,
        seed          = 42,
    ),
    "5": dict(
        name          = "exp5_resnet18",
        # Model: ImageNet-pretrained ResNet-18 backbone
        model         = "resnet18",
        use_eca       = False,          # N/A for ResNet-18
        # Training: identical to exp4
        augment_level = "improved",
        mixup_alpha   = 0.3,
        # Optimiser: identical to exp4
        lr            = 1e-3,
        weight_decay  = 5e-4,
        # Scheduler: identical to exp4
        scheduler     = "cosine",
        cosine_t0     = 20,
        cosine_t_mult = 2,
        cosine_eta_min= 1e-6,
        # Loop: identical to exp4
        batch_size    = 64,
        num_epochs    = 100,
        early_stop    = 15,
        seed          = 42,
    ),
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single Task 2a experiment (train + eval)."
    )
    parser.add_argument(
        "--exp", type=str, required=True, choices=list(EXPERIMENTS.keys()),
        help="Experiment number to run (1–5).",
    )
    args = parser.parse_args()

    config = EXPERIMENTS[args.exp]
    print(f"\nExperiment {args.exp}: {config['name']}")

    # Skip training if best checkpoint already exists
    # (CKPT_DIR is defined in train.py / eval.py)
    from train import CKPT_DIR
    best_ckpt = CKPT_DIR / config["name"] / "best.pth"
    if best_ckpt.exists():
        print(f"Checkpoint found at {best_ckpt} — skipping training.")
    else:
        train(config)

    evaluate(config)


if __name__ == "__main__":
    main()
