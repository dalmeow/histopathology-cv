"""
Tissue segmentation ablation runner.

Each experiment is a self-contained config dict. Submit one SLURM job per
experiment so they run in parallel:

    sbatch slurm_run.sh --exp 1
    sbatch slurm_run.sh --exp 2
    ...

Or run a single experiment locally:

    python run.py --exp 3
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "1_shared"))

from train import train
from eval  import evaluate

# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------
# Every field is written out explicitly in each config — no inheritance —
# so each experiment is independently readable.

EXPERIMENTS = {
    "1": dict(
        name               = "exp1_baseline",
        # Architecture
        norm_type          = "batch",
        use_residual       = False,
        use_deep_sup       = False,
        dropout_p          = 0.0,
        # Augmentation
        augment_hed        = False,
        # Loss
        loss_type          = "ce",
        use_class_weights  = False,
        loss_lambda        = 0.0,
        # Data
        other_threshold    = 0.0,
        other_oversample_k = 0,
        # Training
        img_size           = 512,
        batch_size         = 8,
        epochs             = 100,
        early_stop_patience= 25,
        seed               = 42,
    ),
    "2": dict(
        name               = "exp2_domain",
        # Architecture
        norm_type          = "instance",
        use_residual       = False,
        use_deep_sup       = False,
        dropout_p          = 0.0,
        # Augmentation: + InstanceNorm + HED stain aug + brightness jitter
        augment_hed        = True,
        # Loss
        loss_type          = "ce",
        use_class_weights  = False,
        loss_lambda        = 0.0,
        # Data
        other_threshold    = 0.0,
        other_oversample_k = 0,
        # Training
        img_size           = 512,
        batch_size         = 8,
        epochs             = 100,
        early_stop_patience= 25,
        seed               = 42,
    ),
    "3a": dict(
        name               = "exp3a_loss_balance",
        # Architecture
        norm_type          = "instance",
        use_residual       = False,
        use_deep_sup       = False,
        dropout_p          = 0.0,
        # Augmentation
        augment_hed        = True,
        # Loss: + sqrt-weighted CE + per-image Dice (lambda=2)
        loss_type          = "ce+dice",
        use_class_weights  = True,
        loss_lambda        = 2.0,
        # Data: no oversampling — loss handles balance
        other_threshold    = 0.0,
        other_oversample_k = 0,
        # Training
        img_size           = 512,
        batch_size         = 8,
        epochs             = 100,
        early_stop_patience= 25,
        seed               = 42,
    ),
    "3b": dict(
        name               = "exp3b_oversample",
        # Architecture
        norm_type          = "instance",
        use_residual       = False,
        use_deep_sup       = False,
        dropout_p          = 0.0,
        # Augmentation
        augment_hed        = True,
        # Loss: plain CE — data handles balance
        loss_type          = "ce",
        use_class_weights  = False,
        loss_lambda        = 0.0,
        # Data: + physical oversampling of Other-rich images (>=10% Other pixels, ×3 total)
        other_threshold    = 0.10,
        other_oversample_k = 2,
        # Training
        img_size           = 512,
        batch_size         = 8,
        epochs             = 100,
        early_stop_patience= 25,
        seed               = 42,
    ),
    "4": dict(
        name               = "exp4_architecture",
        # Architecture: + deep supervision + residual decoder + dropout
        norm_type          = "instance",
        use_residual       = True,
        use_deep_sup       = True,
        dropout_p          = 0.3,
        # Augmentation
        augment_hed        = True,
        # Loss: weighted CE + Dice (from 3a)
        loss_type          = "ce+dice",
        use_class_weights  = True,
        loss_lambda        = 2.0,
        # Data: no oversampling
        other_threshold    = 0.0,
        other_oversample_k = 0,
        # Training
        img_size           = 512,
        batch_size         = 8,
        epochs             = 100,
        early_stop_patience= 25,
        seed               = 42,
    ),
    "5": dict(
        name               = "exp5_lovasz",
        # Architecture
        norm_type          = "instance",
        use_residual       = True,
        use_deep_sup       = True,
        dropout_p          = 0.3,
        # Augmentation
        augment_hed        = True,
        # Loss: Focal (gamma=2) + Lovász-Softmax (batch mode, lambda=2); replaces weighted CE+Dice
        loss_type          = "focal+lovasz",
        use_class_weights  = False,
        loss_lambda        = 2.0,
        # Data: no oversampling
        other_threshold    = 0.0,
        other_oversample_k = 0,
        # Training
        img_size           = 512,
        batch_size         = 8,
        epochs             = 100,
        early_stop_patience= 25,
        seed               = 42,
    ),
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run a single ablation experiment (train + eval).")
    parser.add_argument(
        "--exp", type=str, required=True, choices=list(EXPERIMENTS.keys()),
        help="Experiment key to run (1, 2, 3a, 3b, 4, 5).",
    )
    args = parser.parse_args()

    config = EXPERIMENTS[args.exp]
    print(f"Experiment {args.exp}: {config['name']}")

    train(config)
    evaluate(config)


if __name__ == "__main__":
    main()
