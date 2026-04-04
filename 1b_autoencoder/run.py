"""
Autoencoder ablation runner.

Each experiment pre-trains an encoder (if not already done), fine-tunes the
SegDecoder, and evaluates on the test set.  Experiments sharing the same
pre-train config reuse the same checkpoint — pre-training is skipped if the
checkpoint directory already exists.

Submit one SLURM job per experiment:

    sbatch slurm_run.sh --exp 1
    sbatch slurm_run.sh --exp 2
    ...

Or run locally:

    python run.py --exp 3
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "1_shared"))

from train_ae import train_pretrain
from finetune import finetune
from eval     import evaluate

# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------
# Every field is written out explicitly — no inheritance — so each experiment
# is independently readable.
#
# Ablation story (2×2):
#   exp1       : AE baseline — vanilla pretrain + vanilla finetune
#   exp1→exp2  : pretrain objective only changes (vanilla MSE → masked MAE),
#                finetune kept identical — isolates encoder strategy
#   exp1→exp4  : finetune quality only changes (all UNet learnings applied),
#                pretrain stays vanilla MSE — isolates finetune improvements
#   exp4→exp3  : pretrain objective changes (MSE → masked MAE),
#                finetune kept identical to exp4 — completes the 2×2

EXPERIMENTS = {
    "1": dict(
        name               = "ae_exp1_baseline",
        # Pre-train: vanilla MSE, BatchNorm
        pretrain_name      = "ae_pretrain_mse_batch",
        pretrain_masked    = False,
        # Architecture
        norm_type          = "batch",
        use_residual       = False,
        use_deep_sup       = False,
        dropout_p          = 0.0,
        # Augmentation
        augment_hed        = False,
        # Loss: plain CE — mirrors UNet exp1
        loss_type          = "ce",
        use_class_weights  = False,
        loss_lambda        = 0.0,
        # Training
        img_size           = 512,
        batch_size         = 8,
        pretrain_epochs    = 100,
        finetune_epochs    = 100,
        early_stop_patience= 25,
        lr                 = 1e-4,
        weight_decay       = 1e-2,
        seed               = 42,
    ),
    "2": dict(
        name               = "ae_exp2_masked",
        # Pre-train: masked MAE, BatchNorm — same finetune as exp1, only pretrain changes
        pretrain_name      = "ae_pretrain_masked_batch",
        pretrain_masked    = True,
        # Architecture: identical to exp1
        norm_type          = "batch",
        use_residual       = False,
        use_deep_sup       = False,
        dropout_p          = 0.0,
        # Augmentation: identical to exp1
        augment_hed        = False,
        # Loss: identical to exp1
        loss_type          = "ce",
        use_class_weights  = False,
        loss_lambda        = 0.0,
        # Training
        img_size           = 512,
        batch_size         = 8,
        pretrain_epochs    = 100,
        finetune_epochs    = 100,
        early_stop_patience= 25,
        lr                 = 1e-4,
        weight_decay       = 1e-2,
        seed               = 42,
    ),
    "3": dict(
        name               = "ae_exp3_full",
        # Pre-train: masked MAE, InstanceNorm + HED — all UNet learnings applied
        pretrain_name      = "ae_pretrain_masked_instance",
        pretrain_masked    = True,
        # Architecture: + InstanceNorm + residual decoder + deep supervision + dropout
        norm_type          = "instance",
        use_residual       = True,
        use_deep_sup       = True,
        dropout_p          = 0.3,
        # Augmentation: + HED stain aug
        augment_hed        = True,
        # Loss: Focal + Lovász (mirrors UNet exp5)
        loss_type          = "focal+lovasz",
        use_class_weights  = False,
        loss_lambda        = 2.0,
        # Training
        img_size           = 512,
        batch_size         = 8,
        pretrain_epochs    = 100,
        finetune_epochs    = 100,
        early_stop_patience= 25,
        lr                 = 1e-4,
        weight_decay       = 1e-2,
        seed               = 42,
    ),
    "4": dict(
        name               = "ae_exp4_mse_full",
        # Pre-train: vanilla MSE, InstanceNorm + HED — identical to exp3 except no masking
        pretrain_name      = "ae_pretrain_mse_instance",
        pretrain_masked    = False,
        # Architecture: identical to exp3
        norm_type          = "instance",
        use_residual       = True,
        use_deep_sup       = True,
        dropout_p          = 0.3,
        # Augmentation: identical to exp3
        augment_hed        = True,
        # Loss: identical to exp3
        loss_type          = "focal+lovasz",
        use_class_weights  = False,
        loss_lambda        = 2.0,
        # Training
        img_size           = 512,
        batch_size         = 8,
        pretrain_epochs    = 100,
        finetune_epochs    = 100,
        early_stop_patience= 25,
        lr                 = 1e-4,
        weight_decay       = 1e-2,
        seed               = 42,
    ),
}

CKPT_DIR = _HERE / "checkpoints"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a single AE ablation experiment (pre-train + finetune + eval)."
    )
    parser.add_argument(
        "--exp", type=str, required=True, choices=list(EXPERIMENTS.keys()),
        help="Experiment number to run (1–4).",
    )
    args = parser.parse_args()

    config = EXPERIMENTS[args.exp]
    print(f"Experiment {args.exp}: {config['name']}")

    # Pre-train: skip if checkpoint already exists
    pretrain_best = CKPT_DIR / config["pretrain_name"] / "ae_best.pt"
    if pretrain_best.exists():
        print(f"\nPre-train checkpoint found at {pretrain_best} — skipping pre-training.")
    else:
        train_pretrain(config)

    finetune(config)
    evaluate(config)


if __name__ == "__main__":
    main()
