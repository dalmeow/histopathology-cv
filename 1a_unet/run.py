"""UNet runner — bakes in the best known config (0.416 avg Dice).

Usage:
    python run.py train
    python run.py eval
    python run.py both
"""
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

# ---------------------------------------------------------------------------
# Best config
# ---------------------------------------------------------------------------
RUN_NAME            = "unet_v1_perdice"
EPOCHS              = 100
BATCH_SIZE          = 4
LR                  = 1e-4
WEIGHT_DECAY        = 1e-2
LOSS                = "focal"
LOSS_RATIO          = 2.0
FOCAL_GAMMA         = 2.0
NO_CLASS_WEIGHTS    = True
EARLY_STOP_PATIENCE = 25
SAVE_INTERVAL       = 10


def run_train():
    import argparse as ap
    import train as train_module
    args = ap.Namespace(
        run_name            = RUN_NAME,
        epochs              = EPOCHS,
        batch_size          = BATCH_SIZE,
        lr                  = LR,
        weight_decay        = WEIGHT_DECAY,
        momentum            = 0.9,
        save_interval       = SAVE_INTERVAL,
        early_stop_patience = EARLY_STOP_PATIENCE,
        resume              = False,
        loss                = LOSS,
        loss_ratio          = LOSS_RATIO,
        focal_gamma         = FOCAL_GAMMA,
        no_class_weights    = NO_CLASS_WEIGHTS,
    )
    train_module.train(args)


def run_eval():
    import argparse as ap
    import eval as eval_module
    args = ap.Namespace(run_name=RUN_NAME)
    eval_module.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNet tissue segmentation runner")
    parser.add_argument("mode", choices=["train", "eval", "both"],
                        help="train: run training  |  eval: run test evaluation  |  both: train then eval")
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        run_train()
    if args.mode in ("eval", "both"):
        run_eval()
