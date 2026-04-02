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
# Experiment configs
# ---------------------------------------------------------------------------
CONFIGS = {
    "v1": dict(
        run_name     = "unet_v1_perdice",
        use_residual = False,
        img_size     = 512,
        batch_size   = 8,
        resume       = False,
    ),
    "v2_resid": dict(
        run_name     = "unet_v2_resid",
        use_residual = True,
        img_size     = 512,
        batch_size   = 4,
        resume       = False,
    ),
    "v3_1024": dict(
        run_name     = "unet_v3_1024",
        use_residual = False,
        img_size     = 1024,
        batch_size   = 2,
        resume       = False,
    ),
    "v4_256": dict(
        run_name     = "unet_v4_256",
        use_residual = False,
        img_size     = 256,
        batch_size   = 32,
        resume       = False,
    ),
}

# Shared hyperparameters across all runs
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


def clear_gpu_memory():
    import gc
    import os
    import signal
    import subprocess

    # Kill all processes currently using a GPU (except this process)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        import time
        own_pid = os.getpid()
        for line in result.stdout.strip().splitlines():
            pid = int(line.strip())
            if pid != own_pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                    print(f"Sent SIGTERM to GPU process {pid}, waiting for clean exit...")
                    for _ in range(10):
                        time.sleep(0.5)
                        try:
                            os.kill(pid, 0)
                        except ProcessLookupError:
                            break
                    else:
                        os.kill(pid, signal.SIGKILL)
                        print(f"Process {pid} did not exit cleanly, sent SIGKILL")
                except ProcessLookupError:
                    pass
                except PermissionError:
                    print(f"Cannot kill GPU process {pid} (owned by another user)")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Free PyTorch cached allocations
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def run_train(cfg):
    clear_gpu_memory()
    import argparse as ap
    import train as train_module
    args = ap.Namespace(
        run_name            = cfg["run_name"],
        use_residual        = cfg["use_residual"],
        img_size            = cfg["img_size"],
        epochs              = EPOCHS,
        batch_size          = cfg["batch_size"],
        lr                  = LR,
        weight_decay        = WEIGHT_DECAY,
        momentum            = 0.9,
        save_interval       = SAVE_INTERVAL,
        early_stop_patience = EARLY_STOP_PATIENCE,
        resume              = cfg["resume"],
        loss                = LOSS,
        loss_ratio          = LOSS_RATIO,
        focal_gamma         = FOCAL_GAMMA,
        no_class_weights    = NO_CLASS_WEIGHTS,
    )
    train_module.train(args)


def run_eval(cfg):
    import argparse as ap
    import eval as eval_module
    args = ap.Namespace(run_name=cfg["run_name"], use_residual=cfg["use_residual"])
    eval_module.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNet tissue segmentation runner")
    parser.add_argument("mode", choices=["train", "eval", "both"],
                        help="train: run training  |  eval: run test evaluation  |  both: train then eval")
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default="v1",
                        help="experiment config to run (default: v1)")

    args = parser.parse_args()
    cfg  = CONFIGS[args.config]

    if args.mode in ("train", "both"):
        run_train(cfg)
    if args.mode in ("eval", "both"):
        run_eval(cfg)
