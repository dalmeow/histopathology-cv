"""Autoencoder experiment runner.

Usage:
    python run.py pretrain  --config ae_v1_rand
    python run.py finetune  --config ae_v1_rand
    python run.py eval      --config ae_v1_rand
    python run.py all       --config ae_v1_rand
"""
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "1_shared"))

# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------
CONFIGS = {
    "ae_v1_rand": dict(
        pretrain_run    = "ae_v1_pretrain",   # shared with ae_v1_unet_dec
        finetune_run    = "ae_v1_rand",
        masked          = False,
        unet_dec_init   = False,
        unet_dec_run    = "unet_v3_1024",
        use_residual    = False,
        img_size        = 1024,
        batch_size      = 2,
    ),
    "ae_v1_unet_dec": dict(
        pretrain_run    = "ae_v1_pretrain",   # shared with ae_v1_rand
        finetune_run    = "ae_v1_unet_dec",
        masked          = False,
        unet_dec_init   = True,
        unet_dec_run    = "unet_v3_1024",
        use_residual    = False,
        img_size        = 1024,
        batch_size      = 2,
    ),
    "ae_v2_masked": dict(
        pretrain_run    = "ae_v2_masked_pretrain",
        finetune_run    = "ae_v2_masked",
        masked          = True,
        unet_dec_init   = False,
        unet_dec_run    = "unet_v3_1024",
        use_residual    = False,
        img_size        = 1024,
        batch_size      = 2,
    ),
}

# Shared hyperparameters
PRETRAIN_EPOCHS     = 100
FINETUNE_EPOCHS     = 100
LR                  = 1e-4
WEIGHT_DECAY        = 1e-2
EARLY_STOP_PATIENCE = 25


# ---------------------------------------------------------------------------
# GPU cleanup
# ---------------------------------------------------------------------------

def clear_gpu_memory():
    import gc
    import os
    import signal
    import subprocess
    import time

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
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

    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_pretrain(cfg):
    clear_gpu_memory()
    import argparse as ap
    import train_ae as train_ae_module
    args = ap.Namespace(
        run_name            = cfg["pretrain_run"],
        masked              = cfg["masked"],
        use_residual        = cfg["use_residual"],
        img_size            = cfg["img_size"],
        epochs              = PRETRAIN_EPOCHS,
        batch_size          = cfg["batch_size"],
        lr                  = LR,
        weight_decay        = WEIGHT_DECAY,
        early_stop_patience = EARLY_STOP_PATIENCE,
    )
    train_ae_module.train(args)


def run_finetune(cfg):
    clear_gpu_memory()
    import argparse as ap
    import finetune as finetune_module
    args = ap.Namespace(
        run_name            = cfg["finetune_run"],
        pretrain_run        = cfg["pretrain_run"],
        unet_dec_init       = cfg["unet_dec_init"],
        unet_dec_run        = cfg["unet_dec_run"],
        use_residual        = cfg["use_residual"],
        img_size            = cfg["img_size"],
        epochs              = FINETUNE_EPOCHS,
        batch_size          = cfg["batch_size"],
        lr                  = LR,
        weight_decay        = WEIGHT_DECAY,
        early_stop_patience = EARLY_STOP_PATIENCE,
    )
    finetune_module.finetune(args)


def run_eval(cfg):
    import argparse as ap
    import eval as eval_module
    args = ap.Namespace(
        run_name     = cfg["finetune_run"],
        use_residual = cfg["use_residual"],
        img_size     = cfg["img_size"],
    )
    eval_module.main(args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder experiment runner")
    parser.add_argument("mode", choices=["pretrain", "finetune", "eval", "all"])
    parser.add_argument("--config", choices=list(CONFIGS.keys()), required=True)
    args = parser.parse_args()
    cfg  = CONFIGS[args.config]

    if args.mode in ("pretrain", "all"):
        run_pretrain(cfg)
    if args.mode in ("finetune", "all"):
        run_finetune(cfg)
    if args.mode in ("eval", "all"):
        run_eval(cfg)
