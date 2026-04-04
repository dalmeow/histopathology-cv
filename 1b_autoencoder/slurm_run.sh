#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=6:00:00
#SBATCH --output=1b_autoencoder/logs/%j_%x.out
#SBATCH --error=1b_autoencoder/logs/%j_%x.err

# Usage: sbatch --job-name=ae_expN 1b_autoencoder/slurm_run.sh --exp N

set -e

cd ~/cv/histopathology-cv

source ~/cv/venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Args:   $@"

nvidia-smi

python 1b_autoencoder/run.py "$@"
