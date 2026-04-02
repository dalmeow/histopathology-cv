#!/bin/bash
#SBATCH --job-name=unet_v3
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=3:00:00
#SBATCH --output=1a_unet/logs/%j_%x.out
#SBATCH --error=1a_unet/logs/%j_%x.err

# Usage: sbatch 1a_unet/slurm_train.sh --config v3_1024

set -e

cd ~/cv/histopathology-cv/1a_unet

source ~/cv/venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Args:   $@"

nvidia-smi

LOG_NAME=$(echo "$@" | tr ' ' '_' | sed 's/--//g')

python run.py both "$@" 2>&1 | tee "logs/${LOG_NAME}.log"
