#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --output=1a_unet/logs/%j_%x.out
#SBATCH --error=1a_unet/logs/%j_%x.err

# Do not call this script directly.
# Use submit.sh which sets --job-name per experiment:
#   bash 1a_unet/submit.sh 1 2 3a 3b 4 5

set -e

cd ~/cv/histopathology-cv

source ~/cv/venv_combined/bin/activate

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "GPUs   : $CUDA_VISIBLE_DEVICES"
echo "Args   : $@"

nvidia-smi

mkdir -p 1a_unet/logs

python 1a_unet/run.py "$@" 2>&1 | tee "1a_unet/logs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.log"
