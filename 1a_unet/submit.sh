#!/bin/bash
# Submit experiments as parallel slurm jobs.
#
# Usage:
#   bash 1a_unet/submit.sh 1 2 3a 3b 4 5   # submit all
#   bash 1a_unet/submit.sh 4 5              # submit specific experiments

set -e

if [ $# -eq 0 ]; then
    echo "Usage: bash 1a_unet/submit.sh <exp1> [exp2] ..."
    echo "  e.g. bash 1a_unet/submit.sh 1 2 3a 3b 4 5"
    exit 1
fi

cd "$(dirname "$0")/.."   # repo root

mkdir -p 1a_unet/logs

for exp in "$@"; do
    job_name="unet_exp${exp}"
    job_id=$(sbatch --job-name="$job_name" 1a_unet/slurm_run.sh --exp "$exp" | awk '{print $4}')
    echo "Submitted exp${exp} → job ${job_id} (${job_name})"
done
