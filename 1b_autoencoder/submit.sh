#!/bin/bash
# Submit AE experiments as parallel SLURM jobs.
#
# Usage: bash 1b_autoencoder/submit.sh 1 2 3 4
#
# All four have independent pretrain checkpoints and can be submitted simultaneously.

set -e

cd ~/cv/histopathology-cv

for exp in "$@"; do
    job_id=$(sbatch --job-name="ae_exp${exp}" \
                    1b_autoencoder/slurm_run.sh --exp "${exp}" \
             | awk '{print $NF}')
    echo "Submitted exp${exp} → job ${job_id}"
done
