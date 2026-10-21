#!/bin/bash
# One contaminated catalogue per array task (task id = seed): the all-objects and
# pure-Ia reference fits plus M Bayesian-resampling fits, then pooling.
#   COSMO=cosmo5_cmb_bao NREP=5 sbatch --array=0-24 slurm/contam.sbatch
#   COSMO=cosmo1_dust NREP=5 EXTRA="--dust_step --resamples_only" sbatch --array=0-24 slurm/contam.sbatch
#   COSMO=cosmo1 NREP=5 EXTRA="--n_scale 50 --resamples_only --chunk_size 20000" REPS=2-2 \
#       sbatch --array=0 --time=10:00:00 slurm/contam.sbatch
#SBATCH --job-name=flowsn2_contam
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH -c 8
#SBATCH -q regular
#SBATCH --output=slurm_logs/%x_%A_%a.out

source slurm/env.sh
python scripts/infer_contam.py --cosmo "${COSMO:?}" --seed "$SLURM_ARRAY_TASK_ID" \
    --n_resamples "${NREP:-5}" --reps "${REPS:-all}" ${EXTRA:-}
