#!/bin/bash
# One pure SN Ia fit per array task (task id = seed).
#   COSMO=cosmo2 ENGINE=flow sbatch --array=0-24 slurm/pure_ia.sbatch
#SBATCH --job-name=flowsn2_pure_ia
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH -c 8
#SBATCH -q regular
#SBATCH --output=slurm_logs/%x_%A_%a.out

source slurm/env.sh
python scripts/infer_pure_ia.py --engine "${ENGINE:?}" --cosmo "${COSMO:?}" --seed "$SLURM_ARRAY_TASK_ID"
