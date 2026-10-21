#!/bin/bash
# Submit every MCMC run behind the paper's tables and figures (Perlmutter / SLURM).
# Run from the repository root:
#
#   export SBATCH_ACCOUNT=<your_account>      # e.g. m1727_g
#   bash slurm/submit_paper.sh                # everything
#   bash slurm/submit_paper.sh pure_ia        # one block: pure_ia | contam | model_comp | scaling
#   DRYRUN=1 bash slurm/submit_paper.sh       # print the sbatch commands only
#
# All fits are skip-if-exists, so re-running only fills gaps. Seeds are 0-24
# throughout (25 realisations); the figures use C2 seed 1, C5+CMB+BAO seed 1
# and C1 seed 0.
set -e
cd "$(dirname "$0")/.."
mkdir -p slurm_logs
WHAT="${1:-all}"
SEEDS="0-24"

sub () {   # sub VAR=... -- sbatch args
    if [ -n "$DRYRUN" ]; then echo "env $*"; else env "$@"; fi
}
want () { [ "$WHAT" = all ] || [ "$WHAT" = "$1" ]; }

# --- Section 3.1: pure SN Ia, FlowSN2 vs analytical (Tables E1, E2; Figure 5) ---
if want pure_ia; then
    for COSMO in cosmo1 cosmo1_cmb cosmo2 cosmo2_cmb cosmo3 cosmo3_cmb cosmo4_cmb_bao cosmo5_cmb_bao; do
        for ENGINE in flow analytical; do
            sub COSMO=$COSMO ENGINE=$ENGINE sbatch --array=$SEEDS slurm/pure_ia.sbatch
        done
    done
    # naive (selection ignored) posterior, only drawn in Figure 5
    sub COSMO=cosmo2 ENGINE=naive sbatch --array=1 slurm/pure_ia.sbatch
fi

# --- Section 3.2: contamination (Tables 1, 2; Figures 6, 7, E1) ---
if want contam; then
    # The published C1 SN-only column pooled M=10 resamples per seed; every other
    # cosmology used M=5. Resamples 0-4 are identical either way.
    sub COSMO=cosmo1 NREP=10 sbatch --array=$SEEDS --time=06:00:00 slurm/contam.sbatch
    for COSMO in cosmo1_cmb cosmo2 cosmo2_cmb cosmo3 cosmo3_cmb cosmo4_cmb_bao cosmo5_cmb_bao; do
        sub COSMO=$COSMO NREP=5 sbatch --array=$SEEDS slurm/contam.sbatch
    done
fi

# --- Section 3.4: model comparison (Table 3). The wa test reuses the
#     cosmo4_cmb_bao / cosmo5_cmb_bao runs above. ---
if want model_comp; then
    for COSMO in cosmo1_dustnull cosmo1_dust; do
        sub COSMO=$COSMO NREP=5 EXTRA="--dust_step --resamples_only" \
            sbatch --array=$SEEDS slurm/contam.sbatch
    done
fi

# --- Section 3.3: scaling (Figure 8). The 1x point is cosmo1 seed 0 from 'contam'. ---
if want scaling; then
    sub COSMO=cosmo1 NREP=5 EXTRA="--n_scale 10 --resamples_only --chunk_size 20000" \
        sbatch --array=0 --time=06:00:00 slurm/contam.sbatch
    # 50x: one job per resample (~5-10 h each); each pools whatever is on disk when it ends
    for M in 0 1 2 3 4; do
        sub COSMO=cosmo1 NREP=5 REPS=$M-$M EXTRA="--n_scale 50 --resamples_only --chunk_size 20000" \
            sbatch --array=0 --time=12:00:00 slurm/contam.sbatch
    done
fi
echo "Submitted [$WHAT]. When finished: bash paper/make_all.sh"
