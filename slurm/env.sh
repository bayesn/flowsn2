# Sourced by every job: Perlmutter modules, the conda env and runtime settings.
# Edit CONDA_ENV / the module lines for another cluster.
module load cudatoolkit cudnn
module load python
conda activate "${CONDA_ENV:-flowsn2}"

export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Let JAX use ~38 GB of a 40 GB A100 (default ~75%): needed for the 10x / 50x scaling fits.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
