# *FlowSN2*

Code accompanying:

> B. M. Boyd, M. Autenrieth, K. S. Mandel, M. Grayling, M. Ginolin, L. Kelsey and
> S. Thorp, *FlowSN2: Dusty Supernova Cosmology with Photometric Redshifts and
> Contamination using Neural Simulation-Based Inference* (2026).

If you use this code, please cite the paper above.

*FlowSN2* is a modular simulation-based inference framework for photometric SN Ia
cosmology. A hierarchical Bayesian model keeps the cosmology, the SN Ia rate and
the Dusty Simple-BayeSN population analytical. Two normalising flows per survey
learn what cannot be written down:

- `q1` (4D): the observed summaries and photometric redshift given the latent distance, redshift and population.
- `q2` (2D): the selected latent (μ, z) distribution.

Non-Ia contamination is handled with modular Bayesian resampling of the class
labels. This repository reproduces every table and data figure in the paper,
using the same seeds.

## Repository layout

```
flowsn2/                  the library
  simulator.py            forward model (Sec. 2.1, 3.1.1)
  redshifts.py            rates and redshift sampling
  cosmology.py, bao.py    wCDM / w0waCDM distances, CMB and BAO observables (App. D)
  priors.py               global priors (Table C1) + CMB/BAO likelihood terms
  analytical_likelihood.py, analytical_model.py   exact likelihood (App. A)
  flow_likelihood.py, flow_model.py               FlowSN2 likelihood (Sec. 2.2)
  experiments.py          surveys, cosmologies C1-C5, truths, seeds, CMB/BAO mocks
  contamination.py        contaminated catalogues + mock classifier (Sec. 3.2.1)
scripts/                  pipeline
  make_training_data.py   training simulations (25M selected SNe per survey)
  train_q1.py, train_q2.py
  infer_pure_ia.py        pure SN Ia fits (flow / analytical / naive)
  infer_contam.py         contaminated fits (all objects / pure SN Ia / Bayesian resampling)
paper/                    one script per table and figure, and make_all.sh
slurm/                    Perlmutter launchers (submit_paper.sh submits everything)
flow_weights/             the four trained flows used in the paper
wcosmo/                   vendored wcosmo 0.5.0 (C. Talbot, MIT) with small JAX patches; the wCDM backend
```

## Installation

The paper's runs used the exact environment in `environment.yml`: Linux,
NVIDIA A100, CUDA 12, Python 3.10, JAX 0.4.33, numpyro 0.16.1, flowjax 17.0.2.
Both environment files install this repository in editable mode.

```bash
git clone https://github.com/bayesn/flowsn2.git
cd flowsn2

# GPU (as used for the paper; on Perlmutter: module load python cudatoolkit cudnn)
conda env create -f environment.yml
conda activate flowsn2

# or CPU only (tables, figures, small tests)
conda env create -f environment-cpu.yml
conda activate flowsn2-cpu
```

Run every command from the repository root. Outputs go to `results/`, `figs/`,
`tables/` and `training_data/` inside the repository. Set `FLOWSN2_RESULTS` and
`FLOWSN2_TRAINING_DATA` to put the large ones elsewhere, e.g. on `$PSCRATCH`.

## Quick start

The survey and flow figures need only the shipped weights, and run on a laptop in under a minute:

```bash
python paper/fig_flows.py           # figs/survey_sims.pdf (Fig. 1), figs/q1_true_cosmo1.pdf (Fig. 4)
```

A single posterior (≈10 min on an A100):

```bash
python scripts/infer_pure_ia.py --engine flow --cosmo cosmo2 --seed 1
python scripts/infer_contam.py  --cosmo cosmo5_cmb_bao --seed 1      # 2 reference fits + M=5 resamples
```

## Reproducing the paper

### 1. Flows (optional)

`flow_weights/` holds the trained flows used for every result in the paper. To retrain them from scratch:

```bash
python scripts/make_training_data.py          # ~5 GB per survey, PRNGKey(0)
python scripts/train_q1.py --survey highz     # ~10 min on an A100, seed 0
python scripts/train_q1.py --survey lowz
python scripts/train_q2.py --survey highz
python scripts/train_q2.py --survey lowz
```

Retraining on different GPU hardware or drivers will not give bit-identical
weights, so use the shipped flows to reproduce the published numbers exactly.

### 2. MCMC runs

On Perlmutter (or any SLURM cluster, after editing `slurm/env.sh`):

```bash
export SBATCH_ACCOUNT=<your_account>
bash slurm/submit_paper.sh            # or: pure_ia | contam | model_comp | scaling
```

Every fit is skipped if its output already exists, so re-running only fills gaps.
`slurm/submit_paper.sh` submits:

| Paper item | Runs | Command per (cosmology, seed) | Seeds |
|---|---|---|---|
| Tables E1, E2; Fig. 5 | `cosmo{1,2,3}`, `cosmo{1,2,3}_cmb`, `cosmo{4,5}_cmb_bao` × flow, analytical | `infer_pure_ia.py --engine {flow,analytical}` | 0–24 |
| Fig. 5 (naive) | `cosmo2` | `infer_pure_ia.py --engine naive` | 1 |
| Tables 1, 2; Figs. 6, 7, E1 | `cosmo1` (M=10); `cosmo1_cmb`, `cosmo{2,3}`, `cosmo{2,3}_cmb`, `cosmo{4,5}_cmb_bao` (M=5) | `infer_contam.py --n_resamples M` | 0–24 |
| Table 3 (dust vs intrinsic step) | `cosmo1_dustnull`, `cosmo1_dust` | `infer_contam.py --dust_step --resamples_only` | 0–24 |
| Table 3 (constant vs evolving DE) | reuses `cosmo{4,5}_cmb_bao` from the row above | — | 0–24 |
| Fig. 8 (scaling) | `cosmo1` at 10× and 50× | `infer_contam.py --n_scale {10,50} --resamples_only --chunk_size 20000` | 0 |

Each fit runs 4 NUTS chains of 300 warm-up + 300 samples with maximum tree depth 6.
A ~2,500-object fit takes ≈10 min on an A100. The 10× and 50× scaling fits take
≈1 h and ≈5 h per resample. The full campaign is roughly 2,000 standard-size fits.

### 3. Tables and figures

```bash
bash paper/make_all.sh                # CPU is fine
```

| Output | Script | Default (cosmology, seed) |
|---|---|---|
| `tables/contam1-3.tex`, `contam4-5.tex` (Tables 1, 2) | `paper/table_contam.py` | seeds 0–24 |
| `tables/model_comp.tex` (Table 3) | `paper/table_model_comp.py` | seeds 0–24, M=5 |
| `tables/priors_table.tex` (Table C1) | `paper/table_priors.py` | — |
| `tables/ia_only_1-3.tex`, `ia_only_4-5.tex` (Tables E1, E2) | `paper/table_pure_ia.py` | seeds 0–24 |
| `figs/survey_sims.pdf` (Fig. 1) | `paper/fig_flows.py` | C1, key 0 |
| `figs/q1_true_cosmo1.pdf` (Fig. 4) | `paper/fig_flows.py` | C1, key 0 |
| `figs/pure_corners_cosmo2_seed1.pdf` (Fig. 5) | `paper/fig_pure_ia_corner.py` | C2, seed 1 |
| `figs/classification_histogram.pdf` (Fig. 6) | `paper/fig_classification.py` | C1, seed 0 |
| `figs/contam_corner_cosmo5_cmb_bao_seed1.pdf` (Fig. 7) | `paper/fig_contam_corner.py` | C5 + CMB + BAO, seed 1 |
| `figs/contam_scaling_cosmo1_seed0.pdf` (Fig. 8) | `paper/fig_scaling.py` | C1, seed 0, 1×/10×/50× |
| `figs/coverage_grid.pdf` (Fig. E1) | `paper/fig_coverage.py` | seeds 0–24 |

Figures 2 and 3 (graphical model and schematic) were drawn by hand and are not generated here.

## Experiments

### Cosmology tags

The tag selects the simulated truths and which external priors are added.

| Tag | Paper | w0 | wa | Ωm0 | ξ_rate | wa sampled? |
|---|---|---|---|---|---|---|
| `cosmo1` | C1 (fiducial, training) | −1.0 | 0 | 0.315 | 1.5 | no |
| `cosmo2` | C2 | −0.8 | 0 | 0.35 | 2.0 | no |
| `cosmo3` | C3 | −1.2 | 0 | 0.25 | 1.0 | no |
| `cosmo4` | C4 | −1.0 | 0 | 0.315 | 1.5 | yes |
| `cosmo5` | C5 (DESI DR2-like) | −0.752 | −0.86 | 0.316 | 1.5 | yes |

Modifiers:

- `_cmb` adds the CMB (R, ℓ_A) prior.
- `_cmb_bao` adds the CMB and BAO priors.
- `cosmo1_dustnull` and `cosmo1_dust` are the model-comparison simulations. High-mass hosts have τ = 0.098 and R_B = 3.0, and the intrinsic step is γ = 0 or γ = −0.06 respectively.

A modified tag analyses exactly the same SNe as its base cosmology. The SN
population truths are those of Tables 1 and E1. γ multiplies δ = 1 for high-mass
hosts (Eq. 11), so the main experiments' γ = +0.1 makes high-mass hosts 0.1 mag
fainter; tables and figures report γ exactly as sampled.

### Contamination analyses

`infer_contam.py` writes, per realisation:

| File | Meaning |
|---|---|
| `all_objects.npz` | every selected object, contaminants included |
| `pure_ia.npz` | the pure SN Ia sample (true labels) |
| `bayes_resample<m>.npz` | label allocation m ~ Bernoulli(classification score) |
| `bayes_resample_pooled.npz` | modular Bayesian posterior: all allocations pooled (Eq. 26) |

## Seeds and reproducibility

- **Pure SN Ia catalogue:** `PRNGKey(seed + 10^6 × block)`, where the block is 0–4 for C1–C5. The CMB and BAO observations are drawn from that key folded with 777 and 999. Every engine, and every modifier of a cosmology, therefore sees the same data.
- **Contaminated catalogue:** `PRNGKey(900000000 + seed)`. The MCMC keys are folded from it: 1 for all objects, 2 for pure SN Ia, 1000 + m for resample m. The resamples are independent of each other and of how many are run.
- **MCMC initialisation:** a fixed set of hand-chosen values (`experiments.INIT_FIXED`). For the `_cmb`/`_bao` runs, the cosmology starts at the maximum of the CMB/BAO likelihood; the truths are never used.
- **C1 SN-only resamples:** the published C1 SN-only contamination column pooled M = 10 resamples per seed; all other runs pooled M = 5. `submit_paper.sh` reproduces this. The first five resamples are the same whichever M is used.
- **Bit-level reproduction:** reproducing the chains to the last bit needs the pinned environment and the same kind of GPU. XLA reductions can differ in the final bit between hardware platforms, and NUTS amplifies such differences over a long chain.

## Acknowledgements

*FlowSN2* builds on [FlowJAX](https://github.com/danielward27/flowjax),
[numpyro](https://github.com/pyro-ppl/numpyro), [JAX](https://github.com/jax-ml/jax)
and [wcosmo](https://github.com/ColmTalbot/wcosmo). The `wcosmo/` directory is
the copy used for the paper's runs: wcosmo 0.5.0 (MIT licence, included) with
small changes for JAX tracing (`jnp.interp`, array-valued constants, a
log-spaced grid via `exp(linspace)`). Keep it as is to match the published
wCDM distances exactly.
