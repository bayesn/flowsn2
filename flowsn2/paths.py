"""Repository locations. Every input and output path is resolved from here, so
scripts can be launched from any working directory. FLOWSN2_RESULTS and
FLOWSN2_TRAINING_DATA override the two large-data locations (e.g. on $PSCRATCH)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WEIGHTS = ROOT / "flow_weights"          # trained flows shipped with the repo
TRAINING_DATA = Path(os.environ.get("FLOWSN2_TRAINING_DATA", ROOT / "training_data"))
RESULTS = Path(os.environ.get("FLOWSN2_RESULTS", ROOT / "results"))   # MCMC samples
FIGURES = ROOT / "figs"                  # paper figures
TABLES = ROOT / "tables"                 # paper tables

# results/pure_ia/<engine>/<cosmo>/seed<N>/samples.npz
#   engine = flow | analytical | naive   (naive = analytical, selection ignored)
PURE_IA = RESULTS / "pure_ia"
# results/contam/<engine>/<cosmo>/seed<N>[_x<scale>]/{all_objects,pure_ia,
#   bayes_resample<m>,bayes_resample_pooled}.npz
CONTAM = RESULTS / "contam"


def pure_ia_dir(engine, cosmo, seed):
    return PURE_IA / engine / cosmo / f"seed{seed}"


def scale_tag(n_scale):
    """Directory suffix for a sample-size multiplier (none at the standard size)."""
    return "" if float(n_scale) == 1.0 else f"_x{float(n_scale):g}"


def contam_dir(engine, cosmo, seed, n_scale=1.0):
    return CONTAM / engine / cosmo / f"seed{seed}{scale_tag(n_scale)}"
