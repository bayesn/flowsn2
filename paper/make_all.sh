#!/bin/bash
# Rebuild every table and data figure of the paper from results/ (CPU is fine).
#   bash paper/make_all.sh          -> tables/*.tex, figs/*.pdf
set -e
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=cpu

python paper/table_priors.py         # Table C1
python paper/table_pure_ia.py        # Tables E1, E2
python paper/table_contam.py         # Tables 1, 2
python paper/table_model_comp.py     # Table 3

python paper/fig_flows.py            # Figures 1, 4 (flow weights only)
python paper/fig_pure_ia_corner.py   # Figure 5
python paper/fig_classification.py   # Figure 6
python paper/fig_contam_corner.py    # Figure 7
python paper/fig_scaling.py          # Figure 8
python paper/fig_coverage.py         # Figure E1
