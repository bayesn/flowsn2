"""Tables E1 and E2: pure SN Ia parameter recovery, FlowSN2 vs analytical.

    python paper/table_pure_ia.py            # -> tables/ia_only_1-3.tex, ia_only_4-5.tex

Reads results/pure_ia/{flow,analytical}/<cosmo>/seed<N>/samples.npz for seeds 0-24.
"""
import argparse
import re

from flowsn2.paths import PURE_IA, TABLES
from flowsn2.plotting import expand_seeds, PAPER_SEEDS
from flowsn2.tables import build_table, C13, C45, COND_C13, COND_C45

CAPTION_C13 = (
    r"Average of posterior median estimates over 25 repeats and the standard error on this "
    r"average for the photometric SN Ia cosmology experiments without contamination. We include "
    r"\textit{FlowSN2} inferences, as well as result that use the analytical likelihood solution "
    r"for comparison. We also combine SNe with a CMB prior for additional constraining as "
    r"detailed in Appendix~\ref{app:cmb_bao_prior}. Note uncertainties are the standard error on "
    r"the average so are a factor of five smaller than those in individual experiments. The "
    r"$\sigma$ values are calculated using unrounded averages and standard errors."
)
CAPTION_C45 = (
    r"Average of posterior median estimates over 25 repeats and the standard error on this "
    r"average for the photometric SN Ia cosmology experiments without contamination. Here we "
    r"inference results on $w_0w_a$CDM model parameters. We include \textit{FlowSN2} inferences, "
    r"as well as result that use the analytical likelihood solution for comparison. In order to "
    r"constrain the evolving dark energy model we include CMB and BAO priors as detailed in "
    r"Appendix~\ref{app:cmb_bao_prior}. Note uncertainties are the standard error on the average "
    r"so are a factor of five smaller than those in individual experiments. The $\sigma$ values "
    r"are calculated using unrounded averages and standard errors."
)
SERIES = [(r"\textit{FlowSN2}", "flow"), ("Analytical", "analytical")]


def find_files(cosmo_tag, engine):
    out = {}
    for p in (PURE_IA / engine / cosmo_tag).glob("seed*/samples.npz"):
        m = re.fullmatch(r"seed(\d+)", p.parent.name)
        if m:
            out[int(m.group(1))] = p
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="+", default=[PAPER_SEEDS])
    args = ap.parse_args()
    seeds = expand_seeds(args.seeds)
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, blocks, conds, caption, label in [
            ("ia_only_1-3", C13, COND_C13, CAPTION_C13, "tab:c1-c3_ia_only"),
            ("ia_only_4-5", C45, COND_C45, CAPTION_C45, "tab:c4-c5_ia_only")]:
        tex = build_table(blocks, conds, SERIES, find_files, caption, label, seeds)
        (TABLES / f"{name}.tex").write_text(tex)
        print(f"wrote {TABLES / name}.tex")


if __name__ == "__main__":
    main()
