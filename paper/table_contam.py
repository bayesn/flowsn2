"""Tables 1 and 2: parameter recovery under contamination (FlowSN2), modular
Bayesian resampling vs the pure SN Ia sample with the true labels.

    python paper/table_contam.py             # -> tables/contam1-3.tex, contam4-5.tex

Reads results/contam/flow/<cosmo>/seed<N>/{bayes_resample_pooled,pure_ia}.npz for
seeds 0-24.
"""
import argparse
import re

from flowsn2.paths import CONTAM, TABLES
from flowsn2.plotting import expand_seeds, PAPER_SEEDS
from flowsn2.tables import build_table, C13, C45, COND_C13, COND_C45

CAPTION_C13 = (
    r"Average of posterior median estimates over 25 repeats and the standard error on this "
    r"average for the full photometric SN Ia cosmology experiments with non-Ia contamination. "
    r"We include results using our default modular Bayesian approach of resampling of the data "
    r"($M=5$) to handle contamination, as well as results using the pure SN Ia sample with the "
    r"true labels for comparison. Both use the \textit{FlowSN2} normalising flows in their "
    r"inference. We also combine SNe with a CMB prior for additional constraining as detailed "
    r"in Appendix~\ref{app:cmb_bao_prior}. Note uncertainties are the standard error on the "
    r"average so are a factor of five smaller than those in individual experiments. The "
    r"$\sigma$ values are calculated using unrounded averages and standard errors."
)
CAPTION_C45 = (
    r"Average of posterior median estimates over 25 repeats and the standard error on this "
    r"average for the full photometric SN Ia cosmology experiments with non-Ia contamination. "
    r"Here we inference results on $w_0w_a$CDM model parameters. We include results using our "
    r"default modular Bayesian approach of resampling of the data ($M=5$) to handle "
    r"contamination, as well as results using the pure SN Ia sample with the true labels for "
    r"comparison. Both use the \textit{FlowSN2} normalising flows in their inference. In order "
    r"to constrain the evolving dark energy model we include CMB and BAO priors as detailed in "
    r"Appendix~\ref{app:cmb_bao_prior}. Note uncertainties are the standard error on the "
    r"average so are a factor of five smaller than those in individual experiments. The "
    r"$\sigma$ values are calculated using unrounded averages and standard errors."
)
SERIES = [("Bayesian Resampling", "bayes_resample_pooled"), ("Pure SN Ia Sample", "pure_ia")]


def find_files(cosmo_tag, key, engine="flow"):
    out = {}
    for p in (CONTAM / engine / cosmo_tag).glob(f"seed*/{key}.npz"):
        m = re.fullmatch(r"seed(\d+)", p.parent.name)      # excludes the _x<scale> runs
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
            ("contam1-3", C13, COND_C13, CAPTION_C13, "tab:c1-c3_contam"),
            ("contam4-5", C45, COND_C45, CAPTION_C45, "tab:c4-c5_contam")]:
        tex = build_table(blocks, conds, SERIES, find_files, caption, label, seeds)
        (TABLES / f"{name}.tex").write_text(tex)
        print(f"wrote {TABLES / name}.tex")


if __name__ == "__main__":
    main()
