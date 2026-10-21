"""Table 3: Savage-Dickey Bayes factors under contamination (Section 3.4).

    python paper/table_model_comp.py          # -> tables/model_comp.tex

For each realisation, the encompassing-model marginal of the tested parameter
(gamma, or wa for C4/C5) is Gaussian-fitted per label allocation and evaluated
at 0; the densities are averaged over the M allocations (Eq. 28) and compared
with the prior density at 0:  ln BF'_{1,0} = ln pi(0) - ln mean_m p_m(0).
Rows report the mean over seeds 0-24 and its standard error, with M = 5.

Reads results/contam/flow/<cosmo>/seed<N>/bayes_resample<m>.npz.
"""
import argparse
import re

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from flowsn2.experiments import is_w0wa
from flowsn2.priors import prior
from flowsn2.paths import CONTAM, TABLES
from flowsn2.plotting import expand_seeds, PAPER_SEEDS

ROWS = [
    ("cosmo1_dustnull", r"Varying dust only ($\gamma = 0$)"),
    ("cosmo1_dust",     r"Varying dust $+$ intrinsic step ($\gamma = -0.06$)"),
    ("cosmo4_cmb_bao",  r"Constant dark energy ($w_a = 0$)"),
    ("cosmo5_cmb_bao",  r"Evolving dark energy ($w_a = -0.86$)"),
]
SYMBOL = {"gamma": r"\gamma", "wa": r"w_a"}


def evidence(lnB, sym):
    """Jeffreys scale on the natural-log Bayes factor (0.5 / 1 / 2 in log10)."""
    a = abs(lnB)
    strength = ("Barely worth mentioning" if a < 1.1513 else
                "Substantial"             if a < 2.3026 else
                "Strong"                  if a < 4.6052 else
                "Decisive")
    favours = rf"${sym} \neq 0$" if lnB > 0 else rf"${sym} = 0$"
    return f"{strength} evidence for {favours}"


def ln_bf(cosmo, seeds, M, engine="flow"):
    """Per-seed SDDR ln BF'_{1,0} using the first M allocations of each seed."""
    param = "wa" if is_w0wa(cosmo) else "gamma"
    log_pi0 = float(prior(param).log_prob(0.0))
    out = []
    for seed in seeds:
        d = CONTAM / engine / cosmo / f"seed{seed}"
        files = sorted(d.glob("bayes_resample[0-9]*.npz"),
                       key=lambda f: int(re.search(r"(\d+)\.npz$", f.name).group(1)))
        if len(files) < M:
            print(f"  {cosmo} seed{seed}: only {len(files)} allocations -> skipped")
            continue
        log_pm0 = []
        for f in files[:M]:
            x = np.asarray(np.load(f)[param]).ravel()
            log_pm0.append(norm.logpdf(0.0, loc=float(x.mean()), scale=float(x.std(ddof=1))))
        out.append(log_pi0 - (logsumexp(log_pm0) - np.log(M)))
    return param, np.asarray(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", default=[PAPER_SEEDS])
    ap.add_argument("--M", type=int, default=5, help="label allocations per realisation")
    args = ap.parse_args()
    seeds = expand_seeds(args.seeds)

    rows = []
    for cosmo, label in ROWS:
        param, lb = ln_bf(cosmo, seeds, args.M)
        if len(lb) < 2:
            print(f"  (no complete runs for {cosmo} -> row skipped)")
            continue
        mean, sem = lb.mean(), lb.std(ddof=1) / np.sqrt(len(lb))
        print(f"  {cosmo:16s} {param:5s} N={len(lb)}  ln BF = {mean:+.2f} +/- {sem:.2f}")
        rows.append(f"        {label} & ${mean:+.2f} \\pm {sem:.2f}$ & "
                    f"{evidence(mean, SYMBOL[param])} \\\\")

    tex = "\n".join([
        r"\begin{table*}",
        r"    \centering",
        r"    \caption{Savage--Dickey Bayes factors for the nested model comparisons, each",
        rf"    averaged over {len(seeds)} simulated realisations with $M={args.M}$ label resamples",
        r"    per realisation. Uncertainties are the standard error on the mean $\ln \text{BF}'_{1,0}$.",
        r"    Positive $\ln \text{BF}'_{1,0}$ favours the model in which the parameter is free. Strength of",
        r"    evidence follows the Jeffreys scale. Experimental details can be found in Section~\ref{sec:model_comparison}.}",
        r"    \label{tab:bayes_factors}",
        r"    \begin{tabular}{lcl}",
        r"        \hline",
        r"        Simulation & $\ln \text{BF}'_{1,0}$ & Interpretation \\",
        r"        \hline",
        *rows,
        r"        \hline",
        r"    \end{tabular}",
        r"\end{table*}",
    ]) + "\n"
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "model_comp.tex").write_text(tex)
    print(f"wrote {TABLES / 'model_comp.tex'}")


if __name__ == "__main__":
    main()
