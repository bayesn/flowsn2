"""Table C1: inference priors (flowsn2.priors.PRIORS, used by both engines) and
the per-object training distributions (scripts/make_training_data.py).

    python paper/table_priors.py              # -> tables/priors_table.tex
"""
from flowsn2.priors import PRIORS
from flowsn2.experiments import TRUTHS_FIDUCIAL
from flowsn2.plotting import LABELS
from flowsn2.paths import TABLES

CAPTION = (r"Priors on the global parameters of used in our experiments. Both the "
           r"\textit{FlowSN2} and the analytical posteriors use the same priors. We denote a "
           r"normal distribution truncated between $[a,b]$ as $N(\mu,\sigma^2)_{[a,b]}$. Some "
           r"parameters are not varied in training and are kept at their fiducial values.  ")

# Training distribution per parameter: a fiducial value, or U(lo, hi) drawn per SN
# (must match scripts/make_training_data.py).
TRAINING = {
    "Om0": TRUTHS_FIDUCIAL["Om0"], "w0": TRUTHS_FIDUCIAL["w0"], "wa": 0,
    "M0": TRUTHS_FIDUCIAL["M0"], "beta_rate": TRUTHS_FIDUCIAL["beta_rate"], "gamma": 0,
    "alpha": (-0.2, 0), "beta_int": (1, 3), "alpha_c": (-0.02, 0), "RB": (2.2, 5),
    "tau": (0, 0.15), "c0_int": (-0.1, 0.1), "x0": (-0.6, -0.2),
    "sigma_int": (0, 0.2), "sigma_c": (0, 0.1), "sigma_x": (0, 1.5),
}
ROWS = [("Om0", ""), ("w0", ""), ("wa", r" (for $w_0w_a$CDM)"), ("M0", ""), ("beta_rate", ""),
        ("gamma", ""), ("alpha", ""), ("beta_int", ""), ("alpha_c", ""), ("RB", ""), ("tau", ""),
        ("c0_int", ""), ("x0", ""), ("sigma_int", ""), ("sigma_c", ""), ("sigma_x", "")]


def g(x):
    return f"{x:g}"


def prior_tex(name):
    family, a = PRIORS[name]
    if family == "Uniform":
        return rf"$U({g(a[0])},{g(a[1])})$"
    loc, scale, lo, hi = a
    return rf"$N({g(loc)},{g(scale)}^2)_{{[{g(lo)},{g(hi)}]}}$"


def training_tex(name):
    t = TRAINING[name]
    return rf"$U({g(t[0])},{g(t[1])})$" if isinstance(t, tuple) else g(t)


def main():
    lines = [r"\begin{table}", r"\centering", r"\caption{" + CAPTION + "}", r"\label{tab:priors}",
             r"\begin{tabular}{lcc}", r"\hline", r"Global Parameter & Prior & Training Range \\",
             r"\hline"]
    for name, qualifier in ROWS:
        lines.append(f"{LABELS[name]}{qualifier} & {prior_tex(name)} & {training_tex(name)} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "priors_table.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLES / 'priors_table.tex'}")


if __name__ == "__main__":
    main()
