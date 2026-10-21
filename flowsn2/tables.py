"""Builder for the paper's parameter-recovery tables (Tables 1-2 and E1-E2).

Each cell is mean_s(posterior median_s) +/- std_s / sqrt(N) and its pull
(mean - truth) / SEM over the N realisations s.
"""
import numpy as np

from .experiments import resolve_truths
from .plotting import LABELS, PARAM_ORDER, seed_stat, fmt_cell

# (block label, base cosmology tag)
C13 = [(r"\bm{C}_1", "cosmo1"), (r"\bm{C}_2", "cosmo2"), (r"\bm{C}_3", "cosmo3")]
C45 = [(r"\bm{C}_4", "cosmo4"), (r"\bm{C}_5", "cosmo5")]
# (condition label, cosmology-tag suffix)
COND_C13 = [("SN Only", ""), ("SN + CMB Prior", "_cmb")]
COND_C45 = [("SN + CMB + BAO", "_cmb_bao")]


def point_estimates(paths_by_seed, seeds):
    """{param: per-seed posterior medians}."""
    out = {p: [] for p in PARAM_ORDER}
    keys = None
    for s in seeds:
        z = np.load(paths_by_seed[s])
        keys = set(z.files) if keys is None else keys & set(z.files)
        for p in PARAM_ORDER:
            out[p].append(float(np.median(np.asarray(z[p]).ravel()))
                          if p in z.files else np.nan)
    return {p: np.asarray(v) for p, v in out.items()}, keys


def build_table(blocks, conditions, series, find_files, caption, label, seeds):
    """LaTeX table*.

    series     : [(column header, key)] -- two sub-columns per condition
    find_files : (cosmo_tag, key) -> {seed: path to samples}
    """
    ncol = 2 + len(series) * len(conditions)
    lines = [r"\begin{table*}", r"\centering", r"\caption{" + caption + "}",
             r"\label{" + label + "}", r"\begin{tabular}{l" + "c" * (ncol - 1) + "}"]
    for block_label, base in blocks:
        data, present = {}, None
        for clabel, suffix in conditions:
            tag = base + suffix
            files = {key: find_files(tag, key) for _, key in series}
            shared = sorted(set(seeds).intersection(*(set(f) for f in files.values())))
            if len(shared) < 2:
                print(f"  (no runs for {tag} -> skipped)")
                data[clabel] = None
                continue
            if len(shared) != len(seeds):
                print(f"  {tag}: using {len(shared)} of {len(seeds)} requested seeds")
            ests = {}
            for _, key in series:
                ests[key], keys = point_estimates(files[key], shared)
                pres = [p for p in PARAM_ORDER if p in keys]
                present = pres if present is None else [p for p in present if p in pres]
            data[clabel] = ests
        if present is None:
            continue
        truths = resolve_truths(base)

        lines += [r"\hline", rf"\multicolumn{{{ncol}}}{{c}}{{Cosmology ${block_label}$}} \\",
                  r"\hline",
                  "Test &  & " + " & ".join(rf"\multicolumn{{{len(series)}}}{{c}}{{{cl}}}"
                                           for cl, _ in conditions) + r"\\",
                  r"\hline",
                  "Global Parameter & Truth  & "
                  + " & ".join(" & ".join(h for h, _ in series) for _ in conditions) + r" \\",
                  r"\hline"]
        for p in present:
            t = truths[p]
            cells = [LABELS.get(p, p), f"{t:g}"]
            for clabel, _ in conditions:
                if data[clabel] is None:
                    cells += ["-"] * len(series)
                    continue
                for _, key in series:
                    cells.append(fmt_cell(*seed_stat(data[clabel][key][p], t)))
            lines.append(" & ".join(cells) + r"\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"
