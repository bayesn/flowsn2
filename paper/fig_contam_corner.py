"""Figure 7: w0waCDM posteriors under contamination -- all objects (contaminants
included), the pure SN Ia sample (true labels) and modular Bayesian resampling.

    python paper/fig_contam_corner.py     # C5 + CMB + BAO, seed 1 -> figs/contam_corner_cosmo5_cmb_bao_seed1.pdf

Reads results/contam/flow/<cosmo>/seed<N>/{all_objects,pure_ia,bayes_resample_pooled}.npz.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse

import numpy as np
import corner
from flowsn2.plotting import (use_paper_style, LABELS, LEVELS, flip, normalise_marginals,
                              block_decimals, format_pdg)
use_paper_style()
from matplotlib.lines import Line2D

from flowsn2.experiments import resolve_truths
from flowsn2.paths import contam_dir, FIGURES

# (legend label, samples file, colour, linewidth), in DRAWING order
SERIES = [("All Objects",         "all_objects",           "0.62",    3.0),
          ("Pure SN Ia Sample",   "pure_ia",               "#0072B2", 5.0),
          ("Bayesian Resampling", "bayes_resample_pooled", "#C44E52", 3.5)]
DEFAULT_PARAMS = ['wa', 'w0', 'Om0', 'alpha', 'beta_int', 'beta_rate', 'sigma_int', 'M0']
FS = 22


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cosmo", default="cosmo5_cmb_bao")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--engine", default="flow")
    ap.add_argument("--params", nargs="*", default=DEFAULT_PARAMS)
    ap.add_argument("--pct", type=float, default=0.4,
                    help="percentile clipped off each end when setting the axis ranges")
    ap.add_argument("--annotate", nargs="*", default=["w0", "wa", "Om0"])
    args = ap.parse_args()

    root = contam_dir(args.engine, args.cosmo, args.seed)
    samples = []
    for _, name, _, _ in SERIES:
        path = root / f"{name}.npz"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        z = np.load(path)
        samples.append({k: flip(k) * np.asarray(z[k]).ravel() for k in z.files})

    params = [p for p in args.params if all(p in s for s in samples)]
    TRUTHS = resolve_truths(args.cosmo)
    truths = [flip(p) * TRUTHS[p] + 0.0 for p in params]
    data = [np.stack([s[p] for p in params], axis=-1) for s in samples]

    # union of each series' own percentile interval (the pooled resample carries
    # M times the draws, so pooling first would clip the far-off contours)
    ranges = [(min(float(np.percentile(d[:, i], args.pct)) for d in data),
               max(float(np.percentile(d[:, i], 100.0 - args.pct)) for d in data))
              for i in range(len(params))]

    kw = dict(labels=[LABELS.get(p, p) for p in params], range=ranges, levels=LEVELS,
              smooth=1.2, smooth1d=2.0, bins=20, hist_bin_factor=3.0, max_n_ticks=2,
              plot_density=False, plot_datapoints=False, fill_contours=False,
              show_titles=False, truth_color="black",
              label_kwargs=dict(fontsize=FS + 13))

    fig = None
    for d, (_, _, colour, lw) in zip(data, SERIES):
        fig = corner.corner(d, fig=fig, truths=truths, color=colour,
                            contour_kwargs=dict(linewidths=lw),
                            hist_kwargs=dict(lw=lw), **kw)
    fig.set_size_inches(3.0 * len(params), 3.0 * len(params))

    n = len(params)
    axes = np.array(fig.axes).reshape((n, n))
    for ax in fig.get_axes():
        ax.tick_params(axis="both", labelsize=FS - 1)
        if ax.get_xlabel():
            ax.xaxis.label.set_fontsize(FS + 13)
            ax.xaxis.label.set_y(ax.xaxis.label.get_position()[1] - 0.10)
        if ax.get_ylabel():
            ax.yaxis.label.set_fontsize(FS + 13)
            ax.yaxis.label.set_x(ax.yaxis.label.get_position()[0] - 0.13)
    normalise_marginals(axes)

    handles = [Line2D([], [], color=c, lw=6, label=lab) for lab, _, c, _ in SERIES]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.97),
               frameon=False, fontsize=FS + 20, handlelength=1.8,
               labelspacing=0.5, handletextpad=0.8)

    probe = " + ".join(["SN"] + [p.upper() for p in ("cmb", "bao") if f"_{p}" in args.cosmo])
    fig.text(0.38, 0.845, probe, ha="left", va="top", fontsize=FS + 30)

    summary = []
    for i, param in enumerate([p for p in args.annotate if p in params]):
        j = params.index(param)
        if i:
            summary.append("")
        decimals = block_decimals([float(np.std(d[:, j])) for d in data])
        summary.append(rf"{LABELS.get(param, param)}    Truth: ${truths[j]:.4g}$")
        for d, (lab, _, _, _) in zip(data, SERIES):
            summary.append(rf"$\quad${lab}: "
                           + format_pdg(float(np.mean(d[:, j])), float(np.std(d[:, j])),
                                        decimals))
    if summary:
        fig.text(0.635, 0.79, "\n".join(summary), ha="left", va="top", fontsize=FS + 6,
                 linespacing=1.38)

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"contam_corner_{args.cosmo}_seed{args.seed}.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white", edgecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
