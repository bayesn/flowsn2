"""Figure 5: pure SN Ia posteriors -- FlowSN2 vs analytical vs naive (selection
ignored), on the same catalogue. Three corners in a pyramid (cosmology + rate,
dust, light-curve/intrinsic).

    python paper/fig_pure_ia_corner.py          # C2, seed 1 -> figs/pure_corners_cosmo2_seed1.pdf
    python paper/fig_pure_ia_corner.py --cosmo cosmo1 --seed 0

Reads results/pure_ia/{naive,analytical,flow}/<cosmo>/seed<N>/samples.npz.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse

import numpy as np
import corner
from flowsn2.plotting import use_paper_style, LABELS, LEVELS, normalise_marginals
use_paper_style()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from flowsn2.experiments import reported_truths
from flowsn2.paths import pure_ia_dir, FIGURES

# (legend label, engine, colour, linewidth), in DRAWING order
SERIES = [("Naive",               "naive",      "0.62",    1.5),
          ("Analytical",          "analytical", "#D95F02", 4.0),
          (r"$\mathit{FlowSN2}$", "flow",       "#1B9E77", 1.8)]
LEGEND_ORDER = [2, 1, 0]
BLOCKS = [
    ['w0', 'Om0', 'beta_rate', 'M0', 'gamma'],
    ['tau', 'RB', 'beta_int', 'c0_int', 'sigma_c'],
    ['alpha', 'x0', 'sigma_x', 'alpha_c', 'sigma_int'],
]
FS = 17


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cosmo", default="cosmo2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pad", type=float, default=0.15, help="axis padding fraction")
    args = ap.parse_args()

    samples = []
    for _, engine, _, _ in SERIES:
        path = pure_ia_dir(engine, args.cosmo, args.seed) / "samples.npz"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        z = np.load(path)
        samples.append({k: np.asarray(z[k]).ravel() for k in z.files})
    TRUTHS = reported_truths(args.cosmo)

    SIDE = 10.0
    fig = plt.figure(figsize=(2 * SIDE, 2 * SIDE))
    gs = fig.add_gridspec(2, 4, wspace=0.0, hspace=0.0)
    slots = [gs[0, 1:3], gs[1, 0:2], gs[1, 2:4]]

    for block, slot in zip(BLOCKS, slots):
        params = [p for p in block if all(p in s for s in samples)]
        data = [np.stack([s[p] for p in params], axis=-1) for s in samples]
        truths = [TRUTHS[p] for p in params]
        stacked = np.vstack(data)
        lo, hi = stacked.min(axis=0), stacked.max(axis=0)
        ranges = [(a - args.pad * (b - a), b + args.pad * (b - a)) for a, b in zip(lo, hi)]

        kw = dict(levels=LEVELS, smooth=1.0, smooth1d=2.0, bins=20, hist_bin_factor=5.0,
                  max_n_ticks=3, plot_density=False, plot_datapoints=False,
                  fill_contours=False, show_titles=False, truth_color="black",
                  range=ranges, label_kwargs=dict(fontsize=FS + 3))

        sf = fig.add_subfigure(slot)
        sf.patch.set_visible(False)
        cfig = None
        for d, (_, _, colour, lw) in zip(data, SERIES):
            cfig = corner.corner(d, fig=sf if cfig is None else cfig,
                                 labels=[LABELS.get(p, p) for p in params],
                                 truths=truths, color=colour,
                                 contour_kwargs=dict(linewidths=lw),
                                 hist_kwargs=dict(lw=lw), **kw)

        n = len(params)
        axes = np.array(sf.get_axes()).reshape((n, n))
        for ax in sf.get_axes():
            ax.tick_params(axis="both", labelsize=FS - 3)
            if ax.get_xlabel():
                ax.xaxis.label.set_fontsize(FS + 3)
                ax.xaxis.label.set_y(ax.xaxis.label.get_position()[1] - 0.12)
            if ax.get_ylabel():
                ax.yaxis.label.set_fontsize(FS + 3)
                ax.yaxis.label.set_x(ax.yaxis.label.get_position()[0] - 0.16)
        normalise_marginals(axes)

    handles = [Line2D([], [], color=SERIES[i][2], lw=6, label=SERIES[i][0])
               for i in LEGEND_ORDER]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.74, 0.94),
               frameon=False, fontsize=FS + 18, handlelength=1.8,
               labelspacing=0.5, handletextpad=0.8)

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"pure_corners_{args.cosmo}_seed{args.seed}.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white", edgecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
