"""Figure 8: scaling -- the pooled Bayesian-resampling posterior of the
contaminated C1 catalogue at 1x, 10x and 50x the standard sample size.

    python paper/fig_scaling.py            # C1, seed 0 -> figs/contam_scaling_cosmo1_seed0.pdf

Reads results/contam/flow/cosmo1/seed0{,_x10,_x50}/bayes_resample_pooled.npz.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse

import numpy as np
import corner
from scipy.ndimage import gaussian_filter
from flowsn2.plotting import (use_paper_style, LABELS, LEVELS, block_decimals,
                              format_pdg)
use_paper_style()
from matplotlib.lines import Line2D

from flowsn2.experiments import resolve_truths
from flowsn2.paths import contam_dir, FIGURES

DEFAULT_PARAMS = ["w0", "Om0", "alpha", "beta_int", "beta_rate"]
COLOURS = ["#0072B2", "#D55E00", "#009E73"]     # small -> large catalogue
FILL_ALPHA = 0.10


def common_ranges(samples, params, truths):
    ranges = []
    for param in params:
        values = np.concatenate([s[param] for s in samples])
        lo, hi = np.percentile(values, (0.05, 99.95))
        lo = min(lo, truths[param])
        hi = max(hi, truths[param])
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            span = max(abs(lo), 1.0) * 0.05
        ranges.append((float(lo - 0.12 * span), float(hi + 0.12 * span)))
    return ranges


def _credible_levels(histogram):
    flat = np.sort(histogram.ravel())[::-1]
    cumulative = np.cumsum(flat) / flat.sum()
    thresholds = np.array([flat[np.searchsorted(cumulative, level, side="right")]
                           for level in LEVELS])
    thresholds.sort()
    for index in range(1, len(thresholds)):
        if thresholds[index] <= thresholds[index - 1]:
            thresholds[index] = thresholds[index - 1] * (1.0 + 1e-4)
    return thresholds


def add_shaded_contours(fig, data, ranges, colour):
    """Shade only the interior of the outer contour (corner would also fill the
    region outside its lowest level when several posteriors overlap)."""
    ndim = data.shape[1]
    axes = np.asarray(fig.axes).reshape((ndim, ndim))
    for row in range(1, ndim):
        for column in range(row):
            histogram, xedges, yedges = np.histogram2d(
                data[:, column], data[:, row], bins=28,
                range=[ranges[column], ranges[row]],
            )
            histogram = gaussian_filter(histogram, 0.9)
            thresholds = _credible_levels(histogram)
            xcentres = 0.5 * (xedges[:-1] + xedges[1:])
            ycentres = 0.5 * (yedges[:-1] + yedges[1:])
            axis = axes[row, column]
            axis.contourf(xcentres, ycentres, histogram.T,
                          levels=[thresholds[0], histogram.max() * (1 + 1e-4)],
                          colors=[colour], alpha=FILL_ALPHA, antialiased=True)
            axis.contour(xcentres, ycentres, histogram.T, levels=thresholds,
                         colors=[colour], linewidths=1.35)


def inference_annotation(scales, samples, truths, base_objects):
    lines = []
    for index, param in enumerate(("w0", "Om0")):
        if param not in samples[0]:
            continue
        if index:
            lines.append("")
        decimals = block_decimals([np.std(sample[param]) for sample in samples])
        if param == 'w0':
            lines.append(rf"{LABELS[param]}    Truth: ${truths[param]:.4g}$")
        else:
            lines.append(rf"{LABELS[param]}  Truth:${truths[param]:.4g}$")
        for scale, sample in zip(scales, samples):
            n_objects = int(round(scale * base_objects))
            lines.append(rf"$\quad\quad\quad${n_objects:,}: "
                         + format_pdg(np.mean(sample[param]), np.std(sample[param]), decimals))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cosmo", default="cosmo1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scales", nargs="+", type=float, default=[1, 10, 50])
    parser.add_argument("--params", nargs="+", default=DEFAULT_PARAMS)
    parser.add_argument("--base-objects", type=float, default=2500,
                        help="approximate number of objects in the 1x catalogue (legend)")
    args = parser.parse_args()

    loaded = []
    for scale, colour in zip(args.scales, COLOURS):
        path = contam_dir("flow", args.cosmo, args.seed, scale) / "bayes_resample_pooled.npz"
        if not path.exists():
            raise SystemExit(f"missing pooled posterior: {path}")
        with np.load(path) as archive:
            sample = {k: np.asarray(archive[k]).ravel() for k in archive.files}
        loaded.append((scale, sample, colour))

    params = args.params
    samples = []
    for _, sample, _ in loaded:
        samples.append({p: sample[p][np.isfinite(sample[p])] for p in params})
    truths = resolve_truths(args.cosmo)
    ranges = common_ranges(samples, params, truths)

    fig = None
    for index, ((scale, _, colour), sample) in enumerate(zip(loaded, samples)):
        data = np.column_stack([sample[p] for p in params])
        fig = corner.corner(
            data, fig=fig, labels=[LABELS.get(p, p) for p in params],
            truths=[truths[p] for p in params] if index == 0 else None,
            truth_color="black", range=ranges, color=colour, levels=LEVELS,
            smooth=0.9, smooth1d=2.0, bins=20, hist_bin_factor=3.0,
            plot_datapoints=False, plot_density=False, plot_contours=False,
            hist_kwargs={"lw": 1.6}, max_n_ticks=3, label_kwargs={"fontsize": 17},
        )
        add_shaded_contours(fig, data, ranges, colour)

    handles = [Line2D([], [], color=colour, lw=4,
                      label=rf"$\simeq${int(round(scale * args.base_objects)):,} objects")
               for scale, _, colour in loaded]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=20,
               bbox_to_anchor=(0.92, 0.86), handlelength=2.2)
    summary = inference_annotation(args.scales, samples, truths, args.base_objects)
    if summary:
        fig.text(0.64, 0.70, summary, ha="left", va="top", fontsize=20, linespacing=1.2)
    for axis in fig.get_axes():
        axis.tick_params(labelsize=16)

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"contam_scaling_{args.cosmo}_seed{args.seed}.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white", edgecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
