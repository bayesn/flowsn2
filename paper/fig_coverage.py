"""Figure E1: frequentist coverage of the contaminated-sample posteriors.

    python paper/fig_coverage.py              # -> figs/coverage_grid.pdf

Rows: SN only (C1-C3), SN + CMB (C1-C3), SN + CMB + BAO (C4, C5); columns:
Bayesian resampling, pure SN Ia sample. Each curve is the empirical coverage of
the central credible interval (C_r = 2|F(truth) - 0.5|) pooled over the row's
cosmologies, seeds 0-24; the grey band is the 95% simultaneous Monte-Carlo band
for a calibrated posterior. Reads results/contam/flow/<cosmo>/seed<N>/.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import re

import numpy as np
from flowsn2.plotting import use_paper_style, LABELS, expand_seeds, PAPER_SEEDS
use_paper_style()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from flowsn2.experiments import reported_truths
from flowsn2.paths import CONTAM, FIGURES

PARAM_ORDER = ['w0', 'wa', 'Om0', 'M0', 'beta_rate', 'alpha', 'beta_int', 'alpha_c',
               'RB', 'tau', 'c0_int', 'sigma_int', 'sigma_c', 'sigma_x', 'x0', 'gamma']
_WHEEL = [plt.cm.tab10(i) for i in range(8)]
PARAM_STYLE = {p: dict(color=_WHEEL[i % 8], linestyle="-" if i < 8 else "--")
               for i, p in enumerate(PARAM_ORDER)}

ROWS = [("SN Only",        ["cosmo1", "cosmo2", "cosmo3"]),
        ("SN + CMB",       ["cosmo1_cmb", "cosmo2_cmb", "cosmo3_cmb"]),
        ("SN + CMB + BAO", ["cosmo4_cmb_bao", "cosmo5_cmb_bao"])]
COLS = [("Bayesian Resampling", "bayes_resample_pooled.npz"),
        ("Pure SN Ia Sample",   "pure_ia.npz")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", default=[PAPER_SEEDS])
    ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--grid", type=int, default=500)
    ap.add_argument("--band_sims", type=int, default=40000)
    args = ap.parse_args()
    want = set(expand_seeds(args.seeds))
    levels = np.linspace(0, 1, args.grid)

    def cred_1d(s, t):
        return 2.0 * abs(np.mean(s < t) - 0.5)

    def curve(cl):
        cl = np.asarray(cl)
        return np.array([np.mean(cl <= L) for L in levels])

    def panel_creds(cosmos, samples_name):
        creds = {p: [] for p in PARAM_ORDER}
        N = 0
        for cosmo in cosmos:
            truths = reported_truths(cosmo)
            for d in sorted((CONTAM / "flow" / cosmo).glob("seed*")):
                m = re.fullmatch(r"seed(\d+)", d.name)
                if not m or int(m.group(1)) not in want:
                    continue
                p = d / samples_name
                if not p.exists():
                    continue
                z = np.load(p)
                N += 1
                for pr in PARAM_ORDER:
                    if pr in z.files:
                        creds[pr].append(cred_1d(np.asarray(z[pr]).ravel(), truths[pr]))
        return creds, N

    def _smooth(y, frac=0.05):
        w = max(3, (int(len(y) * frac) // 2) * 2 + 1)
        k = np.hanning(w)
        k /= k.sum()
        yp = np.pad(y, w, mode="edge")
        return np.convolve(yp, k, mode="same")[w:-w]

    band_cache = {}

    def band(N):
        """95% simultaneous band for N calibrated realisations (Monte Carlo)."""
        if N < 2:
            return None
        if N not in band_cache:
            rng = np.random.default_rng(0)
            E = np.empty((args.band_sims, len(levels)), dtype=np.float32)
            for i in range(0, args.band_sims, 4000):
                u = rng.random((min(4000, args.band_sims - i), N))
                E[i:i + u.shape[0]] = (u[:, None, :] <= levels[None, :, None]).mean(axis=2)
            a, b = 1e-4, 0.5
            for _ in range(45):
                q = 0.5 * (a + b)
                lo = np.quantile(E, q, axis=0)
                hi = np.quantile(E, 1 - q, axis=0)
                if np.all((E >= lo - 1e-9) & (E <= hi + 1e-9), axis=1).mean() > args.conf:
                    a = q
                else:
                    b = q
            band_cache[N] = (_smooth(np.quantile(E, q, axis=0)),
                             _smooth(np.quantile(E, 1 - q, axis=0)))
        return band_cache[N]

    fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(11, 14))
    for r, (rlabel, cosmos) in enumerate(ROWS):
        for c, (clabel, sname) in enumerate(COLS):
            ax = axes[r, c]
            creds, N = panel_creds(cosmos, sname)
            bd = band(N)
            if bd is not None:
                ax.fill_between(levels * 100, bd[0] * 100, bd[1] * 100, color="grey", alpha=0.2)
            for p in PARAM_ORDER:
                if creds[p]:
                    ax.plot(levels * 100, curve(creds[p]) * 100, lw=1.5, **PARAM_STYLE[p])
            ax.plot([0, 100], [0, 100], color="black", lw=1)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_xlabel("Nominal Bayesian Credible Level (%)", fontsize=12)
            ax.set_ylabel("Empirical Coverage Probability (%)", fontsize=12)
            ax.tick_params(labelsize=10)
            ax.text(0.05, 0.95, f"{rlabel}\n{clabel}\n$N={N}$", transform=ax.transAxes,
                    va="top", ha="left", fontsize=12.5, linespacing=1.4)

    handles = [Line2D([0], [0], lw=2, label=LABELS.get(p, p), **PARAM_STYLE[p])
               for p in PARAM_ORDER]
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    ax00 = axes[0, 0]
    bb = ax00.get_position()
    fw, fh = fig.get_size_inches()
    ang = np.degrees(np.arctan2(bb.height * fh, bb.width * fw))
    for x, y, txt in [(23, 62, "Conservative"), (62, 23, "Overconfident")]:
        ax00.text(x, y, txt, rotation=ang, rotation_mode="anchor", ha="center", va="center",
                  fontsize=12.5, color="0.35")

    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, 0.0))

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "coverage_grid.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white", edgecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
