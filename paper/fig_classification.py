"""Figure 6: mock classification scores of the selected high-z photometric
objects, split by true class. Rebuilds the catalogue the inference used.

    python paper/fig_classification.py        # C1, seed 0 -> figs/classification_histogram.pdf
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse

import numpy as np
from flowsn2.plotting import use_paper_style
use_paper_style()
import matplotlib.pyplot as plt

from flowsn2.contamination import build_catalog
from flowsn2.paths import FIGURES

FS = 18


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cosmo", default="cosmo1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=22)
    ap.add_argument("--annotation",
                    default=r"$\sim 20\%$ post-selection" "\n" "contamination rate")
    args = ap.parse_args()

    grouped, _, _ = build_catalog(seed=args.seed, cosmo=args.cosmo, verbose=False)
    hz = next(g for g in grouped if not g["lowz"])
    q = np.asarray(hz["class_score"])
    is_ia = np.asarray(hz["is_ia"]).astype(bool)
    n_all = sum(len(np.asarray(g["class_score"])) for g in grouped)
    print(f"  high-z selected: {len(q):,} ({(~is_ia).sum():,} contaminants); "
          f"{100 * (~is_ia).sum() / n_all:.1f}% of all {n_all:,} selected objects")

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    bins = np.linspace(0, 1, args.bins)
    ax.hist(q[is_ia], bins=bins, histtype="stepfilled", alpha=0.55, color="tab:green",
            label="SN Ia")
    ax.hist(q[~is_ia], bins=bins, histtype="stepfilled", alpha=0.55, color="tab:red",
            label="Contaminants")
    ax.set_xlabel("SN Ia Classification Score", fontsize=FS + 2)
    ax.set_ylabel("Number of Objects", fontsize=FS + 2)
    ax.tick_params(axis="both", labelsize=FS)
    ax.set_xlim(0, 1)
    tops = [np.histogram(x, bins=bins)[0].max() for x in (q[is_ia], q[~is_ia])]
    ax.set_ylim(0, max(tops) * 1.02)
    ax.legend(frameon=False, fontsize=FS, loc="upper left")
    if args.annotation:
        ax.text(0.035, 0.76, args.annotation, transform=ax.transAxes, ha="left", va="top",
                fontsize=FS + 1)

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "classification_histogram.pdf"
    fig.savefig(out, facecolor="white", edgecolor="white", bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
