"""Shared conventions for the paper's figures and tables."""
import re

import numpy as np

LABELS = {
    'M0': r'$M_0^{\mathrm{int}}$', 'Om0': r'$\Omega_{m0}$', 'RB': r'$R_B$', 'alpha': r'$\alpha$',
    'alpha_c': r'$\alpha_c$', 'beta_int': r'$\beta_\mathrm{int}$',
    'beta_rate': r'$\xi_\mathrm{rate}$', 'c0_int': r'$c^{\mathrm{int}}_0$',
    'sigma_c': r'$\sigma_{c,\mathrm{int}}$', 'sigma_int': r'$\sigma_\mathrm{int}$',
    'sigma_x': r'$\sigma_x$', 'tau': r'$\tau$', 'w0': r'$w_0$', 'wa': r'$w_a$',
    'x0': r'$x_0$', 'gamma': r'$\gamma$',
}

# Table / corner row order: cosmology first, then the SN population.
PARAM_ORDER = ['w0', 'wa', 'Om0', 'M0', 'alpha', 'beta_int', 'alpha_c', 'RB',
               'c0_int', 'sigma_int', 'sigma_c', 'sigma_x', 'tau', 'x0',
               'beta_rate', 'gamma']

# 1, 2 and 3 sigma enclosed probabilities in two dimensions
LEVELS = (1 - np.exp(-0.5), 1 - np.exp(-2.0), 1 - np.exp(-4.5))

PAPER_SEEDS = "0-24"     # the 25 realisations behind every table and coverage panel


def use_paper_style():
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"


def expand_seeds(tokens):
    """['0-24', '30'] -> [0, 1, ..., 24, 30]"""
    if isinstance(tokens, str):
        tokens = [tokens]
    out = set()
    for tok in tokens:
        m = re.fullmatch(r"(\d+)-(\d+)", tok)
        if m:
            out.update(range(int(m.group(1)), int(m.group(2)) + 1))
        elif tok.isdigit():
            out.add(int(tok))
        else:
            raise SystemExit(f"bad seed token '{tok}' (use N or A-B)")
    return sorted(out)


def normalise_marginals(axes):
    """corner scales each diagonal to the LAST series drawn; rescale every 1D
    marginal to unit peak so all overlaid series stay in frame."""
    for ax in np.diag(axes):
        for line in ax.lines:
            y = np.asarray(line.get_ydata(), dtype=float)
            if y.size == 2 and y.min() == 0.0 and y.max() == 1.0:
                continue                          # the truth axvline, in axes coords
            peak = np.nanmax(y) if y.size else 0.0
            if np.isfinite(peak) and peak > 0:
                line.set_ydata(y / peak)
        ax.set_ylim(0.0, 1.25)


# ---- table cells: "mean +/- sem (pull sigma)" --------------------------------
def err_decimals(sem):
    """Decimals showing the error to 1 significant figure, or 2 when its leading
    digit is 1. Display only: pulls use the unrounded values."""
    exp = int(np.floor(np.log10(sem)))
    lead = int(sem / 10.0 ** exp)
    sig = 2 if lead == 1 else 1
    return int(min(6, max(0, -exp + (sig - 1))))


def seed_stat(vals, truth, ddof=1):
    """Mean of per-seed point estimates, its standard error, and the pull."""
    v = vals[~np.isnan(vals)]
    if len(v) < 2:
        return np.nan, np.nan, np.nan
    mean = v.mean()
    sem = v.std(ddof=ddof) / np.sqrt(len(v))
    pull = (mean - truth) / sem if sem > 0 else np.inf
    return mean, sem, pull


def fmt_cell(mean, sem, pull):
    if not np.isfinite(mean):
        return "-"
    if np.isfinite(sem) and sem > 0:
        ndec = err_decimals(sem)
        return rf"{mean:.{ndec}f} $\pm$ {sem:.{ndec}f} ({pull:.1f}$\sigma$)"
    return rf"{mean:.3f}"


# ---- figure annotations: PDG-rounded "mean +/- sd" ---------------------------
def pdg_decimals(standard_deviation):
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        return None
    exponent = int(np.floor(np.log10(abs(standard_deviation))))
    lead = int(abs(standard_deviation) / 10**exponent)
    significant = 2 if lead == 1 else 1
    return max(0, significant - 1 - exponent)


def block_decimals(standard_deviations):
    """One shared precision for rows of the same parameter: the coarsest PDG
    precision, refined until the tightest error does not round to zero."""
    counts = [pdg_decimals(s) for s in standard_deviations]
    counts = [c for c in counts if c is not None]
    if not counts:
        return None
    decimals = min(counts)
    smallest = min(s for s in standard_deviations if np.isfinite(s) and s > 0)
    while decimals < max(counts) and round(smallest, decimals) == 0.0:
        decimals += 1
    return decimals


def format_pdg(mean, standard_deviation, decimals=None):
    if decimals is None:
        decimals = pdg_decimals(standard_deviation)
    if decimals is None:
        return f"{mean:.4g}"
    error = round(standard_deviation, decimals)
    return rf"${mean:.{decimals}f} \pm {error:.{decimals}f}$"
