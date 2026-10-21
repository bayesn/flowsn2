"""Figure 1 (survey simulations) and Figure 4 (what q1 learnt). Needs only the
shipped flow weights -- no MCMC results.

    python paper/fig_flows.py                   # -> figs/survey_sims.pdf, figs/q1_true_cosmo1.pdf
    python paper/fig_flows.py --only survey

Figure 1: apparent magnitude vs redshift for the selected low-z / high-z SNe Ia
and selected contaminants, over all objects, with each survey's selection
functions on the marginals.
Figure 4: q1 samples for faint high-z SNe (1.13 < z < 1.17), each conditioned on
its own latent (z, m0), against the selected and the full simulated
populations at the same latents.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from numpyro.infer import Predictive
from scipy.special import ndtr
from scipy.ndimage import gaussian_filter1d

import corner
from flowsn2.plotting import use_paper_style, LEVELS
use_paper_style()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from flowsn2 import flow_likelihood as fl
from flowsn2.experiments import resolve_truths, SURVEYS
from flowsn2.simulator import sample_model
from flowsn2.cosmology import mu_func
from flowsn2.paths import FIGURES

C_FLOW = "#1B9E77"    # FlowSN2
C_POP = "0.7"         # all objects / no selection
C_TRUE = "#D95F02"    # selected simulation
C_HIGHZ = "#0072B2"   # high-z photometric SNe Ia
C_LOWZ = "#C44E52"    # low-z spectroscopic SNe Ia
C_CONTAM = "#8C6D31"  # contaminants

HIGHZ = next(s for s in SURVEYS if not s["lowz"])
LOWZ = next(s for s in SURVEYS if s["lowz"])
SEL_A, SEL_B = -0.1, -1.0
HIGHZ_CUTS = (HIGHZ["m_cut"], HIGHZ["sigma_mcut"], HIGHZ["z_cut"], HIGHZ["sigma_zcut"],
              SEL_A, SEL_B)
SIGMA_PEC = 300.0
SIGMA_Z_FACTOR = HIGHZ["sigma_z_factor"]
M0_FID = fl.M0_FID
_C_KMS = 299792.458


def finite_rows(a):
    return a[np.isfinite(a).all(axis=1)]


def subsample(a, m, rng):
    if len(a) <= m:
        return a
    return a[rng.choice(len(a), m, replace=False)]


def mu_curve(z, Om0, w0):
    return np.asarray(jax.vmap(lambda zz: mu_func(zz, Om0, w0))(jnp.asarray(z)))


def smooth_hist(data, edges, sigma=1.5, density=True, peak=False):
    h, _ = np.histogram(data, bins=edges, density=density)
    h = gaussian_filter1d(h.astype(float), sigma)
    if peak and h.max() > 0:
        h = h / h.max()
    centres = 0.5 * (edges[:-1] + edges[1:])
    centres = np.concatenate([[edges[0]], centres, [edges[-1]]])
    h = np.concatenate([[0.0], h, [0.0]])
    return centres, h


def fit_marginal_ylims(fig, k, pad=1.08):
    axes = np.array(fig.axes).reshape((k, k))
    for i in range(k):
        peaks = [np.nanmax(ln.get_ydata()) for ln in axes[i, i].get_lines()
                 if ln.get_ydata().size]
        if peaks:
            axes[i, i].set_ylim(0, max(peaks) * pad)


def sample_flow_per_cond(flow, mu_std, std_std, n_tgt, cond_arr, key):
    """One sample per row of cond_arr (n, n_cond), in original target space."""
    cond_std = (jnp.asarray(cond_arr) - mu_std[n_tgt:]) / std_std[n_tgt:]
    keys = jr.split(key, cond_std.shape[0])
    samp = jax.vmap(lambda k, c: flow.sample(k, condition=c))(keys, cond_std)
    return samp * std_std[:n_tgt] + mu_std[:n_tgt]


# ---------------------------------------------------------------------------
# Figure 4 -- q1
# ---------------------------------------------------------------------------
def representative_errors(m0_star, truths):
    """Mean-error-model covariance block for a SN of magnitude ~m0_star; the same
    block conditions the flow and drives the comparison simulation."""
    x0, c0, ac = truths["x0"], truths["c0_int"], truths["alpha_c"]
    al, be = truths["alpha"], truths["beta_int"]
    RB, tau = truths["RB"], truths["tau"]
    m_s_typ = m0_star + al * x0 + be * (c0 + ac * x0) + RB * tau
    sig_m = np.exp(0.1 * (m_s_typ - 56.0))
    sig_x = np.exp(-1.5)
    sig_c = np.exp(-3.5)
    cov_m_c = 0.000520 + 0.288792999 * sig_m ** 2
    cov_m_x = 0.0008097 + 0.03835550 * sig_x ** 2
    cov_c_x = 0.000168655 + 0.01358504 * sig_x ** 2
    W = np.array([[sig_m ** 2, cov_m_c, cov_m_x],
                  [cov_m_c, sig_c ** 2, cov_c_x],
                  [cov_m_x, cov_c_x, sig_x ** 2]])
    vals, vecs = np.linalg.eigh(W)
    W = (vecs * np.maximum(vals, 1e-5)) @ vecs.T
    return W, sig_m, sig_c, sig_x, cov_m_c, cov_m_x, cov_c_x


def simulate_observed_at_latents(z_star, m0_star, W, truths, key, n, sel_cuts=None):
    """Observed [m, c, x, z_obs] at fixed latents and error block, following the
    forward model; with sel_cuts also the survey selection mask."""
    x0, c0, ac = truths["x0"], truths["c0_int"], truths["alpha_c"]
    al, be = truths["alpha"], truths["beta_int"]
    RB, tau = truths["RB"], truths["tau"]
    si = truths["sigma_int"]
    sc, sx = truths["sigma_c"], truths["sigma_x"]

    keys = jr.split(key, 8)
    z_pec = jr.normal(keys[0], (n,)) * (SIGMA_PEC / _C_KMS)
    z_hel = (1.0 + z_star) * (1.0 + z_pec) - 1.0
    z_obs = dist.TruncatedNormal(z_hel, SIGMA_Z_FACTOR * (1.0 + z_hel),
                                 low=0.0, high=1.2).sample(keys[1])
    E_s = dist.Exponential(1.0 / tau).sample(keys[2], (n,))
    x_s = x0 + jr.normal(keys[3], (n,)) * sx
    c_int = (c0 + ac * x_s) + jr.normal(keys[4], (n,)) * sc
    m0_post = m0_star + 10.0 * jnp.log10(1.0 + z_pec)
    m_true = m0_post + al * x_s + be * c_int + RB * E_s + jr.normal(keys[5], (n,)) * si
    c_true = c_int + E_s
    d_hat = dist.MultivariateNormal(jnp.stack([m_true, c_true, x_s], axis=-1),
                                    covariance_matrix=jnp.asarray(W)).sample(keys[6])
    d_hat = np.asarray(d_hat)
    z_obs = np.asarray(z_obs)
    obs = np.column_stack([d_hat[:, 0], d_hat[:, 1], d_hat[:, 2], z_obs])

    if sel_cuts is None:
        return obs, None
    m_cut, s_mcut, z_cut, s_zcut, a, b = sel_cuts
    sel_var = d_hat[:, 0] + a * d_hat[:, 2] + b * d_hat[:, 1]
    p_s = ndtr((m_cut - sel_var) / s_mcut)
    p_z = ndtr((z_cut - z_obs) / s_zcut)
    us = np.asarray(jr.uniform(keys[7], (n,)))
    uz = np.asarray(jr.uniform(jr.fold_in(keys[7], 1), (n,)))
    return obs, (us < p_s) & (uz < p_z)


def q1_conditioning(z_star, truths):
    """19D q1 condition (training order) for a SN at latent redshift z_star."""
    mu_star = float(mu_func(jnp.array(z_star), truths["Om0"], truths["w0"]))
    m0_star = M0_FID + mu_star
    W, sig_m, sig_c, sig_x, cov_m_c, cov_m_x, cov_c_x = representative_errors(m0_star, truths)
    log_sigma_z = float(np.log(SIGMA_Z_FACTOR * (1.0 + z_star)))
    cond = jnp.array([
        m0_star, truths["c0_int"], truths["x0"], z_star,
        np.log(sig_m), np.log(sig_c), np.log(sig_x),
        cov_m_c, cov_m_x, cov_c_x, log_sigma_z,
        truths["alpha"], truths["beta_int"], truths["alpha_c"],
        truths["RB"], truths["tau"],
        truths["sigma_int"], truths["sigma_c"], truths["sigma_x"],
    ])
    return cond, W, m0_star, mu_star


def figure_q1_true(truths, z_lo, z_hi, n, n_sim, key, out):
    flow, mu_q1, std_q1, _ = fl.load_flow("q1_highz", fl.build_q1)
    Om0, w0 = truths["Om0"], truths["w0"]
    c0, x0 = truths["c0_int"], truths["x0"]
    z_mid = 0.5 * (z_lo + z_hi)
    cond_mid, W, _, _ = q1_conditioning(z_mid, truths)
    cond_mid = np.asarray(cond_mid)

    kzf, k_flow, kzs, k_sim = jr.split(key, 4)

    # FlowSN2: one draw per SN, conditioned on its own latent (z, m0)
    zf = np.asarray(jr.uniform(kzf, (n,), minval=z_lo, maxval=z_hi))
    m0f = mu_curve(zf, Om0, w0) + M0_FID
    cond_f = np.tile(cond_mid, (n, 1))
    cond_f[:, 0] = m0f
    cond_f[:, 3] = zf
    cond_f[:, 10] = np.log(SIGMA_Z_FACTOR * (1.0 + zf))
    flow_res = np.asarray(sample_flow_per_cond(flow, mu_q1, std_q1, 4, cond_f, k_flow))
    offset = np.column_stack([m0f, np.full(n, c0), np.full(n, x0), zf])
    flow_obs = finite_rows(flow_res + offset)

    # one large simulation over the same z range: all SNe, and its selected subset
    zs = np.asarray(jr.uniform(kzs, (n_sim,), minval=z_lo, maxval=z_hi))
    m0s = mu_curve(zs, Om0, w0) + M0_FID
    obs_all, mask = simulate_observed_at_latents(zs, m0s, W, truths, k_sim, n_sim, HIGHZ_CUTS)
    fin = np.isfinite(obs_all).all(axis=1)
    obs_all, mask = obs_all[fin], mask[fin]
    true_obs = obs_all[mask]
    print(f"  q1_true: {mask.sum():,} / {len(mask):,} selected in {z_lo}<z<{z_hi}")

    rng = np.random.default_rng(0)
    m = min(len(flow_obs), len(true_obs), len(obs_all), 40000)
    flow_p = subsample(flow_obs, m, rng)
    true_p = subsample(true_obs, m, rng)
    und_p = subsample(obs_all, m, rng)

    both = np.vstack([flow_p, true_p, und_p])
    ranges = [(np.percentile(both[:, i], 0.5), np.percentile(both[:, i], 99.5))
              for i in range(4)]
    labels = [r"$\hat{m}_s$", r"$\hat{c}_s$", r"$\hat{x}_s$", r"$\hat{z}_s$"]

    ckw = dict(labels=labels, range=ranges, bins=20, smooth=1.0, smooth1d=3,
               hist_bin_factor=5.0, max_n_ticks=3, reverse=True,
               plot_datapoints=False, plot_density=False,
               fill_contours=False, plot_contours=True, no_fill_contours=True,
               levels=LEVELS, label_kwargs={"fontsize": 22})
    fig = corner.corner(und_p, color=C_POP, contour_kwargs={"linewidths": 1.7},
                        hist_kwargs={"lw": 1.9, "color": C_POP}, **ckw)
    corner.corner(true_p, fig=fig, color=C_TRUE, contour_kwargs={"linewidths": 4.5},
                  hist_kwargs={"lw": 4.5, "color": C_TRUE}, **ckw)
    corner.corner(flow_p, fig=fig, color=C_FLOW, contour_kwargs={"linewidths": 1.7},
                  hist_kwargs={"lw": 1.9, "color": C_FLOW}, **ckw)
    fit_marginal_ylims(fig, 4)

    handles = [Line2D([], [], color=C_FLOW, lw=4, label=r"$\mathit{FlowSN2}$"),
               Line2D([], [], color=C_TRUE, lw=4, label="Selected SNe"),
               Line2D([], [], color=C_POP, lw=4, label="All SNe")]
    fig.legend(handles=handles, loc="lower left", frameon=False, fontsize=21,
               bbox_to_anchor=(0.12, 0.14))
    fig.text(0.14, 0.42, rf"${z_lo:g} < z_s < {z_hi:g}$", ha="left", va="top", fontsize=26)
    for ax in fig.get_axes():
        ax.tick_params(labelsize=15)

    fig.savefig(out, bbox_inches="tight", facecolor="white", edgecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 1 -- survey simulations
# ---------------------------------------------------------------------------
def _draw_pop_params(key, n):
    """Per-SN population parameters from the training ranges."""
    ks = jr.split(key, 10)
    u = lambda k, lo, hi: jr.uniform(k, (n,), minval=lo, maxval=hi)
    return dict(alpha=u(ks[0], -0.2, 0.0), beta_int=u(ks[1], 1.0, 3.0),
                alpha_c=u(ks[2], -0.02, 0.0), x0=u(ks[3], -0.6, -0.2),
                c0_int=u(ks[4], -0.1, 0.1), RB=u(ks[5], 2.2, 5.0),
                tau=u(ks[6], 0.0, 0.15), sigma_int=u(ks[7], 0.0, 0.2),
                sigma_c=u(ks[8], 0.0, 0.1), sigma_x=u(ks[9], 0.0, 1.5))


def _simulate_survey(key, n, survey, model_kwargs):
    pred = Predictive(sample_model, num_samples=1)(
        jr.fold_in(key, 2), n,
        m_cut=survey["m_cut"], sigma_mcut=survey["sigma_mcut"],
        z_cut=survey["z_cut"], sigma_zcut=survey["sigma_zcut"],
        lowz=survey["lowz"], sigma_z_factor=survey["sigma_z_factor"],
        sigma_pec=300, **model_kwargs)
    z = np.asarray(pred["z_obs"][0])
    m = np.asarray(pred["d_hat_s"][0][:, 0])
    sel = (np.asarray(pred["sel_s"][0]) == 1) & (np.asarray(pred["sel_z"][0]) == 1)
    return z, m, sel


def figure_survey_examples(truths, n_sim, key, out):
    kh, kl, kc = jr.split(key, 3)
    ia_kwargs = lambda k: {**_draw_pop_params(jr.fold_in(k, 1), n_sim),
                           "delta_mu_sigma": 0.25, "gamma": 0.0}
    zh, mh, selh = _simulate_survey(kh, n_sim, HIGHZ, ia_kwargs(kh))
    zl, ml, sell = _simulate_survey(kl, n_sim, LOWZ, ia_kwargs(kl))
    contam_kwargs = dict(alpha=0.0, beta_int=0.0, alpha_c=0.0, gamma=0.0,
                         sigma_int=1.2, M0=truths["M0"] + 0.4, beta_rate=3.0)
    zc, mc, selc = _simulate_survey(kc, n_sim, HIGHZ, contam_kwargs)

    rng = np.random.default_rng(0)
    # all objects: the unselected mix at the 35% pre-selection contaminant fraction
    n_ia_und = int(round(0.65 * 40000))
    und = np.vstack([subsample(np.column_stack([zh, mh]), n_ia_und, rng),
                     subsample(np.column_stack([zc, mc]), 40000 - n_ia_und, rng)])
    N_SEL = 16000
    hz = subsample(np.column_stack([zh[selh], mh[selh]]), N_SEL, rng)
    lz = subsample(np.column_stack([zl[sell], ml[sell]]), N_SEL, rng)
    co = subsample(np.column_stack([zc[selc], mc[selc]]), N_SEL // 4, rng)

    z_hi, m_lo, m_hi = 1.2, 13.5, 27.0

    left, width, bottom, height, hs = 0.10, 0.70, 0.08, 0.78, 0.11
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_axes([left, bottom, width, height])
    ax_hx = fig.add_axes([left, bottom + height, width, hs], sharex=ax)
    ax_hy = fig.add_axes([left + width, bottom, hs, height], sharey=ax)

    Om0, w0, M0 = truths["Om0"], truths["w0"], truths["M0"]

    ax.scatter(und[:, 0], und[:, 1], s=4, color=C_POP, alpha=0.4, linewidths=0, rasterized=True)
    ax.scatter(lz[:, 0], lz[:, 1], s=5, color=C_LOWZ, alpha=0.5, linewidths=0, rasterized=True)
    ax.scatter(hz[:, 0], hz[:, 1], s=5, color=C_HIGHZ, alpha=0.5, linewidths=0, rasterized=True)
    ax.scatter(co[:, 0], co[:, 1], s=12, color=C_CONTAM, alpha=0.35, marker="x",
               linewidths=1.0, rasterized=False, zorder=9)
    z_line = np.linspace(0.01, z_hi, 400)
    ax.plot(z_line, mu_curve(z_line, Om0, w0) + M0, color="black", ls="-", lw=2.0, zorder=10)
    ax.set_xlim(0, z_hi)
    ax.set_ylim(m_lo, m_hi)
    ax.set_xlabel(r"$\hat{z}_s$", fontsize=25)
    ax.set_ylabel(r"$\hat{m}_s$", fontsize=25)

    xb = np.linspace(0, z_hi, 80)
    yb = np.linspace(m_lo, m_hi, 80)
    for (zdat, mdat), col in [((und[:, 0], und[:, 1]), C_POP),
                              ((co[:, 0], co[:, 1]), C_CONTAM),
                              ((lz[:, 0], lz[:, 1]), C_LOWZ),
                              ((hz[:, 0], hz[:, 1]), C_HIGHZ)]:
        cx, hx = smooth_hist(zdat, xb, sigma=2.5, peak=True)
        ax_hx.plot(cx, hx, color=col, lw=2.1)
        cy, hy = smooth_hist(mdat, yb, sigma=2.5, peak=True)
        ax_hy.plot(hy, cy, color=col, lw=2.1)

    zz = np.linspace(0, z_hi, 400)
    for s in (LOWZ, HIGHZ):
        ax_hx.plot(zz, ndtr((s["z_cut"] - zz) / s["sigma_zcut"]), color="black", ls="--",
                   lw=1.8, zorder=10)
    mm = np.linspace(m_lo, m_hi, 400)
    for s in (LOWZ, HIGHZ):
        ax_hy.plot(ndtr((s["m_cut"] - mm) / s["sigma_mcut"]), mm, color="black", ls="--",
                   lw=1.8, zorder=10)
    ax_hx.set_ylim(0, 1.12)
    ax_hy.set_xlim(0, 1.12)

    ax_hx.tick_params(labelbottom=False)
    ax_hy.tick_params(labelleft=False)
    for a in (ax, ax_hx, ax_hy):
        a.tick_params(labelsize=18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("top", "right", "left"):
        ax_hx.spines[s].set_visible(False)
    for s in ("top", "right", "bottom"):
        ax_hy.spines[s].set_visible(False)
    ax_hx.yaxis.set_visible(False)
    ax_hy.xaxis.set_visible(False)

    ms = dict(markersize=9, markeredgewidth=2)
    handles = [Line2D([], [], color=C_HIGHZ, marker="o", ls="", label=r"Selected Photometric SNe Ia", **ms),
               Line2D([], [], color=C_LOWZ, marker="o", ls="", label=r"Selected Spectroscopic SNe Ia", **ms),
               Line2D([], [], color=C_CONTAM, marker="x", ls="", label="Selected Contaminants", **ms),
               Line2D([], [], color=C_POP, marker="o", ls="", label="All Objects", **ms),
               Line2D([], [], color="black", ls="-",
                      label=r"$\mu(\hat{z}_s;\,\mathbf{C}) + M_0^{\mathrm{int}}$"),
               Line2D([], [], color="black", ls="--", label="Selection Functions")]
    ax.legend(handles=handles, frameon=False, fontsize=17, loc="lower right")

    fig.savefig(out, facecolor="white", edgecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cosmo", default="cosmo1")
    parser.add_argument("--z-true-lo", type=float, default=1.13)
    parser.add_argument("--z-true-hi", type=float, default=1.17)
    parser.add_argument("--n", type=int, default=200000, help="flow samples")
    parser.add_argument("--n-true-sim", type=int, default=5000000,
                        help="simulation size for the selected comparison (selection is rare)")
    parser.add_argument("--n-survey", type=int, default=300000,
                        help="simulation size per survey for Figure 1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", default="all", choices=["all", "q1_true", "survey"])
    args = parser.parse_args()

    truths = resolve_truths(args.cosmo)
    # four keys as in the paper's run (the first two drew figures not in the paper)
    _, _, k3, k4 = jr.split(jr.key(args.seed), 4)
    FIGURES.mkdir(parents=True, exist_ok=True)
    if args.only in ("all", "q1_true"):
        figure_q1_true(truths, args.z_true_lo, args.z_true_hi, args.n, args.n_true_sim,
                       k3, str(FIGURES / f"q1_true_{args.cosmo}.pdf"))
    if args.only in ("all", "survey"):
        figure_survey_examples(truths, args.n_survey, k4, str(FIGURES / "survey_sims.pdf"))


if __name__ == "__main__":
    main()
