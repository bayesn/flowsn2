"""Numpyro model with the FlowSN2 normalising-flow likelihood.

Same priors and survey structure as analytical_model.mcmc_model; selection
effects and the data likelihood come from the trained flows, so no analytical
Malmquist correction appears anywhere.

data_err_s columns: [W_mm, W_cc, W_xx, W_mc, W_mx, W_cx] (variances on the
diagonal). q1 was trained on log sigma_m, log sigma_c, log sigma_x and the raw
off-diagonals, hence the 0.5 * log conversion below.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro

from . import cosmology as cw
from .cosmology import H0, mu_func
from .redshifts import pdf_unormalised, pdf_unormalised_w0wa
from .priors import sample_global_params
from .flow_likelihood import FlowSurveyLikelihood


def _safe_log(x):
    return jnp.log(jnp.clip(jnp.abs(x), 1e-30))


def flow_mcmc_model(grouped_data, sigma_pec=300, w0wa=False, cmb=None, bao_obs=None,
                    M_split=10, delta_mu_sigma=0.25, include=None, chunk_size=None,
                    dust_step=False):
    """Joint posterior over all surveys with the FlowSN2 likelihood.

    grouped_data : list of survey dicts (z_s, sigma_z, data_s, data_err_s, mass,
                   zmax, s_id, lowz). include / chunk_size: see FlowSurveyLikelihood
                   and analytical_model.mcmc_model.
    """
    p = sample_global_params(w0wa=w0wa, dust_step=dust_step, cmb=cmb, bao_obs=bao_obs)

    for idx, data in enumerate(grouped_data):
        flow_mcmc_survey(
            data, p, sigma_pec=sigma_pec, w0wa=w0wa, M_split=M_split,
            delta_mu_sigma=delta_mu_sigma, chunk_size=chunk_size,
            include=(None if include is None else include[idx]),
        )


def flow_mcmc_survey(survey, p, sigma_pec=300, w0wa=False, M_split=10,
                     delta_mu_sigma=0.25, include=None, chunk_size=None):
    """Per-survey flow likelihood."""
    z_s, sigma_z = survey["z_s"], survey["sigma_z"]
    data_s, data_err_s, mass = survey["data_s"], survey["data_err_s"], survey["mass"]
    zmax, s_id = survey["zmax"], survey["s_id"]
    Om0, w0, wa, beta_rate = p["Om0"], p["w0"], p["wa"], p["beta_rate"]
    M0, gamma, tau, RB = p["M0"], p["gamma"], p["tau"], p["RB"]
    tau_hi, RB_hi = p["tau_hi"], p["RB_hi"]
    n_sne = len(z_s)

    data = jnp.append(data_s, z_s[..., None], axis=1)   # (n_sne, 4)
    mass_mask = jnp.where(mass > M_split, 1, 0)           # (n_sne,)

    def _mu_scalar(z):
        if w0wa:
            return cw.distance_modulus(z, Om0, w0, wa, H0)
        return mu_func(z, Om0, w0)

    mu_vmap_1d = jax.vmap(_mu_scalar)

    def _rate_unorm(zz):
        if w0wa:
            return pdf_unormalised_w0wa(zz, Om0, w0, wa, beta_rate)
        return pdf_unormalised(zz, Om0, w0, beta_rate)

    def mu_2d(z_2d):
        shape = z_2d.shape
        return mu_vmap_1d(z_2d.ravel()).reshape(shape)

    # Denominator grid (global, (N_dg,))
    N_dg  = 100
    z_dg  = jnp.linspace(0.01, zmax, N_dg)
    mu_dg = mu_vmap_1d(z_dg)                                  # (N_dg,)
    log_P_dg = jnp.log(_rate_unorm(z_dg))                      # (N_dg,)
    log_w_dg = log_P_dg + jnp.log(z_dg[1] - z_dg[0])           # (N_dg,)
    m0_dg = jnp.stack([mu_dg + M0, mu_dg + M0 + gamma], axis=0)  # (2, N_dg)

    # Numerator grid (per-SN local z window, (n_sne, N_num))
    N_num = 25
    sigma_z_s     = jnp.broadcast_to(jnp.atleast_1d(jnp.array(sigma_z)), (n_sne,))
    sigma_z_total = jnp.sqrt(sigma_z_s ** 2 + (sigma_pec / 299792.458) ** 2)
    z_min = jnp.maximum(0.01, z_s - 5 * sigma_z_total)
    z_max = jnp.minimum(zmax,  z_s + 5 * sigma_z_total)
    t     = jnp.linspace(0, 1, N_num)
    z_num = z_min[:, None] + t[None, :] * (z_max - z_min)[:, None]    # (n_sne, N_num)
    dz_num = (z_max - z_min) / (N_num - 1)                             # (n_sne,)
    mu_num = mu_2d(z_num)                                              # (n_sne, N_num)
    log_P_num = jnp.log(_rate_unorm(z_num.ravel())).reshape(z_num.shape)
    log_w_num = log_P_num + jnp.log(dz_num[:, None])                  # (n_sne, N_num)

    # m0_num = mu(z_j; C) + M0 + gamma * mask_i
    m0_num = mu_num + M0 + (gamma * mass_mask)[:, None]                # (n_sne, N_num)

    log_err_mm = 0.5 * _safe_log(data_err_s[:, 0])   # log sigma_m
    log_err_cc = 0.5 * _safe_log(data_err_s[:, 1])   # log sigma_c
    log_err_xx = 0.5 * _safe_log(data_err_s[:, 2])   # log sigma_x
    W_mc       = data_err_s[:, 3]                     # raw (can be negative)
    W_mx       = data_err_s[:, 4]
    W_cx       = data_err_s[:, 5]
    log_sigma_z = _safe_log(sigma_z_s)

    # host-mass dust step: per-SN tau/RB plus the (2,) bins for the denominator
    tau_s = tau if tau_hi is None else jnp.where(mass_mask == 1, tau_hi, tau)
    RB_s  = RB  if RB_hi  is None else jnp.where(mass_mask == 1, RB_hi,  RB)
    tau_bins = None if tau_hi is None else jnp.stack([tau, tau_hi])
    RB_bins  = None if RB_hi  is None else jnp.stack([RB,  RB_hi])

    likelihood = FlowSurveyLikelihood(
        z_num=z_num, m0_num=m0_num, log_w_num=log_w_num,
        z_dg=z_dg, m0_dg=m0_dg, log_w_dg=log_w_dg,
        c0_int=p["c0_int"], x0=p["x0"],
        sigma_int=p["sigma_int"], sigma_c=p["sigma_c"], sigma_x=p["sigma_x"],
        alpha=p["alpha"], beta_int=p["beta_int"], alpha_c=p["alpha_c"],
        RB=RB_s, tau=tau_s,
        log_err_mm=log_err_mm, log_err_cc=log_err_cc, log_err_xx=log_err_xx,
        W_mc=W_mc, W_mx=W_mx, W_cx=W_cx, log_sigma_z=log_sigma_z,
        delta_mu_sigma=delta_mu_sigma,
        lowz=survey.get("lowz", False),
        mass_mask=mass_mask,
        chunk_size=chunk_size,
        tau_bins=tau_bins, RB_bins=RB_bins,
    )

    if include is None:
        with numpyro.plate("plate_" + str(s_id), n_sne):
            numpyro.sample("obs" + str(s_id), likelihood, obs=data)
    else:
        log_lik = likelihood.log_prob(data)                 # (n_sne,)
        numpyro.factor("obs" + str(s_id), jnp.sum(include * log_lik))
