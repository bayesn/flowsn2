"""Numpyro model with the exact analytical likelihood (the comparison baseline).

``build_survey_likelihood`` is also used by the mock classifier
(flowsn2.contamination), so the class-conditional densities that define the
classification scores are built with exactly the machinery used in the fits.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro

from . import cosmology as cw
from .cosmology import H0, mu_func
from .redshifts import pdf_unormalised, pdf_unormalised_w0wa
from .priors import sample_global_params
from .analytical_likelihood import AnalyticalLikelihood


def build_survey_likelihood(survey, params, sigma_pec=300, w0wa=False, M_split=10,
                            a=-0.1, b=-1):
    """Per-survey AnalyticalLikelihood and the stacked (m, c, x, z) data.

    survey : dict with z_s, sigma_z, data_s, data_err_s, mass, m_cut, sigma_mcut,
             z_cut, sigma_zcut, zmax.
    params : global parameters (sampled values or truths). tau_hi / RB_hi, when
             present and not None, give high-mass hosts their own dust.
    Returns (likelihood, data, n_sne).
    """
    z_s, sigma_z = survey["z_s"], survey["sigma_z"]
    data_s, data_err_s, mass = survey["data_s"], survey["data_err_s"], survey["mass"]
    zmax = survey["zmax"]
    Om0, w0, wa = params["Om0"], params["w0"], params.get("wa", 0.0)
    M0, beta_rate, gamma = params["M0"], params["beta_rate"], params["gamma"]
    tau, RB = params["tau"], params["RB"]
    tau_hi, RB_hi = params.get("tau_hi"), params.get("RB_hi")

    data = jnp.append(data_s, z_s[..., None], axis=1)
    mass_mask = jnp.where(mass > M_split, 1, 0)
    n_sne = len(z_s)

    def mu_one(z):
        if w0wa:
            return cw.distance_modulus(z, Om0, w0, wa, H0)
        return mu_func(z, Om0, w0)

    def mu_vmap(z):
        return jax.vmap(mu_one, in_axes=(0))(z)

    # Unnormalised redshift-rate density (wcosmo wCDM, or CPL engine for w0wa)
    def rate_unorm(zz):
        if w0wa:
            return pdf_unormalised_w0wa(zz, Om0, w0, wa, beta_rate)
        return pdf_unormalised(zz, Om0, w0, beta_rate)

    # Denominator: global grid over the full redshift range
    N_dg     = 100
    z_dg     = jnp.linspace(0.01, zmax, N_dg)
    mu_dg    = mu_vmap(z_dg)                                           # (N_dg,)
    log_P_dg = jnp.log(rate_unorm(z_dg))                               # (N_dg,)
    log_w_dg = log_P_dg + jnp.log(z_dg[1] - z_dg[0])                   # (N_dg,)

    # Numerator: per-SN local grid centred on z_obs, width set by its uncertainty
    N_num     = 25
    sigma_z_s = jnp.broadcast_to(jnp.atleast_1d(jnp.array(sigma_z)), (n_sne,))
    sigma_z_total = jnp.sqrt(sigma_z_s**2 + (sigma_pec / 299792.458)**2)
    z_min     = jnp.maximum(0.01, z_s - 5 * sigma_z_total)             # (n_sne,)
    z_max     = jnp.minimum(zmax,  z_s + 5 * sigma_z_total)            # (n_sne,)
    t         = jnp.linspace(0, 1, N_num)                              # (N_num,)
    z_num     = z_min[:, None] + t[None, :] * (z_max - z_min)[:, None]  # (n_sne, N_num)
    dz_num    = (z_max - z_min) / (N_num - 1)                          # (n_sne,)
    mu_num    = mu_vmap(z_num)                                         # (n_sne, N_num)
    log_P_num = jnp.log(rate_unorm(z_num.ravel())).reshape(z_num.shape)
    log_w_num = log_P_num + jnp.log(dz_num[:, None])                   # (n_sne, N_num)

    likelihood = AnalyticalLikelihood(
        z_num=z_num, m0_num=mu_num + M0, log_w_num=log_w_num,
        z_dg=z_dg,   m0_dg=mu_dg + M0,   log_w_dg=log_w_dg,
        c0_int=jnp.array([params["c0_int"]]), x0=jnp.array([params["x0"]]),
        sigma_int=jnp.array([params["sigma_int"]]), sigma_c=jnp.array([params["sigma_c"]]),
        sigma_x=jnp.array([params["sigma_x"]]),
        alpha=jnp.array([params["alpha"]]), beta_int=jnp.array([params["beta_int"]]),
        alpha_c=jnp.array([params["alpha_c"]]),
        # per-SN tau/RB when the host-mass dust step is on (None -> global)
        RB=(jnp.array([RB]) if RB_hi is None else jnp.where(mass_mask == 1, RB_hi, RB)),
        tau=(jnp.array([tau]) if tau_hi is None else jnp.where(mass_mask == 1, tau_hi, tau)),
        gamma=gamma * mass_mask,
        W_mm=data_err_s[:, 0], W_cc=data_err_s[:, 1], W_xx=data_err_s[:, 2],
        W_mc=data_err_s[:, 3], W_mx=data_err_s[:, 4], W_cx=data_err_s[:, 5],
        sigma_z=sigma_z_s, sigma_pec=jnp.array(sigma_pec),
        m_cut=jnp.array([survey["m_cut"]]), sigma_mcut=jnp.array([survey["sigma_mcut"]]),
        z_cut=jnp.array([survey["z_cut"]]), sigma_zcut=jnp.array([survey["sigma_zcut"]]),
        a=jnp.array([a]), b=jnp.array([b]))

    return likelihood, data, n_sne


def mcmc_model(grouped_data, dust_step=False, a=-0.1, b=-1, sigma_pec=300, w0wa=False,
               cmb=None, bao_obs=None, M_split=10, include=None):
    """Joint posterior over all surveys with the analytical likelihood.

    include : optional list (one per survey) of per-SN 0/1 weights. Given, each
        survey's likelihood is the masked sum sum_i include_i * log L_i, so a
        Bayesian-resampling allocation changes only this vector (fixed shapes ->
        the model compiles once). None is the ordinary observed plate.
    """
    params = sample_global_params(w0wa=w0wa, dust_step=dust_step, cmb=cmb, bao_obs=bao_obs)

    for idx, survey in enumerate(grouped_data):
        likelihood, data, n_sne = build_survey_likelihood(
            survey, params, sigma_pec=sigma_pec, w0wa=w0wa, M_split=M_split, a=a, b=b)
        s_id = str(survey["s_id"])
        if include is None:
            with numpyro.plate("plate_" + s_id, n_sne):
                numpyro.sample("obs" + s_id, likelihood, obs=data)
        else:
            log_lik = likelihood.log_prob(data)                 # (n_sne,)
            numpyro.factor("obs" + s_id, jnp.sum(include[idx] * log_lik))
