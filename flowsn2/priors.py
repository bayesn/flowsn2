"""Global-parameter priors (Table C1) and the CMB / BAO likelihood terms
(Appendix D), shared by the analytical and FlowSN2 numpyro models so both
sample exactly the same prior."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from . import cosmology as cw
from . import bao
from .cosmology import H0

# name -> (family, args). Uniform: (lo, hi); TruncatedNormal: (loc, scale, lo, hi).
PRIORS = {
    "Om0":       ("Uniform", (0, 1)),
    "w0":        ("Uniform", (-2., 0.)),
    "wa":        ("Uniform", (-3.0, 3.0)),
    "beta_rate": ("Uniform", (0., 5.)),
    "alpha":     ("Uniform", (-0.2, 0.0)),
    "beta_int":  ("Uniform", (1.0, 3.0)),
    "alpha_c":   ("Uniform", (-0.02, 0.0)),
    "gamma":     ("Uniform", (-0.5, 0.5)),
    "tau":       ("Uniform", (0.0, 0.15)),
    "RB":        ("Uniform", (2.2, 5.0)),
    "tau_hi":    ("Uniform", (0.0, 0.15)),     # dust-step model only
    "RB_hi":     ("Uniform", (2.2, 5.0)),      # dust-step model only
    "M0":        ("Uniform", (-20.5, -18.5)),
    "c0_int":    ("Uniform", (-0.1, 0.1)),
    "x0":        ("Uniform", (-0.6, -0.2)),
    "sigma_int": ("TruncatedNormal", (0.0, 1.0, 0.0, 0.2)),
    "sigma_c":   ("TruncatedNormal", (0.0, 1.0, 0.0, 0.1)),
    "sigma_x":   ("TruncatedNormal", (0.0, 2.0, 0.0, 1.5)),
}


def prior(name):
    family, args = PRIORS[name]
    if family == "Uniform":
        return dist.Uniform(*args)
    loc, scale, low, high = args
    return dist.TruncatedNormal(loc, scale, low=low, high=high)


def prior_bounds(name):
    """(lo, hi) support of a parameter's prior."""
    family, args = PRIORS[name]
    return (args[0], args[1]) if family == "Uniform" else (args[2], args[3])


def sample_global_params(w0wa=False, dust_step=False, cmb=None, bao_obs=None):
    """Sample every global parameter (and add the optional CMB / BAO terms).

    w0wa      : infer wa (C4, C5); otherwise wa = 0 (wCDM).
    dust_step : sample high-mass tau_hi / RB_hi (dust-vs-intrinsic test).
    cmb       : None or dict(R_obs, lA_obs, sigma_R, sigma_lA, rho) -- Eq. D2.
    bao_obs   : None or dict(data, cov) -- Eq. D4.
    Returns a dict of the sampled values (tau_hi / RB_hi are None without the step).
    """
    Om0 = numpyro.sample("Om0", prior("Om0"))
    w0 = numpyro.sample("w0", prior("w0"))

    wa = numpyro.sample("wa", prior("wa")) if w0wa else 0.0

    if cmb is not None:
        R = cw.cmb_shift_R(Om0, w0, wa, H0)
        lA = cw.cmb_acoustic_lA(Om0, w0, wa, H0)
        cov_off = cmb["rho"] * cmb["sigma_R"] * cmb["sigma_lA"]
        cmb_cov = jnp.array([[cmb["sigma_R"] ** 2, cov_off],
                             [cov_off, cmb["sigma_lA"] ** 2]])
        numpyro.sample("cmb_obs",
                       dist.MultivariateNormal(jnp.stack([R, lA]), covariance_matrix=cmb_cov),
                       obs=jnp.array([cmb["R_obs"], cmb["lA_obs"]]))

    if bao_obs is not None:
        bao_model = bao.model_vector(Om0, w0, wa, H0)
        numpyro.sample("bao_obs",
                       dist.MultivariateNormal(bao_model, covariance_matrix=bao_obs["cov"]),
                       obs=bao_obs["data"])

    p = dict(Om0=Om0, w0=w0, wa=wa)
    p["beta_rate"] = numpyro.sample("beta_rate", prior("beta_rate"))
    p["alpha"] = numpyro.sample("alpha", prior("alpha"))
    p["beta_int"] = numpyro.sample("beta_int", prior("beta_int"))
    p["alpha_c"] = numpyro.sample("alpha_c", prior("alpha_c"))
    p["gamma"] = numpyro.sample("gamma", prior("gamma"))
    p["tau"] = numpyro.sample("tau", prior("tau"))
    p["RB"] = numpyro.sample("RB", prior("RB"))
    if dust_step:
        # high-mass ENDPOINTS with the same priors as tau / RB, so both mass bins
        # stay inside the flows' per-SN training ranges
        p["tau_hi"] = numpyro.sample("tau_hi", prior("tau_hi"))
        p["RB_hi"] = numpyro.sample("RB_hi", prior("RB_hi"))
        numpyro.deterministic("delta_tau", p["tau_hi"] - p["tau"])
        numpyro.deterministic("delta_RB", p["RB_hi"] - p["RB"])
    else:
        p["tau_hi"] = p["RB_hi"] = None
    p["M0"] = numpyro.sample("M0", prior("M0"))
    p["c0_int"] = numpyro.sample("c0_int", prior("c0_int"))
    p["x0"] = numpyro.sample("x0", prior("x0"))
    p["sigma_int"] = numpyro.sample("sigma_int", prior("sigma_int"))
    p["sigma_c"] = numpyro.sample("sigma_c", prior("sigma_c"))
    p["sigma_x"] = numpyro.sample("sigma_x", prior("sigma_x"))
    return p
