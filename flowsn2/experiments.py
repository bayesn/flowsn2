"""Experiment configuration (Section 3): surveys, the five test cosmologies,
truths, reproducible PRNG keys, the mock CMB / BAO observations and the
pure-Ia test catalogues.

Cosmology tags
--------------
  cosmo1 .. cosmo5   C1 .. C5 of the paper. cosmo1-3: wCDM (wa = 0, not sampled,
                     wcosmo distances). cosmo4/5: w0waCDM (wa sampled, CPL engine).
  <tag>_cmb          + CMB distance prior (R, l_A)
  <tag>_cmb_bao      + CMB and BAO priors
  cosmo1_dustnull    model comparison: host-mass dependent dust, gamma = 0
  cosmo1_dust        model comparison: host-mass dependent dust, gamma = -0.06
Modifiers share the base cosmology's truths AND its simulated SNe: cosmoN and
cosmoN_cmb analyse the identical catalogue.
"""
import zlib

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
from numpyro.infer import Predictive

from . import bao
from . import cosmology as cw
from .simulator import sample_model

# Section 3.1.1. n_sne is the number simulated BEFORE selection.
SURVEYS = [
    dict(id=0, n_sne=5000, m_cut=24.5, sigma_mcut=0.5,      # high-z photometric
         z_cut=0.9, sigma_zcut=0.1, lowz=False, sigma_z_factor=0.03),
    dict(id=1, n_sne=5000, m_cut=20.0, sigma_mcut=0.25,     # low-z spectroscopic
         z_cut=0.1, sigma_zcut=0.05, lowz=True,  sigma_z_factor=1e-4),
]
ZMAX_HIGHZ, ZMAX_LOWZ = 1.2, 0.3

PARAM_NAMES = [
    "M0", "Om0", "RB", "alpha", "alpha_c", "beta_int",
    "beta_rate", "c0_int", "sigma_c", "sigma_int",
    "sigma_x", "tau", "w0", "wa", "x0", "gamma",
    "tau_hi", "RB_hi",          # only sampled in the dust-step model
]

# SN Ia truths in the SAMPLING convention. The tables and figures report gamma
# with the opposite sign (flowsn2.plotting.SIGN_FLIP), i.e. gamma = -0.1.
TRUTHS_FIDUCIAL = {
    "M0": -19.380, "Om0": 0.315, "RB": 3.740, "alpha": -0.149,
    "alpha_c": -0.006, "beta_int": 2.322, "beta_rate": 1.5,
    "c0_int": -0.056, "sigma_c": 0.071, "sigma_int": 0.101,
    "sigma_x": 1.137, "tau": 0.068, "w0": -1.0, "wa": 0.0, "x0": -0.452, "gamma": 0.1,
    "tau_hi": 0.068, "RB_hi": 3.740,     # no dust step unless a _dust tag overrides
}

# Model comparison, dust vs intrinsic mass step (Section 3.4.1): high-mass hosts
# have larger tau and lower R_B; the extended simulation adds gamma = -0.06.
DUST_STEP_TRUTHS = {"tau_hi": 0.098, "RB_hi": 3.00}
GAMMA_ALT = -0.06

COSMOLOGIES = {
    "cosmo1": dict(w0=-1.0, Om0=0.315, beta_rate=1.5, wa=0.0),     # C1, fiducial
    "cosmo2": dict(w0=-0.8, Om0=0.35,  beta_rate=2.0, wa=0.0),     # C2
    "cosmo3": dict(w0=-1.2, Om0=0.25,  beta_rate=1.0, wa=0.0),     # C3
    "cosmo4": dict(w0=-1.0, Om0=0.315, beta_rate=1.5, wa=0.0),     # C4 (wa inferred)
    "cosmo5": dict(w0=-0.752, Om0=0.316, beta_rate=1.5, wa=-0.86),  # C5, DESI DR2
}
W0WA_COSMOS = {"cosmo4", "cosmo5"}

# Planck 2018 distance-prior precisions (Appendix D.1)
CMB_SIGMA_R, CMB_SIGMA_LA, CMB_RHO = 0.0046, 0.09, 0.46


def base_cosmo(cosmo):
    """Strip the _cmb / _bao / _dust / _dustnull modifiers."""
    for suf in ("_cmb", "_bao", "_dustnull", "_dust"):   # _dustnull BEFORE _dust
        cosmo = cosmo.replace(suf, "")
    return cosmo


def is_cmb(cosmo):
    return "_cmb" in cosmo


def is_bao(cosmo):
    return "_bao" in cosmo


def is_w0wa(cosmo):
    return base_cosmo(cosmo) in W0WA_COSMOS


def is_dust_step(cosmo):
    return "_dust" in cosmo


def resolve_truths(cosmo="cosmo1"):
    """Truth dict for a cosmology tag (sampling convention)."""
    truths = dict(TRUTHS_FIDUCIAL)
    truths.update(COSMOLOGIES[base_cosmo(cosmo)])
    if "_dust" in cosmo:
        truths.update(DUST_STEP_TRUTHS)
        truths["gamma"] = 0.0 if "_dustnull" in cosmo else GAMMA_ALT
    return truths


# Per-cosmology PRNG block: each cosmology draws an independent realisation at
# the same seed; modifiers share their base cosmology's block (same SNe).
COSMO_SEED_BLOCK = {"cosmo1": 0, "cosmo2": 1, "cosmo3": 2, "cosmo4": 3, "cosmo5": 4}
_SEED_STRIDE = 1_000_000


def simulation_key(cosmo="cosmo1", seed=0):
    """PRNGKey of the pure-Ia catalogue (and of the CMB/BAO observations, which
    are folded off it) for (cosmology, seed)."""
    name = base_cosmo(cosmo)
    if name in COSMO_SEED_BLOCK:
        block = COSMO_SEED_BLOCK[name]
    else:
        block = 3 + zlib.adler32(name.encode("utf8")) % 100
    return random.PRNGKey(int(seed) + block * _SEED_STRIDE)


# Fixed MCMC starting point: hand-set literals near the fiducial scales, never
# derived from the truths. For _cmb/_bao runs Om0/w0/wa are replaced by the MAP
# of the CMB/BAO observations (cosmo_init_from_priors).
INIT_FIXED = {
    "M0": -19.3,  "Om0": 0.30,   "RB": 3.6,      "alpha": -0.14,
    "alpha_c": -0.005, "beta_int": 2.4, "beta_rate": 1.7, "c0_int": -0.05,
    "sigma_c": 0.08, "sigma_int": 0.12, "sigma_x": 1.1,  "tau": 0.07,
    "w0": -0.9,   "wa": 0.2,     "x0": -0.43,    "gamma": 0.12,
    "tau_hi": 0.07, "RB_hi": 3.6,
}


def external_observations(cosmo, seed):
    """Mock CMB (R, l_A) and BAO data for (cosmology, seed), drawn around the
    truth from keys folded off the simulation key (777 CMB, 999 BAO). Identical
    for every engine and every resample of a seed.
    Returns (cmb, bao_obs, cmb_cov): cmb/bao_obs are None when not requested."""
    truths = resolve_truths(cosmo)
    rng_key = simulation_key(cosmo, seed)
    cmb = bao_obs = cmb_cov = None
    if is_cmb(cosmo):
        R_true = float(cw.cmb_shift_R(truths["Om0"], truths["w0"], truths["wa"], cw.H0))
        lA_true = float(cw.cmb_acoustic_lA(truths["Om0"], truths["w0"], truths["wa"], cw.H0))
        off = CMB_RHO * CMB_SIGMA_R * CMB_SIGMA_LA
        cmb_cov = np.array([[CMB_SIGMA_R ** 2, off], [off, CMB_SIGMA_LA ** 2]])
        draw = random.multivariate_normal(random.fold_in(rng_key, 777),
                                          np.array([R_true, lA_true]), cmb_cov)
        cmb = dict(R_obs=float(draw[0]), lA_obs=float(draw[1]),
                   sigma_R=CMB_SIGMA_R, sigma_lA=CMB_SIGMA_LA, rho=CMB_RHO)
        print(f"CMB: R_true={R_true:.5f} R_obs={cmb['R_obs']:.5f} | "
              f"lA_true={lA_true:.3f} lA_obs={cmb['lA_obs']:.3f}")
    if is_bao(cosmo):
        bao_true = np.asarray(bao.model_vector(truths["Om0"], truths["w0"], truths["wa"]))
        cov = bao.covariance(bao_true)
        data = np.asarray(random.multivariate_normal(random.fold_in(rng_key, 999),
                                                     bao_true, cov))
        bao_obs = dict(data=data, cov=cov)
        print(f"BAO: {bao_true.size} DESI-like points")
    return cmb, bao_obs, cmb_cov


def cosmo_init_from_priors(init0, w0wa, R_obs=None, lA_obs=None, cmb_cov=None,
                           bao_obs=None, bao_cov=None, H0=70.0):
    """(Om0, w0, wa) minimising the CMB/BAO chi^2 of the mock observations: a
    feasible start on the stiff CMB/BAO manifold that uses no truth information."""
    from scipy.optimize import minimize

    cmb_P = jnp.asarray(np.linalg.inv(cmb_cov)) if R_obs is not None else None
    bao_P = jnp.asarray(np.linalg.inv(bao_cov)) if bao_obs is not None else None
    baob = jnp.asarray(bao_obs) if bao_obs is not None else None

    def chi2(theta):
        Om0, w0 = theta[0], theta[1]
        wa = theta[2] if w0wa else 0.0
        c = 0.0
        if cmb_P is not None:
            d = jnp.array([cw.cmb_shift_R(Om0, w0, wa, H0) - R_obs,
                           cw.cmb_acoustic_lA(Om0, w0, wa, H0) - lA_obs])
            c = c + d @ cmb_P @ d
        if bao_P is not None:
            d = bao.model_vector(Om0, w0, wa, H0) - baob
            c = c + d @ bao_P @ d
        return c

    vg = jax.jit(jax.value_and_grad(chi2))

    def f(t):
        v, g = vg(jnp.asarray(t))
        return float(v), np.asarray(g, dtype=float)

    x0 = [init0["Om0"], init0["w0"]] + ([init0.get("wa", 0.0)] if w0wa else [])
    bnds = [(0.05, 0.95), (-1.95, -0.05)] + ([(-2.9, 2.9)] if w0wa else [])
    res = minimize(f, np.array(x0, float), jac=True, method="L-BFGS-B", bounds=bnds)
    Om0, w0 = float(res.x[0]), float(res.x[1])
    wa = float(res.x[2]) if w0wa else 0.0
    return Om0, w0, wa


def initial_values(cosmo, cmb=None, bao_obs=None, cmb_cov=None):
    """NUTS init: INIT_FIXED, minus wa for wCDM, with the cosmology replaced by
    the CMB/BAO MAP when external priors are used."""
    w0wa = is_w0wa(cosmo)
    init = dict(INIT_FIXED)
    if not w0wa:
        init.pop("wa", None)
    if cmb is not None or bao_obs is not None:
        Om0_i, w0_i, wa_i = cosmo_init_from_priors(
            init, w0wa,
            R_obs=None if cmb is None else cmb["R_obs"],
            lA_obs=None if cmb is None else cmb["lA_obs"], cmb_cov=cmb_cov,
            bao_obs=None if bao_obs is None else bao_obs["data"],
            bao_cov=None if bao_obs is None else bao_obs["cov"])
        init["Om0"], init["w0"] = Om0_i, w0_i
        if "wa" in init:
            init["wa"] = wa_i
        print(f"cosmo init (CMB/BAO MAP): Om0={Om0_i:.4f} w0={w0_i:.4f} wa={wa_i:.4f}")
    return init


def survey_zmax(survey):
    return ZMAX_LOWZ if survey["lowz"] else ZMAX_HIGHZ


def simulate_survey(sim_key, survey, w0=-1.0, Om0=0.315, beta_rate=1.5, sigma_pec=300, wa=None):
    """Selected SNe of one survey (pure Ia). wa=None -> wCDM, float -> w0waCDM."""
    pp = Predictive(sample_model, num_samples=1)(
        sim_key, survey["n_sne"],
        m_cut=survey["m_cut"],     sigma_mcut=survey["sigma_mcut"],
        z_cut=survey["z_cut"],     sigma_zcut=survey["sigma_zcut"],
        lowz=survey["lowz"],       sigma_z_factor=survey["sigma_z_factor"],
        Om0=Om0, w0=w0, beta_rate=beta_rate, sigma_pec=sigma_pec, wa=wa,
    )
    sel = jnp.logical_and(pp["sel_s"][0] == 1, pp["sel_z"][0] == 1)
    W_s = pp["W_s"][0, sel]
    return dict(
        z_s=pp["z_obs"][0][sel],
        sigma_z=pp["sigma_z"][0][sel],
        data_s=pp["d_hat_s"][0][sel],
        data_err_s=jnp.array([
            W_s[:, 0, 0], W_s[:, 1, 1], W_s[:, 2, 2],
            W_s[:, 0, 1], W_s[:, 0, 2], W_s[:, 1, 2],
        ]).T,
        mass=pp["mass"][0][sel],
        m_cut=survey["m_cut"],     sigma_mcut=survey["sigma_mcut"],
        z_cut=survey["z_cut"],     sigma_zcut=survey["sigma_zcut"],
        zmax=survey_zmax(survey),
        s_id=survey["id"], lowz=survey["lowz"],
    )


def simulate_all(rng_key, w0=-1.0, Om0=0.315, beta_rate=1.5, sigma_pec=300, surveys=SURVEYS, wa=None):
    """Split rng_key into one key per survey plus the MCMC key.
    Returns (grouped_data, mcmc_key)."""
    keys = random.split(rng_key, len(surveys) + 1)
    survey_keys, mcmc_key = keys[:-1], keys[-1]
    grouped = [simulate_survey(k, s, w0=w0, Om0=Om0, beta_rate=beta_rate, sigma_pec=sigma_pec, wa=wa)
               for k, s in zip(survey_keys, surveys)]
    n_sne = sum(len(s["z_s"]) for s in grouped)
    wa_str = "" if wa is None else f", wa={wa}"
    print(f"Simulated (w0={w0}, Om0={Om0}, beta_rate={beta_rate}{wa_str}): {n_sne} SNe")
    return grouped, mcmc_key


def pure_ia_catalogue(cosmo, seed, sigma_pec=300):
    """The pure-Ia test catalogue for (cosmology, seed) and its MCMC key."""
    truths = resolve_truths(cosmo)
    wa = truths["wa"] if is_w0wa(cosmo) else None
    return simulate_all(simulation_key(cosmo, seed), w0=truths["w0"], Om0=truths["Om0"],
                        beta_rate=truths["beta_rate"], sigma_pec=sigma_pec, wa=wa)
