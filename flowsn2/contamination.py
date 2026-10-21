"""Contaminated catalogues and mock classification scores (Section 3.2.1).

1. Each high-z photometric object is a SN Ia with probability pi_Ia = 1 -
   contam_frac (pre-selection) and a non-Ia contaminant otherwise. Contaminants
   use the same forward model with alpha = beta_int = alpha_c = gamma = 0,
   sigma_int = 1.2, M0 + 0.4 mag and their own rate xi_rate = 3.0; their colour,
   stretch and dust populations equal the SN Ia ones (deliberately confusable).
   Selection is applied identically to both classes. The low-z spectroscopic
   survey is confirmed SNe Ia only.
2. Mock classification scores are the exact conditional class probabilities
       C_i = pi_sel f_Ia(d_i) / [pi_sel f_Ia(d_i) + (1 - pi_sel) f_nonIa(d_i)],
   with f the selection-conditional analytical densities at the true parameters
   and pi_sel the post-selection SN Ia fraction (Eq. 33).
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
from numpyro.infer import Predictive

from .simulator import sample_model
from .analytical_model import build_survey_likelihood
from .experiments import SURVEYS, resolve_truths, is_w0wa, survey_zmax

# PRNG root for contaminated catalogues, disjoint from experiments.simulation_key
_CONTAM_BLOCK = 900_000_000
# The contamination baseline is 1.35x the pure-Ia survey sizes (~2000 SNe Ia and
# ~500 contaminants selected); n_scale multiplies on top (scaling test).
CONTAM_N_MULT = 1.35
CONTAM_SIGMA_INT = 1.2

DEFAULTS = dict(contam_frac=0.35, mag_offset=0.4, contam_beta_rate=3.0)


def _simulate_population(sim_key, survey, overrides, w0, Om0, beta_rate, wa, sigma_pec):
    """Forward-model one survey; return the FULL pre-selection arrays plus the
    boolean selection mask."""
    pp = Predictive(sample_model, num_samples=1)(
        sim_key, survey["n_sne"],
        m_cut=survey["m_cut"],     sigma_mcut=survey["sigma_mcut"],
        z_cut=survey["z_cut"],     sigma_zcut=survey["sigma_zcut"],
        lowz=survey["lowz"],       sigma_z_factor=survey["sigma_z_factor"],
        Om0=Om0, w0=w0, beta_rate=beta_rate, sigma_pec=sigma_pec, wa=wa,
        **overrides,
    )
    sel = jnp.logical_and(pp["sel_s"][0] == 1, pp["sel_z"][0] == 1)
    W = pp["W_s"][0]
    data_err = jnp.stack([
        W[:, 0, 0], W[:, 1, 1], W[:, 2, 2],
        W[:, 0, 1], W[:, 0, 2], W[:, 1, 2],
    ], axis=1)
    return dict(z_s=pp["z_obs"][0], sigma_z=pp["sigma_z"][0], data_s=pp["d_hat_s"][0],
                data_err_s=data_err, mass=pp["mass"][0], sel=sel)


def _class_logprob(survey_dict, params, sigma_pec, w0wa):
    """Per-object selection-conditional log f(d_i | class, theta)."""
    ll, data, _ = build_survey_likelihood(survey_dict, params, sigma_pec=sigma_pec, w0wa=w0wa)
    return np.asarray(ll.log_prob(data))


def build_catalog(seed=0, cosmo="cosmo1", contam_frac=0.35, mag_offset=0.4,
                  contam_beta_rate=3.0, sigma_pec=300, n_scale=1.0, verbose=True):
    """One contaminated realisation and its classification scores.

    Returns (grouped_data, mcmc_key, meta). Each survey dict also carries
      class_score : (n_sel,) mock classifier P(Ia | d_i, selected)
      is_ia       : (n_sel,) 1 for true SNe Ia (the hidden labels)
    """
    truths = resolve_truths(cosmo)
    w0wa = is_w0wa(cosmo)
    wa = truths["wa"] if w0wa else None
    M0_contam = truths["M0"] + mag_offset
    contam_overrides = dict(alpha=0.0, beta_int=0.0, alpha_c=0.0, gamma=0.0,
                            sigma_int=CONTAM_SIGMA_INT, M0=M0_contam)
    co_beta_rate = truths["beta_rate"] if contam_beta_rate is None else contam_beta_rate
    pi_A = 1.0 - contam_frac                      # pre-selection SN Ia fraction

    # Parameter sets of the two class-conditional densities
    A_params = dict(truths)
    B_params = dict(truths)
    B_params.update(alpha=0.0, beta_int=0.0, alpha_c=0.0, gamma=0.0,
                    sigma_int=CONTAM_SIGMA_INT, M0=M0_contam, beta_rate=co_beta_rate,
                    tau_hi=truths["tau"], RB_hi=truths["RB"])   # no dust step for non-Ia

    root = random.PRNGKey(_CONTAM_BLOCK + int(seed))
    survey_keys = random.split(root, len(SURVEYS) + 1)
    mcmc_key = survey_keys[-1]

    surveys = [dict(s, n_sne=int(round(s["n_sne"] * CONTAM_N_MULT * n_scale))) for s in SURVEYS]

    grouped = []
    n_ia_tot = n_co_tot = 0
    n_hz_ia = n_hz_co = 0          # high-z photometric survey only
    for survey, skey in zip(surveys, survey_keys[:-1]):
        k_ia, k_co, k_cls = random.split(skey, 3)
        ia_ov = {"gamma": truths["gamma"]}
        if truths.get("tau_hi", truths["tau"]) != truths["tau"]:
            ia_ov["tau_hi"] = truths["tau_hi"]           # host-mass dust step
        if truths.get("RB_hi", truths["RB"]) != truths["RB"]:
            ia_ov["RB_hi"] = truths["RB_hi"]
        ia = _simulate_population(k_ia, survey, ia_ov, truths["w0"], truths["Om0"],
                                  truths["beta_rate"], wa, sigma_pec)

        n = survey["n_sne"]
        if survey["lowz"]:
            # low-z spectroscopic survey: confirmed SNe Ia, always included
            is_ia      = jnp.ones((n,), dtype=bool)
            z_s        = ia["z_s"]
            sigma_z    = ia["sigma_z"]
            data_s     = ia["data_s"]
            data_err_s = ia["data_err_s"]
            mass       = ia["mass"]
            sel        = ia["sel"]
        else:
            co = _simulate_population(k_co, survey, contam_overrides, truths["w0"],
                                      truths["Om0"], co_beta_rate, wa, sigma_pec)
            is_ia = random.bernoulli(k_cls, pi_A, shape=(n,))   # Bernoulli(pi_Ia)

            def pick(arr_ia, arr_co, mask=is_ia):
                m = mask.reshape((-1,) + (1,) * (arr_ia.ndim - 1))
                return jnp.where(m, arr_ia, arr_co)

            z_s        = jnp.where(is_ia, ia["z_s"], co["z_s"])
            sigma_z    = jnp.where(is_ia, ia["sigma_z"], co["sigma_z"])
            data_s     = pick(ia["data_s"], co["data_s"])
            data_err_s = pick(ia["data_err_s"], co["data_err_s"])
            mass       = jnp.where(is_ia, ia["mass"], co["mass"])
            sel        = jnp.where(is_ia, ia["sel"], co["sel"])

        keep = np.asarray(sel)
        is_ia_np = np.asarray(is_ia)
        n_ia_k = int(np.sum(is_ia_np[keep]))
        n_co_k = int(np.sum(~is_ia_np[keep]))
        n_ia_tot += n_ia_k
        n_co_tot += n_co_k
        if not survey["lowz"]:
            n_hz_ia += n_ia_k
            n_hz_co += n_co_k

        grouped.append(dict(
            z_s=z_s[keep], sigma_z=sigma_z[keep], data_s=data_s[keep],
            data_err_s=data_err_s[keep], mass=mass[keep],
            m_cut=survey["m_cut"], sigma_mcut=survey["sigma_mcut"],
            z_cut=survey["z_cut"], sigma_zcut=survey["sigma_zcut"],
            zmax=survey_zmax(survey), s_id=survey["id"], lowz=survey["lowz"],
            is_ia=jnp.asarray(is_ia[keep]).astype(jnp.float64),
        ))

    # Mock classifier: exact conditional class probability given selection
    n_hz = n_hz_ia + n_hz_co
    pi_sel = n_hz_ia / max(n_hz, 1)
    for d in grouped:
        if d["lowz"]:
            d["class_score"] = jnp.ones_like(d["is_ia"])
            continue
        log_fA = _class_logprob(d, A_params, sigma_pec, w0wa=w0wa)
        log_fB = _class_logprob(d, B_params, sigma_pec, w0wa=w0wa)
        log_wA = np.log(pi_sel) + log_fA
        log_wB = np.log(1.0 - pi_sel) + log_fB
        q = 1.0 / (1.0 + np.exp(log_wB - log_wA))
        d["class_score"] = jnp.asarray(q)

    n_sel = n_ia_tot + n_co_tot
    meta = dict(seed=seed, cosmo=cosmo, contam_frac_target=contam_frac, pi_A=pi_A,
                n_scale=n_scale, mag_offset=mag_offset, contam_beta_rate=co_beta_rate,
                ia_beta_rate=truths["beta_rate"], n_selected=n_sel, n_ia=n_ia_tot,
                n_contam=n_co_tot, n_highz=n_hz, pi_sel=pi_sel,
                sel_contam_frac=n_co_tot / max(n_sel, 1),
                hz_contam_frac=n_hz_co / max(n_hz, 1),
                truths=truths, contam_params=B_params)
    if verbose:
        print(f"Contaminated catalogue [{cosmo}, seed {seed}]: {n_sel} selected "
              f"({n_ia_tot} SNe Ia + {n_co_tot} contaminants, "
              f"{100 * meta['sel_contam_frac']:.1f}% contaminated); "
              f"high-z contaminated fraction {meta['hz_contam_frac']:.3f}")
    return grouped, mcmc_key, meta
