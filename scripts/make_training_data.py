"""Simulate the flow training sets (Sections 2.5.1 and 3.1.2).

    python scripts/make_training_data.py                 # both surveys
    python scripts/make_training_data.py --survey lowz

25 million SELECTED SNe Ia per survey at the fiducial C1, xi_rate = 1.5,
M0 = -19.38 and gamma = 0, with the population parameters drawn per object from
the training ranges of Table C1 and a N(0, 0.25^2) smearing on the latent
distance. Writes training_data/{highz,lowz}_training_set.npy, 23 columns:

    0-2  m_hat, c_hat, x_hat       3  z_hat          4  m0 (latent, pre pec-vel)
    5    c0_int    6  x0           7  latent z
    8-10 sigma_m, sigma_c, sigma_x                   11-13  W_mc, W_mx, W_cx
    14   sigma_z   15-22  alpha, beta_int, alpha_c, RB, tau, sigma_int, sigma_c, sigma_x

Each survey takes ~1h and ~5 GB of disk.
"""
import argparse

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import Predictive

from flowsn2.simulator import sample_model
from flowsn2.experiments import SURVEYS
from flowsn2.paths import TRAINING_DATA

N_TRAIN = int(2.5e7)       # selected SNe per survey
CHUNK = 1_000_000          # objects simulated per forward-model call
DELTA_MU_SIGMA = 0.25      # sigma_mu, Section 2.5.1


def training_survey(sim_key, n_sne, m_cut, sigma_mcut, z_cut, sigma_zcut, lowz=False,
                    sigma_z_factor=0.03, sigma_pec=300):
    """One chunk: per-object population parameters from the training ranges,
    forward model, keep the selected objects."""
    prior_predictive = Predictive(sample_model, num_samples=1)

    sim_key, _ = jax.random.split(sim_key)
    alpha = dist.Uniform(-0.2, 0.0).sample(sim_key, sample_shape=(n_sne,))
    sim_key, _ = jax.random.split(sim_key)
    beta_int = dist.Uniform(1., 3.).sample(sim_key, sample_shape=(n_sne,))
    sim_key, _ = jax.random.split(sim_key)
    alpha_c = dist.Uniform(-0.02, 0.0).sample(sim_key, sample_shape=(n_sne,))
    sim_key, _ = jax.random.split(sim_key)
    x0 = dist.Uniform(-0.6, -0.2).sample(sim_key, sample_shape=(n_sne,))
    sim_key, _ = jax.random.split(sim_key)
    c0_int = dist.Uniform(-0.1, 0.1).sample(sim_key, sample_shape=(n_sne,))
    sim_key, _ = jax.random.split(sim_key)
    RB = dist.Uniform(2.2, 5.).sample(sim_key, sample_shape=(n_sne,))
    sim_key, _ = jax.random.split(sim_key)
    tau = dist.Uniform(0, 0.15).sample(sim_key, sample_shape=(n_sne,))
    sim_key, _ = jax.random.split(sim_key)
    sigma_int = dist.Uniform(0, 0.2).sample(sim_key, sample_shape=(n_sne,))
    sim_key, _ = jax.random.split(sim_key)
    sigma_c = dist.Uniform(0, 0.1).sample(sim_key, sample_shape=(n_sne,))
    sim_key, _ = jax.random.split(sim_key)
    sigma_x = dist.Uniform(0, 1.5).sample(sim_key, sample_shape=(n_sne,))
    sim_key, _ = jax.random.split(sim_key)

    pp = prior_predictive(
        sim_key, n_sne,
        m_cut=m_cut, sigma_mcut=sigma_mcut, z_cut=z_cut, sigma_zcut=sigma_zcut,
        lowz=lowz, sigma_z_factor=sigma_z_factor, sigma_pec=sigma_pec,
        delta_mu_sigma=DELTA_MU_SIGMA,
        alpha=alpha, beta_int=beta_int, alpha_c=alpha_c, RB=RB, tau=tau,
        c0_int=c0_int, x0=x0, sigma_int=sigma_int, sigma_c=sigma_c, sigma_x=sigma_x,
        gamma=0.,
    )

    sel = jnp.logical_and(pp["sel_s"][0, :] == 1, pp["sel_z"][0, :] == 1)
    W_s = pp["W_s"][0, sel]
    d_err_s = jnp.array([
        W_s[:, 0, 0]**0.5, W_s[:, 1, 1]**0.5, W_s[:, 2, 2]**0.5,
        W_s[:, 0, 1], W_s[:, 0, 2], W_s[:, 1, 2],
    ]).T
    return (pp["d_hat_s"][0][sel], pp["z_obs"][0][sel], pp["m_0"][0][sel],
            pp["latent_z"][0][sel], d_err_s, pp["sigma_z"][0, sel],
            alpha[sel], beta_int[sel], alpha_c[sel], RB[sel], tau[sel],
            c0_int[sel], x0[sel], sigma_int[sel], sigma_c[sel], sigma_x[sel])


def make_training(sim_key, survey, total=N_TRAIN, sigma_pec=300):
    """Accumulate chunks until `total` selected SNe, then truncate."""
    acc = None
    current_total = 0
    while current_total < total:
        sim_key, _ = jax.random.split(sim_key)
        out = training_survey(sim_key, CHUNK, m_cut=survey["m_cut"],
                              sigma_mcut=survey["sigma_mcut"], z_cut=survey["z_cut"],
                              sigma_zcut=survey["sigma_zcut"], lowz=survey["lowz"],
                              sigma_z_factor=survey["sigma_z_factor"], sigma_pec=sigma_pec)
        acc = out if acc is None else tuple(jnp.concatenate([a, b], axis=0)
                                            for a, b in zip(acc, out))
        current_total = acc[1].shape[0]
        print(f"  {current_total:,} / {total:,} selected", flush=True)
    return tuple(arr[:int(total)] for arr in acc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--survey", choices=["highz", "lowz", "both"], default="both")
    ap.add_argument("--sigma_pec", type=float, default=300)
    args = ap.parse_args()

    TRAINING_DATA.mkdir(parents=True, exist_ok=True)
    for survey in SURVEYS:
        name = "lowz" if survey["lowz"] else "highz"
        if args.survey not in (name, "both"):
            continue
        print(f"[{name}] simulating {N_TRAIN:,} selected SNe")
        (d_s, z_s, m0, latent_z, d_err_s, sigma_z, alpha, beta_int, alpha_c, RB, tau,
         c0_int, x0, sigma_int, sigma_c, sigma_x) = make_training(
            jax.random.PRNGKey(0), survey, total=N_TRAIN, sigma_pec=args.sigma_pec)

        variables = [d_s, z_s, m0, c0_int, x0, latent_z, d_err_s, sigma_z, alpha,
                     beta_int, alpha_c, RB, tau, sigma_int, sigma_c, sigma_x]
        columns = [jnp.atleast_2d(v).T if v.ndim == 1 else v for v in variables]
        out = TRAINING_DATA / f"{name}_training_set.npy"
        np.save(out, jnp.concatenate(columns, axis=1))
        print(f"[{name}] wrote {out}")


if __name__ == "__main__":
    main()
