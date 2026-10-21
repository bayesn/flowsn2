"""Simplified forward model (Sections 2.1 and 3.1.1).

Latent redshifts from the rate -> peculiar velocities and photometric redshifts
-> Dusty Simple-BayeSN latent (m, c, x) with a host-mass step -> summary-fit
covariances (Appendix B) -> observed summaries -> magnitude and redshift
selection. Used for the training sets, the test catalogues and the
contaminants. ``delta_mu_sigma`` adds the N(0, sigma_mu^2) distance smearing
used only for training.

Do not reorder the numpyro sample statements: the PRNG stream given to each
site depends on its position, and the paper's catalogues are tied to it.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.stats import norm

from . import cosmology as cw
from .cosmology import H0, time_jax_comoving_distance
from .redshifts import sample_redshifts


def sample_model(n_sne, d_hat_s=None, log_x_err=None, log_c_err=None, log_mag_err=None,
                 cov_m_c=None, cov_c_x=None, cov_m_x=None, sel_s=None,
                 M0=-19.380, sigma_int=0.101, alpha=-0.149, alpha_c=-0.006, beta_int=2.322,
                 gamma=0.1, x0=-0.452, sigma_x=1.137, c0_int=-0.056, sigma_c=0.071,
                 tau=0.068, RB=3.740, h=H0 / 100, Om0=0.315, w0=-1, beta_rate=1.5,
                 m_cut=24.5, sigma_mcut=0.5, a=-0.1, b=-1, z_cut=0.9, sigma_zcut=0.1,
                 sigma_pec=300, sigma_z_factor=0.03, M_split=10, lowz=False,
                 delta_mu_sigma=None, wa=None, tau_hi=None, RB_hi=None):
    """One survey's population. Defaults are the fiducial SN Ia truths.

    wa=None simulates wCDM distances with wcosmo; a float uses the CPL engine.
    tau_hi / RB_hi (None = no step) give high-mass hosts their own dust.
    """
    latent_z, z_obs, sel_z, z_pec, sigma_z = sample_redshifts(
        size=n_sne, Om0=Om0, w0=w0, beta_rate=beta_rate, sigma_z_factor=sigma_z_factor,
        sigma_pec=sigma_pec, z_cut=z_cut, sigma_zcut=sigma_zcut, wa=wa, lowz=lowz)

    with numpyro.plate("plate_i", n_sne):

        mass = numpyro.sample("mass", dist.Uniform(8, 12))
        mass_mask = jnp.where(mass > M_split, 1, 0)
        if wa is None:
            d_s = time_jax_comoving_distance(latent_z, jnp.array([Om0]), jnp.array([w0]))
            mu_s = 5 * jnp.log10(d_s * (1 + latent_z) * 1e6 / 10)
        else:
            mu_s = cw.distance_modulus(latent_z, Om0, w0, wa, H0)
        # host-mass dust step: high-mass hosts get their own tau / RB
        tau_s = tau if tau_hi is None else jnp.where(mass_mask == 1, tau_hi, tau)
        RB_s = RB if RB_hi is None else jnp.where(mass_mask == 1, RB_hi, RB)
        E_s = numpyro.sample("E_s", dist.Exponential(1 / (tau_s)))
        x_s = numpyro.sample("x_s", dist.Normal(x0, sigma_x))
        c_int_s = numpyro.sample("c_s", dist.Normal(c0_int + alpha_c * x_s, sigma_c))

        if delta_mu_sigma is not None:
            # training-set distance smearing (site name kept as run)
            delta_mu = numpyro.sample("detla_mu", dist.Normal(0, delta_mu_sigma))
        else:
            delta_mu = 0

        # m_0 is recorded BEFORE peculiar velocities: the flow is conditioned on it
        m_0 = numpyro.deterministic("m_0", M0 + mu_s + delta_mu + gamma * mass_mask)

        m_0 = m_0 + 10 * jnp.log10(1 + z_pec)  # apply peculiar velocities

        m_s = numpyro.sample("m_s", dist.Normal(m_0 + alpha * x_s + beta_int * c_int_s
                                                + RB_s * E_s, sigma_int))

        c_s = c_int_s + E_s

        # summary-fit covariances (Appendix B)
        log_x_err = numpyro.sample("log_x_err_s", dist.Normal(-1.5, 0.5), obs=log_x_err)
        log_c_err = numpyro.sample("log_c_err_s", dist.Normal(-3.5, 0.3), obs=log_c_err)
        log_mag_err = numpyro.sample("log_mag_err_s", dist.Normal(0.1 * (m_s - 56), 0.6),
                                     obs=log_mag_err)

        cov_m_c = numpyro.sample("cov_m_c", dist.Normal(
            0.000520 + 0.288792999 * jnp.exp(log_mag_err)**2,
            jnp.exp(log_mag_err)**2 * 0.125979), obs=cov_m_c)
        cov_m_x = numpyro.sample("cov_m_x", dist.Normal(
            0.0008097 + 0.03835550 * jnp.exp(log_x_err)**2,
            jnp.exp(log_x_err)**2 * 0.02032), obs=cov_m_x)
        cov_c_x = numpyro.sample("cov_c_x", dist.Normal(
            0.000168655 + 0.01358504 * jnp.exp(log_x_err)**2,
            jnp.exp(log_x_err)**2 * 0.011840861), obs=cov_c_x)

        W_s_unstable = jnp.array([[jnp.exp(log_mag_err)**2, cov_m_c, cov_m_x],
                                  [cov_m_c, jnp.exp(log_c_err)**2, cov_c_x],
                                  [cov_m_x, cov_c_x, jnp.exp(log_x_err)**2]]).T

        # eigenvalue floor (lambda_min > 1e-5) -> strictly positive-definite
        W_s = (lambda vals, vecs: (vecs * jnp.maximum(vals, 1e-5)[:, None, :])
               @ jnp.swapaxes(vecs, -1, -2))(*jnp.linalg.eigh(W_s_unstable))

    W_s = numpyro.deterministic("W_s", W_s)

    d_hat_s = numpyro.sample("d_hat_s", dist.MultivariateNormal(
        jnp.column_stack((m_s, c_s, x_s)), W_s), obs=d_hat_s)
    p_s = norm.cdf(-(d_hat_s[:, 0] + a * d_hat_s[:, 2] + b * d_hat_s[:, 1]),
                   loc=-m_cut, scale=sigma_mcut)
    sel_s = numpyro.sample("sel_s", dist.Bernoulli(p_s), obs=sel_s)
