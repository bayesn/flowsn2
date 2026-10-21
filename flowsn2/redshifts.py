"""Redshift distributions (Section 2.1.1): the volumetric power-law rate
P(z | xi_rate, C) ∝ (1+z)^xi_rate / (1+z) * dV_c/dz, and the photometric /
spectroscopic redshift sampling used by the forward model."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from jax.scipy.integrate import trapezoid
from jax.scipy.stats import norm
import numpyro
import numpyro.distributions as dist

from . import cosmology as cw
from .cosmology import jax_dvcdz

ZMIN = 0.01
ZMAX = 1.2          # high-z photometric survey
ZMAX_LOWZ = 0.3     # low-z spectroscopic survey

z_values = jnp.linspace(ZMIN, ZMAX, 1000)
z_values_lowz = jnp.linspace(ZMIN, ZMAX_LOWZ, 1000)


@jit
def integrate_row(pdf_row, x_row):
    return trapezoid(pdf_row, x=x_row, axis=-1)


@jit
def cdf(pdf_values):
    dz = z_values[1] - z_values[0]
    bin_areas = (pdf_values[:-1] + pdf_values[1:]) * 0.5 * dz
    cdf_vals = jnp.concatenate([jnp.zeros(1), jnp.cumsum(bin_areas)])
    return cdf_vals / cdf_vals[-1]


@jit
def cdf_lowz(pdf_values):
    dz = z_values_lowz[1] - z_values_lowz[0]
    bin_areas = (pdf_values[:-1] + pdf_values[1:]) * 0.5 * dz
    cdf_vals = jnp.concatenate([jnp.zeros(1), jnp.cumsum(bin_areas)])
    return cdf_vals / cdf_vals[-1]


@jit
def pdf(z, omega_m, w, beta_rate):
    prob = (
        (1 + z) ** (beta_rate)
        / (1 + z)
        * jax_dvcdz(z, jnp.array([omega_m]), jnp.array([w]))
    )
    all_prob = (
        (1 + z_values) ** (beta_rate)
        / (1 + z_values)
        * jax_dvcdz(z_values, jnp.array([omega_m]), jnp.array([w]))
    )
    integ = integrate_row(all_prob, z_values)
    return prob / integ


@jit
def pdf_lowz(z, omega_m, w, beta_rate):
    prob = (
        (1 + z) ** (beta_rate)
        / (1 + z)
        * jax_dvcdz(z, jnp.array([omega_m]), jnp.array([w]))
    )
    # The normalising constant mixes the low-z rate factor with dV_c/dz on the
    # high-z grid. It is a scalar that cdf_lowz divides out again, so sampling is
    # unaffected; it is kept exactly as run so the simulated redshifts are
    # reproduced to the last bit.
    all_prob = (
        (1 + z_values_lowz) ** (beta_rate)
        / (1 + z_values_lowz)
        * jax_dvcdz(z_values, jnp.array([omega_m]), jnp.array([w]))
    )
    integ = integrate_row(all_prob, z_values_lowz)
    return prob / integ


@jit
def pdf_unormalised(z, omega_m, w, beta_rate):
    return (
        (1 + z) ** (beta_rate)
        / (1 + z)
        * jax_dvcdz(z, jnp.array([omega_m]), jnp.array([w]))
    )


# w0waCDM (C4, C5): same rate shape with dV_c/dz from the CPL engine.
@jit
def jax_dvcdz_w0wa(z, Om0, w0, wa):
    return cw.differential_comoving_volume(z, Om0, w0, wa, cw.H0)


@jit
def pdf_unormalised_w0wa(z, Om0, w0, wa, beta_rate):
    return (1 + z) ** beta_rate / (1 + z) * jax_dvcdz_w0wa(z, Om0, w0, wa)


@jit
def pdf_w0wa(z, Om0, w0, wa, beta_rate):
    prob = pdf_unormalised_w0wa(z, Om0, w0, wa, beta_rate)
    all_prob = pdf_unormalised_w0wa(z_values, Om0, w0, wa, beta_rate)
    integ = integrate_row(all_prob, z_values)
    return prob / integ


@jit
def pdf_lowz_w0wa(z, Om0, w0, wa, beta_rate):
    prob = pdf_unormalised_w0wa(z, Om0, w0, wa, beta_rate)
    all_prob = pdf_unormalised_w0wa(z_values_lowz, Om0, w0, wa, beta_rate)
    integ = integrate_row(all_prob, z_values_lowz)
    return prob / integ


def sample_redshifts(size, Om0=0.315, w0=-1, beta_rate=1.5, sigma_z_factor=0.03,
                     sigma_pec=300, z_cut=0.8, sigma_zcut=0.05, sel_z=None, wa=None,
                     lowz=False):
    """Latent CMB-frame z from the rate, heliocentric z with peculiar velocity,
    observed z with Gaussian uncertainty sigma_z_factor*(1+z_hel), and the
    redshift selection draw (Eq. 22). wa=None -> wCDM rate, else CPL."""
    with numpyro.plate("plate_z", size):
        u_samps = numpyro.sample("u_samp", dist.Uniform(0, 1))

        if lowz:
            z_pdf = (pdf_lowz(z_values_lowz, Om0, w0, beta_rate) if wa is None
                     else pdf_lowz_w0wa(z_values_lowz, Om0, w0, wa, beta_rate))
            latent_z = jnp.interp(u_samps, cdf_lowz(z_pdf), z_values_lowz)
        else:
            z_pdf = (pdf(z_values, Om0, w0, beta_rate) if wa is None
                     else pdf_w0wa(z_values, Om0, w0, wa, beta_rate))
            latent_z = jnp.interp(u_samps, cdf(z_pdf), z_values)

        latent_z = numpyro.deterministic("latent_z", latent_z)

        z_pec = numpyro.sample("z_pec", dist.Normal(0, sigma_pec / 299792.458))
        z_hel = (1 + latent_z) * (1 + z_pec) - 1
        z_obs = numpyro.sample("z_obs", dist.TruncatedNormal(
            z_hel, sigma_z_factor * (1 + z_hel), low=0., high=1.2))
        sigma_z = numpyro.deterministic("sigma_z", sigma_z_factor * (1 + z_hel))
        p_z = norm.cdf(-(z_obs), loc=-z_cut, scale=sigma_zcut)
        sel_z = numpyro.sample("sel_z", dist.Bernoulli(p_z), obs=sel_z)

        return latent_z, z_obs, sel_z, z_pec, sigma_z
