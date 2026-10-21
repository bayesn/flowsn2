"""Cosmological distances for FlowSN2.

wCDM (cosmologies C1-C3) uses the vendored ``wcosmo`` package. w0waCDM (C4, C5)
and every CMB / BAO observable use the JAX-native flat CPL implementation below,
for the dark-energy equation of state

    w(a) = w0 + wa (1 - a) = w0 + wa * z / (1 + z).

Flat universe; the late-time distances are matter + dark energy only, matching
astropy.cosmology.Flatw0waCDM(..., Tcmb0=0) and reducing to wCDM when wa = 0.
Everything is pure jnp: vmappable and differentiable for NUTS.

    rho_DE(z)/rho_DE(0) = (1+z)^{3(1+w0+wa)} * exp(-3 wa z/(1+z))
    E(z)                = sqrt( Om0 (1+z)^3 + (1-Om0) rho_DE(z)/rho_DE(0) )
    D_C(z)              = (c/H0) \\int_0^z dz'/E(z')
    mu                  = 5 log10((1+z) D_C [Mpc]) + 25
    dV_C/dz/dOmega      = (c/H0) D_C(z)^2 / E(z)
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
import wcosmo

H0 = 70.0                    # [km/s/Mpc], fixed throughout
C_KM_S = 299792.458          # speed of light [km/s]
H0_DEFAULT = H0


# -------------------------------------------------------------------------
# wCDM via wcosmo (C1-C3)
# -------------------------------------------------------------------------
@jit
def time_jax_comoving_distance(jdata, omega_m, w):
    return wcosmo.FlatwCDM(H0, omega_m[0], w[0]).comoving_distance(jdata)


@jit
def jax_dvcdz(jdata, omega_m, w):
    return wcosmo.FlatwCDM(H0, omega_m[0], w[0]).differential_comoving_volume(jdata)


def mu_func(z, Om0, w0):
    """wCDM distance modulus mu(z) [mag]."""
    d__ = wcosmo.FlatwCDM(H0, Om0, w0).comoving_distance(z)
    return 5 * jnp.log10(d__ * (1 + z) * 1e6 / 10)


# -------------------------------------------------------------------------
# Flat w0waCDM (CPL). Fixed Gauss-Legendre quadrature on [0, z]: the integrand
# 1/E(z) is smooth and monotone over z in [0, ~2.5], so 64 nodes agree with
# adaptive quadrature (astropy) to machine precision.
# -------------------------------------------------------------------------
_GL_DEG = 64
_x, _w = np.polynomial.legendre.leggauss(_GL_DEG)
_GL_X = jnp.asarray(_x)
_GL_W = jnp.asarray(_w)


def de_density(z, w0, wa):
    """rho_DE(z) / rho_DE(0) for CPL, flat (exact closed form)."""
    zp1 = 1.0 + z
    return zp1 ** (3.0 * (1.0 + w0 + wa)) * jnp.exp(-3.0 * wa * z / zp1)


def E(z, Om0, w0, wa):
    """Dimensionless Hubble rate H(z)/H0 (flat, matter + DE, no radiation)."""
    zp1 = 1.0 + z
    return jnp.sqrt(Om0 * zp1 ** 3 + (1.0 - Om0) * de_density(z, w0, wa))


def comoving_distance(z, Om0, w0, wa, H0=H0_DEFAULT):
    """Line-of-sight comoving distance D_C(z) [Mpc]. Vectorized over z."""
    z = jnp.asarray(z, dtype=jnp.float64)
    half = 0.5 * z                              # map [-1,1] -> [0, z]
    zz = half[..., None] * (_GL_X + 1.0)        # (*z.shape, GL_DEG)
    integ = jnp.sum(_GL_W / E(zz, Om0, w0, wa), axis=-1)   # (*z.shape,)
    return (C_KM_S / H0) * half * integ


def luminosity_distance(z, Om0, w0, wa, H0=H0_DEFAULT):
    """Luminosity distance D_L(z) [Mpc] (flat)."""
    return (1.0 + jnp.asarray(z)) * comoving_distance(z, Om0, w0, wa, H0)


def distance_modulus(z, Om0, w0, wa, H0=H0_DEFAULT):
    """Distance modulus mu(z) [mag]: 5 log10(D_L[Mpc]) + 25."""
    return 5.0 * jnp.log10(luminosity_distance(z, Om0, w0, wa, H0)) + 25.0


def differential_comoving_volume(z, Om0, w0, wa, H0=H0_DEFAULT):
    """dV_C/dz/dOmega [Mpc^3/sr] (flat): D_H * D_C(z)^2 / E(z)."""
    DC = comoving_distance(z, Om0, w0, wa, H0)
    return (C_KM_S / H0) * DC ** 2 / E(z, Om0, w0, wa)


# -------------------------------------------------------------------------
# CMB distance priors (Appendix D.1): shift parameter R and acoustic scale l_A,
# both evaluated at the Hu & Sugiyama (1996) recombination redshift z_* with a
# radiation-aware D_M. Early-universe inputs are fixed (omega_b, T_cmb, N_eff)
# since H0 is fixed and the baryon density is not sampled.
# -------------------------------------------------------------------------
Z_STAR = 1089.0
_GL_HZ_DEG = 256
_xh, _wh = np.polynomial.legendre.leggauss(_GL_HZ_DEG)
_GL_HZ_X = jnp.asarray(_xh)
_GL_HZ_W = jnp.asarray(_wh)

OMEGA_B_H2 = 0.02237      # fixed physical baryon density (Planck)
T_CMB      = 2.7255       # K
N_EFF      = 3.046

# GL nodes for the z_* -> inf sound-horizon integral (log-spaced)
_xr, _wr = np.polynomial.legendre.leggauss(256)
_GL_RS_X = jnp.asarray(_xr)
_GL_RS_W = jnp.asarray(_wr)


def cmb_shift_R(Om0, w0, wa, H0=H0_DEFAULT, z_star=None):
    """CMB shift parameter R = sqrt(Om0) * H0 * D_M(z_*) / c."""
    if z_star is None:
        z_star = z_star_hu_sugiyama(Om0, H0)
    DM = comoving_distance_rad(z_star, Om0, w0, wa, H0)
    return jnp.sqrt(Om0) * H0 * DM / C_KM_S


def _omega_gamma_h2():
    return 2.469e-5 * (T_CMB / 2.725) ** 4


def z_star_hu_sugiyama(Om0, H0=H0_DEFAULT):
    """Recombination redshift z_* (Hu & Sugiyama 1996, eq. E-1), a function of
    omega_b (fixed) and omega_m = Om0*(H0/100)^2."""
    ob = OMEGA_B_H2
    om = Om0 * (H0 / 100.0) ** 2
    g1 = 0.0783 * ob ** -0.238 / (1.0 + 39.5 * ob ** 0.763)
    g2 = 0.560 / (1.0 + 21.1 * ob ** 1.81)
    return 1048.0 * (1.0 + 0.00124 * ob ** -0.738) * (1.0 + g1 * om ** g2)


def _E_rad(z, Om0, w0, wa, H0):
    """H(z)/H0 INCLUDING radiation (photons + neutrinos), for the high-z
    sound-horizon and distance-to-recombination integrals only."""
    h = H0 / 100.0
    Or = _omega_gamma_h2() * (1.0 + 0.2271 * N_EFF) / h ** 2
    Ode = 1.0 - Om0 - Or
    zp = 1.0 + z
    return jnp.sqrt(Om0 * zp ** 3 + Or * zp ** 4 + Ode * de_density(z, w0, wa))


def _sound_speed(z):
    """Baryon-photon sound speed c_s(z) [km/s]."""
    Rb_pref = 0.75 * OMEGA_B_H2 / _omega_gamma_h2()
    return C_KM_S / jnp.sqrt(3.0 * (1.0 + Rb_pref / (1.0 + z)))


def sound_horizon_rs(Om0, w0, wa, H0=H0_DEFAULT, z_star=Z_STAR):
    """Comoving sound horizon r_s(z_*) [Mpc] = int_{z_*}^inf c_s/H dz."""
    x0 = jnp.log1p(z_star)
    x1 = jnp.log1p(1e8)
    half = 0.5 * (x1 - x0)
    mid = 0.5 * (x1 + x0)
    xx = mid + half * _GL_RS_X
    zz = jnp.expm1(xx)
    integrand = _sound_speed(zz) / (H0 * _E_rad(zz, Om0, w0, wa, H0)) * jnp.exp(xx)
    return half * jnp.sum(_GL_RS_W * integrand)


def comoving_distance_rad(z, Om0, w0, wa, H0=H0_DEFAULT):
    """D_M(z) [Mpc] with the radiation-aware E, integrated in x = ln(1+z)."""
    xmax = jnp.log1p(jnp.asarray(z, dtype=jnp.float64))
    half = 0.5 * xmax
    xx = half[..., None] * (_GL_HZ_X + 1.0)
    zz = jnp.expm1(xx)
    integ = jnp.sum(_GL_HZ_W * jnp.exp(xx) / _E_rad(zz, Om0, w0, wa, H0), axis=-1)
    return (C_KM_S / H0) * half * integ


def cmb_acoustic_lA(Om0, w0, wa, H0=H0_DEFAULT, z_star=None):
    """CMB acoustic scale l_A = pi * D_M(z_*) / r_s(z_*)."""
    if z_star is None:
        z_star = z_star_hu_sugiyama(Om0, H0)
    DM = comoving_distance_rad(z_star, Om0, w0, wa, H0)
    rs = sound_horizon_rs(Om0, w0, wa, H0, z_star)
    return jnp.pi * DM / rs


# -------------------------------------------------------------------------
# BAO distances (Appendix D.2). r_d from the Aubourg et al. (2015) fitting
# formula: a function of (omega_m, omega_b) only, with no w0/wa dependence.
# -------------------------------------------------------------------------
def sound_horizon_rd(Om0, H0=H0_DEFAULT):
    """Drag-epoch sound horizon r_d [Mpc] (Aubourg 2015; massless neutrinos)."""
    om = Om0 * (H0 / 100.0) ** 2
    ob = OMEGA_B_H2
    return 55.154 * jnp.exp(-72.3 * (0.0006) ** 2) / (om ** 0.25351 * ob ** 0.12807)


def hubble_distance(z, Om0, w0, wa, H0=H0_DEFAULT):
    """Hubble distance D_H(z) = c / H(z) [Mpc] (late-time E)."""
    return C_KM_S / (H0 * E(z, Om0, w0, wa))


def dV(z, Om0, w0, wa, H0=H0_DEFAULT):
    """Volume-averaged distance D_V(z) = (z * D_M^2 * D_H)^(1/3) [Mpc]."""
    DM = comoving_distance(z, Om0, w0, wa, H0)
    DH = hubble_distance(z, Om0, w0, wa, H0)
    return (z * DM ** 2 * DH) ** (1.0 / 3.0)
