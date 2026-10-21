"""Exact selection-aware likelihood for the simplified forward model (Appendix A).

``AnalyticalLikelihood`` evaluates log P(d_hat, z_hat | I=1, Theta) per SN: the
numerator integrates the closed-form marginal (Eq. B4) times the selection
probability over a local latent-z grid; the denominator integrates the
analytical selection probability over the full redshift range. It is the
ground truth FlowSN2 is validated against and also defines the mock classifier.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats.norm import logcdf as log_ndtr
from jax.scipy.stats import multivariate_normal
from jax.scipy.stats.norm import cdf as ndtr
from jax.scipy.stats.norm import logcdf
from numpyro.distributions import (
    Distribution,
    constraints)
from numpyro.distributions.util import validate_sample

def _sufficient_stats(y, Sigma, e_E):
    # y: (..., 4)
    # Sigma: (..., 4, 4)
    # e_E: (..., 4)

    Sigma_inv_eE = jnp.linalg.solve(Sigma, e_E[..., None])[..., 0]
    Sigma_inv_y  = jnp.linalg.solve(Sigma, y[..., None])[..., 0]

    s_E_inv2 = jnp.sum(e_E * Sigma_inv_eE, axis=-1)
    s_E2     = 1.0 / s_E_inv2
    s_E      = jnp.sqrt(s_E2)

    E_hat    = s_E2 * jnp.sum(e_E * Sigma_inv_y, axis=-1)

    return s_E, E_hat

@jax.jit
def log_b4(m_resid,c_resid,x_resid,z_resid,var_m, var_c, var_x, var_z, cov_m_c, cov_m_x, cov_m_z, cov_c_x, RB, tau):
    """
    Log marginal likelihood log P(d_s | z_s; Theta, Omega) from eq. B4.

    Computes all four factors in log space for numerical stability:

        log P = 0.5 * log(2 pi s_E^2)
                + log N(y | e_E * E_hat, Sigma)
                + [ (s_E/tau)^2 / 2  -  E_hat/tau  -  log(tau) ]
                + log Phi( E_hat/s_E - s_E/tau )

    Inherits dtype from inputs.
    """

    y = jnp.stack([m_resid, c_resid, x_resid, z_resid], axis=-1)

    zeros_ = jnp.zeros_like(var_m)

    Sigma = jnp.stack([
        jnp.stack([var_m,    cov_m_c, cov_m_x, cov_m_z], axis=-1),
        jnp.stack([cov_m_c,  var_c,   cov_c_x, zeros_  ], axis=-1),
        jnp.stack([cov_m_x,  cov_c_x, var_x,   zeros_  ], axis=-1),
        jnp.stack([cov_m_z,  zeros_,  zeros_,  var_z   ], axis=-1),
    ], axis=-2)

    ones_ = jnp.ones_like(RB)
    zeros_ = jnp.zeros_like(RB)
    e_E = jnp.stack([RB, ones_, zeros_, zeros_], axis=-1)

    tau   = jnp.asarray(tau)

    s_E, E_hat = _sufficient_stats(y, Sigma, e_E)

    log_f1 = 0.5 * jnp.log(2.0 * jnp.pi * s_E ** 2)
    log_f2 = multivariate_normal.logpdf(y, mean=e_E * E_hat[..., None], cov=Sigma)
    log_f3 = -jnp.log(tau) + 0.5 * (s_E / tau) ** 2 - E_hat / tau
    log_f4 = log_ndtr(E_hat / s_E - s_E / tau)

    return log_f1 + log_f2 + log_f3 + log_f4


# ---------------------------------------------------------------------------
# Gauss-Legendre nodes & weights on [-1, 1], n = 20.
# Genz (2004) Table 1.
# ---------------------------------------------------------------------------
_GL_X_np = (
    -0.9931285991850949, -0.9639719272779138, -0.9122344282513259,
    -0.8391169718222188, -0.7463062256567499, -0.6360536807265150,
    -0.5108670019508271, -0.3737060887154195, -0.2277858511416451,
    -0.0765265211334973,  0.0765265211334973,  0.2277858511416451,
     0.3737060887154195,  0.5108670019508271,  0.6360536807265150,
     0.7463062256567499,  0.8391169718222188,  0.9122344282513259,
     0.9639719272779138,  0.9931285991850949,
)
_GL_W_np = (
    0.0176140071391521, 0.0406014298003869, 0.0626720483341091,
    0.0832767415767048, 0.1019301198172404, 0.1181945319615184,
    0.1316886384491766, 0.1420961093183820, 0.1491729864726037,
    0.1527533871307258, 0.1527533871307258, 0.1491729864726037,
    0.1420961093183820, 0.1316886384491766, 0.1181945319615184,
    0.1019301198172404, 0.0832767415767048, 0.0626720483341091,
    0.0406014298003869, 0.0176140071391521,
)

@jax.jit
def bvn_cdf(h, k, rho):
    """BVN CDF P(Z1<=h, Z2<=k; rho). Genz GL-20, branch-free, differentiable."""
    h   = jnp.asarray(h)
    k   = jnp.asarray(k)
    rho = jnp.asarray(rho)
    GL_X = jnp.array(_GL_X_np, dtype=h.dtype)
    GL_W = jnp.array(_GL_W_np, dtype=h.dtype)
    asr  = jnp.arcsin(rho)
    t    = asr * (1.0 + GL_X) * 0.5
    sin_t  = jnp.sin(t)
    cos2_t = jnp.cos(t) ** 2
    exponent = -(h*h + k*k - 2.0*h*k*sin_t) / (2.0*cos2_t)
    quad = (asr / (4.0*jnp.pi)) * jnp.dot(GL_W, jnp.exp(exponent)/cos2_t)
    return jnp.clip(quad + ndtr(h)*ndtr(k), 0.0, 1.0)


@jax.jit
def log_B(ha, hb, k, rho, tau):
    """
    log B = log integral_0^inf BVN(ha + hb*E, k; rho) * (1/tau) * exp(-E/tau) dE

    For hb > 0:
        B = BVN(ha,k;rho) + exp(shift)*[Phi(k+rho/(hb*tau)) - BVN(ha+1/(hb*tau), k+rho/(hb*tau); rho)]
    For hb < 0:
        B = BVN(ha,k;rho) - exp(shift)*BVN(ha+1/(hb*tau), k+rho/(hb*tau); rho)
    where shift = ha/(hb*tau) + 1/(2*hb^2*tau^2)
    """
    ha  = jnp.asarray(ha)
    hb  = jnp.asarray(hb)
    k   = jnp.asarray(k)
    rho = jnp.asarray(rho)
    tau = jnp.asarray(tau)

    shift   = ha / (hb * tau) + 0.5 / (hb * tau)**2
    shift_a = ha + 1.0 / (hb * tau)
    shift_c = k  + rho / (hb * tau)

    bvn1  = bvn_cdf(ha, k, rho)
    bvn2  = bvn_cdf(shift_a, shift_c, rho)
    phi_c = ndtr(shift_c)

    val_pos = bvn1 + jnp.exp(shift) * (phi_c - bvn2)
    val_neg = bvn1 - jnp.exp(shift) * bvn2

    B = jnp.where(hb > 0, val_pos, val_neg)
    return jnp.log(B)


def _bcast_sne(x, n_sne, N):
    """Broadcast a global (shape (1,)) or per-SN (shape (n_sne,)) parameter to
    (n_sne, N). The (1,) branch reproduces the previous jnp.full(..., x[0])
    exactly, so behaviour is unchanged unless a per-SN array is passed (which
    only happens with the host-mass dust step, dust_step=True)."""
    x = jnp.asarray(x)
    if x.shape[0] == 1:
        return jnp.full((n_sne, N), x[0])
    return jnp.broadcast_to(x[:, None], (n_sne, N))


def log_B_vmap(ha, hb, k, rho, tau):
    # inputs are all (n_sne, N) — vmap over both axes
    return jax.vmap(jax.vmap(log_B))(ha, hb, k, rho, tau)


class AnalyticalLikelihood(Distribution):
    arg_constraints = {
        # Numerator: per-SN local grid (n_sne, N_num) — centred on z_obs
        "z_num": constraints.real, "m0_num": constraints.real, "log_w_num": constraints.real,
        # Denominator: global uniform grid (N_dg,) — covers full redshift range
        "z_dg":  constraints.real, "m0_dg":  constraints.real, "log_w_dg":  constraints.real,
        "c0_int": constraints.real, "x0": constraints.real,
        "sigma_int": constraints.real, "sigma_c": constraints.real, "sigma_x": constraints.real,
        "alpha": constraints.real, "beta_int": constraints.real, "alpha_c": constraints.real,
        "RB": constraints.real, "tau": constraints.real, "gamma":  constraints.real,
        "m_cut": constraints.real, "sigma_mcut": constraints.real,
        "z_cut": constraints.real, "sigma_zcut": constraints.real,
        "a": constraints.real, "b": constraints.real,
        "W_mm": constraints.real, "W_cc": constraints.real, "W_xx": constraints.real,
        "W_mc": constraints.real, "W_mx": constraints.real, "W_cx": constraints.real,
        "sigma_z": constraints.real, "sigma_pec": constraints.real,
    }

    support = constraints.real
    reparametrized_params = [
        "z_num", "m0_num", "log_w_num", "z_dg", "m0_dg", "log_w_dg",
        "c0_int", "x0", "sigma_int", "sigma_c", "sigma_x",
        "alpha", "beta_int", "alpha_c", "RB", "tau", "gamma",
        "W_mm", "W_cc", "W_xx", "W_mc", "W_mx", "W_cx",
        "sigma_z", "sigma_pec", "m_cut", "sigma_mcut", "z_cut", "sigma_zcut", "a", "b",
    ]

    def __init__(self, z_num, m0_num, log_w_num, z_dg, m0_dg, log_w_dg,
                 c0_int, x0, sigma_int, sigma_c, sigma_x,
                 alpha, beta_int, alpha_c, RB, tau, gamma,
                 W_mm, W_cc, W_xx, W_mc, W_mx, W_cx,
                 sigma_z, sigma_pec, m_cut, sigma_mcut, z_cut, sigma_zcut, a, b,
                 *, validate_args=False):
        (self.z_num, self.m0_num, self.log_w_num,
         self.z_dg,  self.m0_dg,  self.log_w_dg,
         self.c0_int, self.x0, self.sigma_int, self.sigma_c, self.sigma_x,
         self.alpha, self.beta_int, self.alpha_c, self.RB, self.tau, self.gamma,
         self.W_mm, self.W_cc, self.W_xx, self.W_mc, self.W_mx, self.W_cx,
         self.sigma_z, self.sigma_pec, self.m_cut, self.sigma_mcut,
         self.z_cut, self.sigma_zcut, self.a, self.b) = (
            z_num, m0_num, log_w_num, z_dg, m0_dg, log_w_dg,
            c0_int, x0, sigma_int, sigma_c, sigma_x,
            alpha, beta_int, alpha_c, RB, tau, gamma,
            W_mm, W_cc, W_xx, W_mc, W_mx, W_cx,
            sigma_z, sigma_pec, m_cut, sigma_mcut,
            z_cut, sigma_zcut, a, b)
        super(AnalyticalLikelihood, self).__init__(
            batch_shape=(W_mm.shape[0],),
            event_shape=(4,),
            validate_args=validate_args,
        )

    @jax.jit
    @validate_sample
    def log_prob(self, value):
        n_sne = value.shape[0]
        N_num = self.z_num.shape[1]
        N_dg  = self.z_dg.shape[0]

        exp_x = self.x0                                               # (1,)
        exp_c = self.c0_int + self.alpha_c * self.x0                 # (1,)

        # Per-SN, z-independent covariance terms
        var_m    = (10*self.sigma_pec/(jnp.log(10)*299792.458))**2 + self.sigma_int**2 + (self.alpha+self.beta_int*self.alpha_c)**2*self.sigma_x**2 + self.beta_int**2*self.sigma_c**2 + self.W_mm
        var_c    = self.sigma_c**2 + self.alpha_c**2*self.sigma_x**2 + self.W_cc
        var_x    = self.sigma_x**2 + self.W_xx
        var_z    = self.sigma_pec**2/(299792.458)**2 + self.sigma_z**2
        cov_m_c  = self.alpha*self.alpha_c*self.sigma_x**2 + self.beta_int*(self.sigma_c**2 + self.alpha_c**2*self.sigma_x**2) + self.W_mc
        cov_m_x  = self.alpha*self.sigma_x**2 + self.beta_int*self.alpha_c*self.sigma_x**2 + self.W_mx
        cov_c_x  = self.alpha_c*self.sigma_x**2 + self.W_cx
        cov_m_z  = jnp.repeat((10*self.sigma_pec**2/(jnp.log(10)*(299792.458)**2))[...,None], n_sne, axis=0)

        var_c_giv_x  = var_c - cov_c_x**2/var_x
        var_c_giv_x  = jnp.where(var_c_giv_x > 0, var_c_giv_x, 1e-8)
        var_m_giv_x  = var_m - cov_m_x**2/var_x
        var_m_giv_x  = jnp.where(var_m_giv_x > 0, var_m_giv_x, 1e-8)
        cov_mc_giv_x = cov_m_c - cov_m_x*cov_c_x/var_x
        var_m_giv_cx = var_m_giv_x - cov_mc_giv_x**2/var_c_giv_x
        var_m_giv_cx = jnp.where(var_m_giv_cx > 0, var_m_giv_cx, 1e-8)

        b_         = self.b + cov_mc_giv_x/var_c_giv_x
        a_         = self.a + cov_m_x/var_x + cov_c_x/var_x*self.b

        sigma_cut1 = self.sigma_mcut * jnp.sqrt(1 + var_m_giv_cx/self.sigma_mcut**2)
        sigma_cut2 = sigma_cut1 * jnp.sqrt(1 + var_c_giv_x*b_**2/sigma_cut1**2)
        sigma_cut3 = sigma_cut2 * jnp.sqrt(1 + var_x*a_**2/sigma_cut2**2)

        _dscale = jnp.sqrt(sigma_cut3[...,None]**2 + cov_m_z[...,None]**2 / var_z[...,None])   # (n_sne, 1)
        ha_     = (self.m_cut[...,None] - (exp_c*self.b[...,None] + exp_x*self.a[...,None])) / _dscale  # (n_sne, 1)
        hb      = -(self.RB[...,None] + self.b[...,None]) / _dscale                                     # (n_sne, 1)
        rho     = cov_m_z[...,None] / (jnp.sqrt(self.sigma_zcut**2 + var_z)[...,None] * _dscale)       # (n_sne, 1)

        # Denominator: ∫ B(z) P(z|rate) dz on global grid
        exp_m_dg = self.m0_dg + self.alpha*self.x0 + self.beta_int*(self.alpha_c*self.x0 + self.c0_int)  # (N_dg,)
        ha_dg  = ha_  - (self.gamma[...,None] + exp_m_dg[None, :]) / _dscale                                                      # (n_sne, N_dg)
        hb_dg  = jnp.repeat(hb,  N_dg, axis=1)
        k_dg   = (self.z_cut - self.z_dg[None, :]) / jnp.sqrt(self.sigma_zcut**2 + var_z)[:, None]
        rho_dg = jnp.repeat(rho, N_dg, axis=1)
        tau_dg = _bcast_sne(self.tau, n_sne, N_dg)
        B_dg   = log_B_vmap(ha_dg, hb_dg, k_dg, rho_dg, tau_dg)
        log_denom_s = logsumexp(B_dg + self.log_w_dg[None, :], axis=-1)                                  # (n_sne,)

        # Numerator: ∫ A(z) P(z|rate) dz on per-SN local grid
        exp_m_num = self.m0_num + self.gamma[...,None] + self.alpha*self.x0 + self.beta_int*(self.alpha_c*self.x0 + self.c0_int)  # (n_sne, N_num)
        m_resid   = value[:, 0, None] - exp_m_num                                                           # (n_sne, N_num)
        c_resid   = jnp.repeat((value[:, 1] - exp_c)[:, None], N_num, axis=1)
        x_resid   = jnp.repeat((value[:, 2] - exp_x)[:, None], N_num, axis=1)
        z_resid   = value[:, 3, None] - self.z_num                                                          # (n_sne, N_num)

        var_m_r   = jnp.repeat(var_m[:,   None], N_num, axis=1)
        var_c_r   = jnp.repeat(var_c[:,   None], N_num, axis=1)
        var_x_r   = jnp.repeat(var_x[:,   None], N_num, axis=1)
        var_z_r   = jnp.repeat(var_z[:,   None], N_num, axis=1)
        cov_m_c_r = jnp.repeat(cov_m_c[:, None], N_num, axis=1)
        cov_m_x_r = jnp.repeat(cov_m_x[:, None], N_num, axis=1)
        cov_m_z_r = jnp.repeat(cov_m_z[:, None], N_num, axis=1)
        cov_c_x_r = jnp.repeat(cov_c_x[:, None], N_num, axis=1)
        RB_r      = _bcast_sne(self.RB,  n_sne, N_num)
        tau_r     = _bcast_sne(self.tau, n_sne, N_num)

        A  = log_b4(m_resid, c_resid, x_resid, z_resid,
                    var_m_r, var_c_r, var_x_r, var_z_r,
                    cov_m_c_r, cov_m_x_r, cov_m_z_r, cov_c_x_r, RB_r, tau_r)
        A += logcdf((self.m_cut - (value[:,0] + self.a*value[:,2] + self.b*value[:,1])) / self.sigma_mcut)[:, None]
        A += logcdf((self.z_cut - value[:, 3]) / self.sigma_zcut)[:, None]

        numerator = logsumexp(A + self.log_w_num, axis=-1)                                               # (n_sne,)

        return numerator - log_denom_s
