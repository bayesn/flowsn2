"""FlowSN2 per-survey likelihood (Section 2.2, Eq. 13).

Two flows per survey:
  q1 (4D MAF) : q1(d_resid | z_j, m0_j, Sigma, sigma_z, Theta_SN) -- data likelihood
  q2 (2D MAF) : q2(Delta_mu_j, z_j | Theta_SN)                   -- selected (mu, z)

Log likelihood for SN i, with latent z integrated by quadrature:
  log L_i = logsumexp_j [log q1_ij + log wbar_ij + log P(z_j|C) + log dz_j]
          - logsumexp_j [log wbar_j^dg + log P(z_j|C) + log dz_dg]
  log wbar(z_j) = log q2(Delta_mu_j, z_j) - log P(z_j | C1) - log N(Delta_mu_j | 0, sigma_mu)

with m0_j = mu(z_j; C) + M0 + gamma * mass_mask_i and Delta_mu_j = m0_j - mu(z_j; C1) - M0_fid.

Weights are read from flow_weights/ (see load_flows):
  q1_highz, q1_lowz  -- 4D, conditioned on 19 variables
  q2_highz, q2_lowz  -- 2D, conditioned on the 10 population parameters
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats.norm import logpdf as log_norm_pdf

import equinox as eqx
import numpy as np
import yaml

from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from flowjax.bijections import Affine

from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import validate_sample

from .flowjax_patch import patched_masked_autoregressive_flow
from .cosmology import mu_func
from .redshifts import pdf_unormalised
from .paths import WEIGHTS

# Fiducial training values C1, xi_rate_fid and M0_fid (Section 3.1.2)
OM0_FID       = 0.315
W0_FID        = -1.0
BETA_RATE_FID = 1.5
M0_FID        = -19.380

_mu_fid_1d = jax.jit(jax.vmap(lambda z: mu_func(z, OM0_FID, W0_FID)))


def _mu_fid(z):
    shape = z.shape
    return _mu_fid_1d(z.ravel()).reshape(shape)


def _log_pz_fid(z):
    return jnp.log(jnp.clip(pdf_unormalised(z, OM0_FID, W0_FID, BETA_RATE_FID), 1e-30))


# ---------------------------------------------------------------------------
# Flow construction / loading
# ---------------------------------------------------------------------------
Q1_COND_DIM = 19
Q2_COND_DIM = 10


def build_q1(arch, key):
    """q1 MAF: 4D targets, 19D condition, GELU activations."""
    fn = patched_masked_autoregressive_flow if arch.get("patched") else masked_autoregressive_flow
    kw = dict(transformer=Affine()) if arch.get("patched") else {}
    return fn(
        key=key,
        base_dist=Normal(jnp.zeros(4)),
        cond_dim=Q1_COND_DIM,
        nn_activation=jax.nn.gelu,
        flow_layers=arch["no_flows"],
        nn_width=arch["nn_width"],
        nn_depth=arch["nn_depth"],
        **kw,
    )


def build_q2(arch, key):
    """q2 MAF (loop-based patched variant): 2D targets, 10D population condition."""
    return patched_masked_autoregressive_flow(
        key=key,
        base_dist=Normal(jnp.zeros(2)),
        cond_dim=arch.get("cond_dim", Q2_COND_DIM),
        nn_width=arch["nn_width"],
        nn_depth=arch["nn_depth"],
        flow_layers=arch["no_flows"],
    )


_init_key = jax.random.PRNGKey(42)   # skeleton only; overwritten by the weights


def load_flow(name, build_fn, weights_dir=WEIGHTS):
    """Deserialise flow_weights/<name>.eqx into the skeleton described by
    <name>_arch.yml. Returns (flow, mu, std, log-Jacobian of the standardisation)."""
    with open(weights_dir / f"{name}_arch.yml") as f:
        arch = yaml.safe_load(f)
    flow = eqx.tree_deserialise_leaves(str(weights_dir / f"{name}.eqx"),
                                       build_fn(arch, _init_key))
    d = np.load(weights_dir / f"{name}_std.npz")
    mu, std = jnp.array(d["mu"]), jnp.array(d["std"])
    n_tgt = 4 if build_fn is build_q1 else 2
    add = float(jnp.sum(jnp.log(std[:n_tgt])))
    return flow, mu, std, add


# survey -> (q1 tuple, q2 tuple); filled by load_flows()
_FLOWS = {}


def load_flows(q1="q1", q2="q2", weights_dir=WEIGHTS):
    """Load the four production flows {q1,q2}_{highz,lowz}. Call before MCMC."""
    for survey in ("highz", "lowz"):
        _FLOWS[survey] = (load_flow(f"{q1}_{survey}", build_q1, weights_dir),
                          load_flow(f"{q2}_{survey}", build_q2, weights_dir))
    print(f"Loaded flows: {q1}_{{highz,lowz}}, {q2}_{{highz,lowz}} from {weights_dir}")


def get_flows(lowz):
    if not _FLOWS:
        raise RuntimeError("Flows not loaded. Call load_flows() before running MCMC.")
    return _FLOWS["lowz" if lowz else "highz"]


# ---------------------------------------------------------------------------
# Flow log-prob evaluation on batched inputs
# ---------------------------------------------------------------------------
def _flow_log_prob(flow, mu_std, std_std, add, targets, conds=None):
    """log-density in the original (un-standardised) space.

    targets : (..., n_tgt) ; conds : (..., n_cond) or None. Returns (...,).
    """
    n_tgt = targets.shape[-1]

    if conds is not None:
        full   = jnp.concatenate([targets, conds], axis=-1)
        full_s = (full - mu_std) / std_std
        y = full_s[..., :n_tgt]
        c = full_s[..., n_tgt:]
    else:
        y = (targets - mu_std) / std_std
        c = None

    shape  = y.shape[:-1]
    y_flat = y.reshape(-1, n_tgt)

    if c is not None:
        c_flat     = c.reshape(-1, c.shape[-1])
        log_p_flat = jax.vmap(lambda yi, ci: flow.log_prob(yi, condition=ci))(y_flat, c_flat)
    else:
        log_p_flat = jax.vmap(lambda yi: flow.log_prob(yi))(y_flat)

    return log_p_flat.reshape(shape) - add


def _bc_par(x, n, N):
    """Broadcast a global (scalar) or per-SN (n,) parameter to (n, N). Per-SN
    arrays only occur with the host-mass dust step."""
    x = jnp.asarray(x)
    if x.ndim == 0:
        return jnp.full((n, N), x)
    return jnp.broadcast_to(x[:, None], (n, N))


# ---------------------------------------------------------------------------
# FlowSurveyLikelihood -- numpyro Distribution
# ---------------------------------------------------------------------------
class FlowSurveyLikelihood(Distribution):
    """Flow-based likelihood for one survey; ``value`` is (n_sne, 4) = [m, c, x, z].

    z_num / m0_num / log_w_num : (n_sne, N_num) per-SN numerator grid
    z_dg / log_w_dg            : (N_dg,) global denominator grid
    m0_dg                      : (2, N_dg) for mass_mask = 0, 1
    log_err_* , W_* , log_sigma_z : per-SN observational conditions (n_sne,)
    chunk_size : evaluate the numerator in blocks of this many SNe (bounded
                 memory at large N, bit-identical result); None = vectorised.
    """

    arg_constraints = {
        "z_num": constraints.real, "m0_num": constraints.real, "log_w_num": constraints.real,
        "z_dg":  constraints.real, "m0_dg":  constraints.real, "log_w_dg":  constraints.real,
        "c0_int": constraints.real, "x0": constraints.real,
        "sigma_int": constraints.real, "sigma_c": constraints.real, "sigma_x": constraints.real,
        "alpha": constraints.real, "beta_int": constraints.real, "alpha_c": constraints.real,
        "RB": constraints.real, "tau": constraints.real,
        "log_err_mm": constraints.real, "log_err_cc": constraints.real,
        "log_err_xx": constraints.real, "W_mc": constraints.real,
        "W_mx": constraints.real, "W_cx": constraints.real, "log_sigma_z": constraints.real,
        "delta_mu_sigma": constraints.positive,
    }
    support = constraints.real
    reparametrized_params = [
        "z_num", "m0_num", "log_w_num", "z_dg", "m0_dg", "log_w_dg",
        "c0_int", "x0", "sigma_int", "sigma_c", "sigma_x",
        "alpha", "beta_int", "alpha_c", "RB", "tau",
        "log_err_mm", "log_err_cc", "log_err_xx", "W_mc",
        "W_mx", "W_cx", "log_sigma_z", "delta_mu_sigma",
    ]

    def __init__(
        self,
        z_num, m0_num, log_w_num,
        z_dg,  m0_dg,  log_w_dg,
        c0_int, x0,
        sigma_int, sigma_c, sigma_x,
        alpha, beta_int, alpha_c, RB, tau,
        log_err_mm, log_err_cc, log_err_xx, W_mc,
        W_mx, W_cx, log_sigma_z,
        delta_mu_sigma=0.25,
        lowz=False,
        mass_mask=None,
        chunk_size=None,
        tau_bins=None, RB_bins=None,
        *,
        validate_args=False,
    ):
        self.lowz = lowz
        self.mass_mask = mass_mask
        (self.z_num, self.m0_num, self.log_w_num,
         self.z_dg,  self.m0_dg,  self.log_w_dg,
         self.c0_int, self.x0,
         self.sigma_int, self.sigma_c, self.sigma_x,
         self.alpha, self.beta_int, self.alpha_c, self.RB, self.tau,
         self.log_err_mm, self.log_err_cc, self.log_err_xx, self.W_mc,
         self.W_mx, self.W_cx, self.log_sigma_z,
         self.delta_mu_sigma) = (
            z_num, m0_num, log_w_num,
            z_dg,  m0_dg,  log_w_dg,
            c0_int, x0,
            sigma_int, sigma_c, sigma_x,
            alpha, beta_int, alpha_c, RB, tau,
            log_err_mm, log_err_cc, log_err_xx, W_mc,
            W_mx, W_cx, log_sigma_z,
            delta_mu_sigma,
        )
        self.chunk_size = chunk_size
        # (2,) tau/RB for the mass_mask=0/1 denominator grid (equal without the step)
        self.tau_bins = jnp.broadcast_to(jnp.asarray(tau_bins if tau_bins is not None else tau), (2,))
        self.RB_bins  = jnp.broadcast_to(jnp.asarray(RB_bins  if RB_bins  is not None else RB),  (2,))
        n_sne = log_err_mm.shape[0]
        super().__init__(
            batch_shape=(n_sne,),
            event_shape=(4,),
            validate_args=validate_args,
        )

    @validate_sample
    def log_prob(self, value):
        """value : (n_sne, 4) = [m_obs, c_obs, x_obs, z_obs] -> (n_sne,)."""
        if self.chunk_size is not None and value.shape[0] > self.chunk_size:
            return self._chunked_log_prob(value)
        n_sne = value.shape[0]
        N_num = self.z_num.shape[1]
        N_dg  = self.z_dg.shape[0]

        (flow_q1, mu_q1, std_q1, add_q1), (flow_q2, mu_q2, std_q2, add_q2) = get_flows(self.lowz)

        # Fiducial quantities (constant; no MCMC gradients flow through)
        mu_fid_num     = jax.lax.stop_gradient(_mu_fid(self.z_num))        # (n_sne, N_num)
        log_Pz_fid_num = jax.lax.stop_gradient(_log_pz_fid(self.z_num))    # (n_sne, N_num)

        # === NUMERATOR ===  q1 targets: residuals [m-m0, c-c0_int, x-x0, z_obs-z_j]
        z_resid = value[:, 3:4] - self.z_num                           # (n_sne, N_num)
        m_resid = value[:, 0:1] - self.m0_num                          # (n_sne, N_num)
        c_resid = jnp.broadcast_to(
            (value[:, 1] - self.c0_int)[:, None], (n_sne, N_num))
        x_resid = jnp.broadcast_to(
            (value[:, 2] - self.x0)[:, None],     (n_sne, N_num))

        tgt_q1_num = jnp.stack([m_resid, c_resid, x_resid, z_resid], axis=-1)  # (n_sne, N_num, 4)
        cond_q1_num = self._build_q1_cond(self.m0_num, self.z_num, n_sne, N_num)
        log_q1_num = _flow_log_prob(flow_q1, mu_q1, std_q1, add_q1,
                                    tgt_q1_num, cond_q1_num)         # (n_sne, N_num)

        # q2 at the numerator grid: targets = [delta_mu, z_j]
        delta_mu_num = self.m0_num - mu_fid_num - M0_FID                # (n_sne, N_num)
        tgt_q2_num = jnp.stack([delta_mu_num, self.z_num], axis=-1)     # (n_sne, N_num, 2)
        q2_cond = self._build_q2_cond_population(n_sne, N_num)
        log_q2_num = _flow_log_prob(flow_q2, mu_q2, std_q2, add_q2,
                                    tgt_q2_num, q2_cond)                  # (n_sne, N_num)

        # Importance weight: w_bar = q2(delta_mu, z) / [P_fid(z) * N(delta_mu | 0, sigma_mu)]
        log_gauss_num = log_norm_pdf(delta_mu_num, 0.0, self.delta_mu_sigma)
        log_wbar_num = log_q2_num - log_Pz_fid_num - log_gauss_num      # (n_sne, N_num)

        log_numerator = logsumexp(log_q1_num + log_wbar_num + self.log_w_num, axis=-1)  # (n_sne,)

        # === DENOMINATOR ===  only two cases (mass_mask = 0, 1) over (2, N_dg)
        mu_fid_dg_2     = jax.lax.stop_gradient(_mu_fid(
            jnp.broadcast_to(self.z_dg[None, :], (2, N_dg))
        ))
        log_Pz_fid_dg_2 = jax.lax.stop_gradient(_log_pz_fid(
            jnp.broadcast_to(self.z_dg[None, :], (2, N_dg))
        ))
        delta_mu_dg_2  = self.m0_dg - mu_fid_dg_2 - M0_FID              # (2, N_dg)
        z_dg_2x        = jnp.broadcast_to(self.z_dg[None, :], (2, N_dg))
        tgt_q2_dg_2    = jnp.stack([delta_mu_dg_2, z_dg_2x], axis=-1)   # (2, N_dg, 2)
        q2_cond_dg_2   = self._build_q2_cond_population(2, N_dg, self.tau_bins, self.RB_bins)
        log_q2_dg_2    = _flow_log_prob(flow_q2, mu_q2, std_q2, add_q2, tgt_q2_dg_2, q2_cond_dg_2)
        log_gauss_dg_2 = log_norm_pdf(delta_mu_dg_2, 0.0, self.delta_mu_sigma)
        log_wbar_dg_2  = log_q2_dg_2 - log_Pz_fid_dg_2 - log_gauss_dg_2

        log_denom_both  = logsumexp(log_wbar_dg_2 + self.log_w_dg[None, :], axis=-1)  # (2,)
        log_denominator = log_denom_both[self.mass_mask]                               # (n_sne,)

        return log_numerator - log_denominator

    # ------------------------------------------------------------------
    # Chunked-over-SNe path (scaling runs; bit-identical, bounded memory)
    # ------------------------------------------------------------------
    def _flow_numerator(self, value, z_num, m0_num, log_w_num,
                        log_err_mm, log_err_cc, log_err_xx, W_mc, W_mx, W_cx,
                        log_sigma_z, tau, RB,
                        flow_q1, mu_q1, std_q1, add_q1, flow_q2, mu_q2, std_q2, add_q2):
        """log numerator for a BLOCK of SNe. Per-SN arrays are arguments (mapped
        over chunks); global MCMC scalars come from self."""
        n_sne = value.shape[0]
        N_num = z_num.shape[1]
        mu_fid_num     = jax.lax.stop_gradient(_mu_fid(z_num))
        log_Pz_fid_num = jax.lax.stop_gradient(_log_pz_fid(z_num))

        z_resid = value[:, 3:4] - z_num
        m_resid = value[:, 0:1] - m0_num
        c_resid = jnp.broadcast_to((value[:, 1] - self.c0_int)[:, None], (n_sne, N_num))
        x_resid = jnp.broadcast_to((value[:, 2] - self.x0)[:, None],     (n_sne, N_num))

        tgt_q1_num  = jnp.stack([m_resid, c_resid, x_resid, z_resid], axis=-1)
        cond_q1_num = self._q1_cond_from(m0_num, z_num, log_err_mm, log_err_cc, log_err_xx,
                                         W_mc, W_mx, W_cx, log_sigma_z, tau, RB)
        log_q1_num  = _flow_log_prob(flow_q1, mu_q1, std_q1, add_q1, tgt_q1_num, cond_q1_num)

        delta_mu_num = m0_num - mu_fid_num - M0_FID
        tgt_q2_num   = jnp.stack([delta_mu_num, z_num], axis=-1)
        q2_cond      = self._build_q2_cond_population(n_sne, N_num, tau, RB)
        log_q2_num   = _flow_log_prob(flow_q2, mu_q2, std_q2, add_q2, tgt_q2_num, q2_cond)
        log_gauss_num = log_norm_pdf(delta_mu_num, 0.0, self.delta_mu_sigma)
        log_wbar_num  = log_q2_num - log_Pz_fid_num - log_gauss_num
        return logsumexp(log_q1_num + log_wbar_num + log_w_num, axis=-1)

    def _chunked_numerator(self, num_args, flows):
        """_flow_numerator in blocks of chunk_size via lax.map. The body is
        jax.checkpoint-wrapped so the backward pass re-materialises one block at a
        time. The SN axis is padded with copies of row 0 (sliced off after)."""
        cs = self.chunk_size
        n  = num_args[0].shape[0]
        n_chunks = -(-n // cs)                       # ceil
        n_pad    = n_chunks * cs
        pad      = n_pad - n

        def _pad(a):
            if pad == 0:
                return a
            reps = jnp.broadcast_to(a[:1], (pad,) + a.shape[1:])
            return jnp.concatenate([a, reps], axis=0)

        blocks = tuple(_pad(a).reshape((n_chunks, cs) + a.shape[1:]) for a in num_args)

        @jax.checkpoint
        def body(block_args):
            return self._flow_numerator(*block_args, *flows)      # (cs,)

        out = jax.lax.map(body, blocks)                            # (n_chunks, cs)
        return out.reshape(n_pad)[:n]

    def _chunked_log_prob(self, value):
        """Numerator chunked over SNe; denominator (N-independent) computed once."""
        n_sne = value.shape[0]
        N_dg  = self.z_dg.shape[0]
        (flow_q1, mu_q1, std_q1, add_q1), (flow_q2, mu_q2, std_q2, add_q2) = get_flows(self.lowz)

        # tau/RB may be global scalars or per-SN (dust step): broadcast to (n_sne,)
        tau_a = jnp.broadcast_to(jnp.asarray(self.tau), (n_sne,))
        RB_a  = jnp.broadcast_to(jnp.asarray(self.RB),  (n_sne,))
        num_args = (value, self.z_num, self.m0_num, self.log_w_num,
                    self.log_err_mm, self.log_err_cc, self.log_err_xx,
                    self.W_mc, self.W_mx, self.W_cx, self.log_sigma_z, tau_a, RB_a)
        flows = (flow_q1, mu_q1, std_q1, add_q1, flow_q2, mu_q2, std_q2, add_q2)
        log_numerator = self._chunked_numerator(num_args, flows)          # (n_sne,)

        mu_fid_dg_2     = jax.lax.stop_gradient(_mu_fid(
            jnp.broadcast_to(self.z_dg[None, :], (2, N_dg))))
        log_Pz_fid_dg_2 = jax.lax.stop_gradient(_log_pz_fid(
            jnp.broadcast_to(self.z_dg[None, :], (2, N_dg))))
        delta_mu_dg_2  = self.m0_dg - mu_fid_dg_2 - M0_FID
        z_dg_2x        = jnp.broadcast_to(self.z_dg[None, :], (2, N_dg))
        tgt_q2_dg_2    = jnp.stack([delta_mu_dg_2, z_dg_2x], axis=-1)
        q2_cond_dg_2   = self._build_q2_cond_population(2, N_dg, self.tau_bins, self.RB_bins)
        log_q2_dg_2    = _flow_log_prob(flow_q2, mu_q2, std_q2, add_q2, tgt_q2_dg_2, q2_cond_dg_2)
        log_gauss_dg_2 = log_norm_pdf(delta_mu_dg_2, 0.0, self.delta_mu_sigma)
        log_wbar_dg_2  = log_q2_dg_2 - log_Pz_fid_dg_2 - log_gauss_dg_2
        log_denom_both  = logsumexp(log_wbar_dg_2 + self.log_w_dg[None, :], axis=-1)  # (2,)
        log_denominator = log_denom_both[self.mass_mask]                               # (n_sne,)
        return log_numerator - log_denominator

    # ------------------------------------------------------------------
    # Condition vectors (training column order)
    # ------------------------------------------------------------------
    def _q1_cond_from(self, m0, z_grid, log_err_mm, log_err_cc, log_err_xx,
                      W_mc, W_mx, W_cx, log_sigma_z, tau=None, RB=None):
        """q1 19D condition from per-SN error ARGUMENTS (chunk path)."""
        n_sne, N = m0.shape

        def _bc(x):
            return jnp.full((n_sne, N), x)

        def _bc_sne(x):
            return jnp.broadcast_to(jnp.asarray(x)[:, None], (n_sne, N))

        return jnp.stack([
            m0, _bc(self.c0_int), _bc(self.x0), z_grid,
            _bc_sne(log_err_mm), _bc_sne(log_err_cc), _bc_sne(log_err_xx),
            _bc_sne(W_mc), _bc_sne(W_mx), _bc_sne(W_cx), _bc_sne(log_sigma_z),
            _bc(self.alpha), _bc(self.beta_int), _bc(self.alpha_c),
            _bc_par(self.RB  if RB  is None else RB,  n_sne, N),
            _bc_par(self.tau if tau is None else tau, n_sne, N),
            _bc(self.sigma_int), _bc(self.sigma_c), _bc(self.sigma_x),
        ], axis=-1)  # (n_sne, N, 19)

    def _build_q1_cond(self, m0, z_grid, n_sne, N):
        """q1 19D condition:
          [m0, c0_int, x0, latent_z, log_err_mm, log_err_cc, log_err_xx, W_mc,
           W_mx, W_cx, log_sigma_z, alpha, beta_int, alpha_c, RB, tau,
           sigma_int, sigma_c, sigma_x]
        """
        def _bc(x):
            return jnp.full((n_sne, N), x)

        def _bc_sne(x):
            return jnp.broadcast_to(jnp.asarray(x)[:, None], (n_sne, N))

        return jnp.stack([
            m0,
            _bc(self.c0_int),
            _bc(self.x0),
            z_grid,
            _bc_sne(self.log_err_mm),
            _bc_sne(self.log_err_cc),
            _bc_sne(self.log_err_xx),
            _bc_sne(self.W_mc),
            _bc_sne(self.W_mx),
            _bc_sne(self.W_cx),
            _bc_sne(self.log_sigma_z),
            _bc(self.alpha),
            _bc(self.beta_int),
            _bc(self.alpha_c),
            _bc_par(self.RB,  n_sne, N),
            _bc_par(self.tau, n_sne, N),
            _bc(self.sigma_int),
            _bc(self.sigma_c),
            _bc(self.sigma_x),
        ], axis=-1)  # (n_sne, N, 19)

    def _build_q2_cond_population(self, n_sne, N, tau=None, RB=None):
        """q2 10D condition (population parameters only, z-independent):
          [c0_int, x0, alpha, beta_int, alpha_c, RB, tau, sigma_int, sigma_c, sigma_x]"""
        tau = self.tau if tau is None else tau
        RB  = self.RB  if RB  is None else RB

        def _bc(x):
            return jnp.full((n_sne, N), x)

        return jnp.stack([
            _bc(self.c0_int), _bc(self.x0),
            _bc(self.alpha), _bc(self.beta_int), _bc(self.alpha_c),
            _bc_par(RB, n_sne, N), _bc_par(tau, n_sne, N), _bc(self.sigma_int),
            _bc(self.sigma_c), _bc(self.sigma_x),
        ], axis=-1)  # (n_sne, N, 10)
