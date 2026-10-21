"""Patched MaskedAutoregressive bijection for 2D data on CUDA.

Root cause
----------
jax.lax.scan(length=dim) in the autoregressive inverse is compiled by XLA as
vmap(while), batching all dim steps together.  With large batches (e.g. 8092)
XLA enables CUDA graph capture, which then fails because vmap(while) creates
non-static graph shapes.  The backward pass materialises a (MADE_output, hidden)
weight-gradient GEMM; for Affine transformer MADE_output=4, giving a (4,32)
matrix that cuBLAS/CUDA graph capture rejects.

Fix summary
-----------
* _SequentialPerDimBijection  — applies scalar bijections element-wise via a
                                  static Python for-loop (no vmap over dim)
* PatchedMaskedAutoregressive  — overrides _flat_params_to_transformer and
                                  inverse_and_log_det to use Python for-loops,
                                  eliminating the vmap(while) and its GEMMs
* patched_masked_autoregressive_flow  — drop-in for masked_autoregressive_flow
"""

from collections.abc import Callable

import equinox as eqx
import jax
jax.config.update("jax_enable_x64", True)
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.masked_autoregressive import MaskedAutoregressive
from flowjax.bijections import Affine, Chain, Flip, Invert, Permute
from flowjax.distributions import Transformed


# ---------------------------------------------------------------------------
# Per-dimension bijection — static Python loop instead of vmap
# ---------------------------------------------------------------------------

class _SequentialPerDimBijection(AbstractBijection):
    """Applies dim scalar bijections element-wise using a static for-loop.

    Replaces Vmap(transformer, in_axes=eqx.if_array(0)) to avoid creating
    (dim, hidden) batched Jacobians during backpropagation.
    """

    transformers: list
    shape: tuple
    cond_shape: tuple | None

    def __init__(self, transformers):
        self.transformers = list(transformers)
        self.shape = (len(transformers),)
        self.cond_shape = None

    def transform_and_log_det(self, x, condition=None):
        ys, lds = [], []
        for i, t in enumerate(self.transformers):
            y_i, ld_i = t.transform_and_log_det(x[i])
            ys.append(y_i)
            lds.append(ld_i)
        return jnp.stack(ys), jnp.sum(jnp.stack(lds))

    def inverse_and_log_det(self, y, condition=None):
        xs, lds = [], []
        for i, t in enumerate(self.transformers):
            x_i, ld_i = t.inverse_and_log_det(y[i])
            xs.append(x_i)
            lds.append(ld_i)
        return jnp.stack(xs), jnp.sum(jnp.stack(lds))


# ---------------------------------------------------------------------------
# Patched MaskedAutoregressive
# ---------------------------------------------------------------------------

class PatchedMaskedAutoregressive(MaskedAutoregressive):
    """MaskedAutoregressive without small-dimension GEMMs.

    Two overrides:
    * _flat_params_to_transformer: Python loop instead of vmap over dim axis.
    * inverse_and_log_det: Python loop instead of jax.lax.scan(length=dim),
      which XLA compiles as vmap(while) and batches the dim iterations together,
      triggering CUDA graph capture failures with large batch sizes.
    """

    def _flat_params_to_transformer(self, params: Array):
        dim = self.shape[-1]
        num_params = params.shape[-1] // dim
        transformers = [
            self.transformer_constructor(params[i * num_params:(i + 1) * num_params])
            for i in range(dim)
        ]
        return _SequentialPerDimBijection(transformers)

    def inverse_and_log_det(self, y, condition=None):
        """Autoregressive inverse via static Python loop (no jax.lax.scan)."""
        dim = self.shape[-1]
        x = y
        num_params = None
        for rank in range(dim):
            nn_input = x if condition is None else jnp.hstack((x, condition))
            params = self.masked_autoregressive_mlp(nn_input)
            if num_params is None:
                num_params = params.shape[-1] // dim
            t = self.transformer_constructor(
                params[rank * num_params:(rank + 1) * num_params]
            )
            x = x.at[rank].set(t.inverse(x[rank]))
        log_det = self.transform_and_log_det(x, condition)[1]
        return x, -log_det


# ---------------------------------------------------------------------------
# Helper (private in flowjax.flows — replicated here)
# ---------------------------------------------------------------------------

def _add_default_permute(bijection: AbstractBijection, dim: int, key: PRNGKeyArray):
    if dim == 1:
        return bijection
    if dim == 2:
        return Chain([bijection, Flip((dim,))]).merge_chains()
    perm = Permute(jr.permutation(key, jnp.arange(dim)))
    return Chain([bijection, perm]).merge_chains()


# ---------------------------------------------------------------------------
# Drop-in replacement for masked_autoregressive_flow
# ---------------------------------------------------------------------------

def patched_masked_autoregressive_flow(
    key: PRNGKeyArray,
    *,
    base_dist,
    transformer=None,
    cond_dim: int | None = None,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    nn_activation: Callable = jnn.relu,
    invert: bool = True,
) -> Transformed:
    """Drop-in for masked_autoregressive_flow; fixes CUDA graph capture failures.

    Uses Python for-loops instead of jax.lax.scan / vmap over the dim axis,
    eliminating the vmap(while) GEMMs that trigger cuBLAS failures with large
    batch sizes.  Default transformer is Chain([Affine(), Affine()]) so
    MADE_output=8 for 2D flows, avoiding the (4,32) cuBLAS GEMM failure.
    """
    if transformer is None:
        transformer = Chain([Affine(), Affine()])

    dim = base_dist.shape[-1]

    def make_layer(key):
        bij_key, perm_key = jr.split(key)
        bijection = PatchedMaskedAutoregressive(
            key=bij_key,
            transformer=transformer,
            dim=dim,
            cond_dim=cond_dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
            nn_activation=nn_activation,
        )
        return _add_default_permute(bijection, dim, perm_key)

    keys = jr.split(key, flow_layers)
    layer_list = [make_layer(k) for k in keys]
    bijection = Chain(layer_list)
    if invert:
        bijection = Invert(bijection)
    return Transformed(base_dist, bijection)
