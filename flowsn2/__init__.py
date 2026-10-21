"""FlowSN2: modular simulation-based inference for photometric SN Ia cosmology.

Every module runs JAX in float64, as in the paper's runs.
"""
import jax

jax.config.update("jax_enable_x64", True)
