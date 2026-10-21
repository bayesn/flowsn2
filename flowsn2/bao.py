"""DESI DR2-like BAO mock prior (Appendix D.2).

Seven effective redshifts (0.295-2.33) with DESI DR2 observable types and
representative fractional precisions. The mock is self-consistent: the data
vector is drawn N(truth, C), with C built from the fractional errors times the
truth observables (block-diagonal, 2x2 per D_M/D_H tracer with its correlation),
and the model recomputes the observables at the sampled cosmology.
"""
import numpy as np
import jax.numpy as jnp

from . import cosmology as cw

# (z_eff, type, errors).  type "DM_DH": (frac_DM, frac_DH, corr);  "DV": (frac_DV,)
DESI_DR2 = [
    (0.295, "DV",    (0.0085,)),                 # BGS
    (0.510, "DM_DH", (0.012, 0.020, -0.45)),     # LRG1
    (0.706, "DM_DH", (0.010, 0.017, -0.40)),     # LRG2
    (0.934, "DM_DH", (0.007, 0.011, -0.42)),     # LRG3+ELG1
    (1.321, "DM_DH", (0.011, 0.015, -0.43)),     # ELG2
    (1.484, "DM_DH", (0.024, 0.040, -0.50)),     # QSO
    (2.330, "DM_DH", (0.014, 0.011, -0.45)),     # Lya
]


def model_vector(Om0, w0, wa, H0=70.0, data=DESI_DR2):
    """The 13 BAO observables (D_V/r_d, D_M/r_d, D_H/r_d) in DESI order."""
    rd = cw.sound_horizon_rd(Om0, H0)
    vals = []
    for z, typ, _ in data:
        if typ == "DM_DH":
            vals.append(cw.comoving_distance(z, Om0, w0, wa, H0) / rd)
            vals.append(cw.hubble_distance(z, Om0, w0, wa, H0) / rd)
        else:
            vals.append(cw.dV(z, Om0, w0, wa, H0) / rd)
    return jnp.stack(vals)


def covariance(truth_vec, data=DESI_DR2):
    """Block-diagonal covariance: fractional errors x truth observables, with
    the within-tracer D_M-D_H correlation; tracers independent."""
    v = np.asarray(truth_vec, dtype=float)
    C = np.zeros((v.size, v.size))
    i = 0
    for z, typ, e in data:
        if typ == "DM_DH":
            sDM, sDH, r = e[0] * v[i], e[1] * v[i + 1], e[2]
            C[i, i] = sDM ** 2
            C[i + 1, i + 1] = sDH ** 2
            C[i, i + 1] = C[i + 1, i] = r * sDM * sDH
            i += 2
        else:
            C[i, i] = (e[0] * v[i]) ** 2
            i += 1
    return C
