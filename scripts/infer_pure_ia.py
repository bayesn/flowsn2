"""Pure SN Ia inference (Section 3.1): one simulated catalogue, one posterior.

    python scripts/infer_pure_ia.py --engine flow       --cosmo cosmo2 --seed 1
    python scripts/infer_pure_ia.py --engine analytical --cosmo cosmo5_cmb_bao --seed 0
    python scripts/infer_pure_ia.py --engine naive      --cosmo cosmo2 --seed 1

engine
  flow        FlowSN2 normalising-flow likelihood
  analytical  exact likelihood (Appendix A)
  naive       analytical likelihood with selection ignored: the cuts are widened
              to m_cut = 30, z_cut = 2 in the LIKELIHOOD only, on the same SNe

For a given (cosmo, seed) every engine analyses the identical catalogue and the
identical CMB/BAO observations. Writes results/pure_ia/<engine>/<cosmo>/seed<N>/
samples.npz and skips the fit if it already exists.
"""
import argparse

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_value

from flowsn2 import experiments as ex
from flowsn2.paths import pure_ia_dir

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--engine", choices=["flow", "analytical", "naive"], default="flow")
parser.add_argument("--cosmo", default="cosmo1")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_samples", type=int, default=300)
parser.add_argument("--num_warmup", type=int, default=300)
parser.add_argument("--num_chains", type=int, default=4)
parser.add_argument("--max_tree_depth", type=int, default=6)
parser.add_argument("--sigma_pec", type=float, default=300)
parser.add_argument("--naive_m_cut", type=float, default=30.0)
parser.add_argument("--naive_z_cut", type=float, default=2.0)
args = parser.parse_args()

numpyro.set_host_device_count(args.num_chains)

output_dir = pure_ia_dir(args.engine, args.cosmo, args.seed)
npz_path = output_dir / "samples.npz"
if npz_path.exists():
    raise SystemExit(f"{npz_path} exists -- skipping")
output_dir.mkdir(parents=True, exist_ok=True)

w0wa = ex.is_w0wa(args.cosmo)
grouped_data, mcmc_key = ex.pure_ia_catalogue(args.cosmo, args.seed, sigma_pec=args.sigma_pec)

if args.engine == "naive":
    # widen the cuts AFTER simulating, so the SNe are those of the standard run
    for d in grouped_data:
        d["m_cut"] = args.naive_m_cut
        d["z_cut"] = args.naive_z_cut
    print(f"NAIVE: likelihood cuts widened to m_cut={args.naive_m_cut}, z_cut={args.naive_z_cut}")

cmb, bao_obs, cmb_cov = ex.external_observations(args.cosmo, args.seed)
init = ex.initial_values(args.cosmo, cmb, bao_obs, cmb_cov)

if args.engine == "flow":
    from flowsn2.flow_model import flow_mcmc_model as model
    from flowsn2.flow_likelihood import load_flows
    load_flows()
else:
    from flowsn2.analytical_model import mcmc_model as model

mcmc = MCMC(NUTS(model, init_strategy=init_to_value(values=init),
                 max_tree_depth=args.max_tree_depth),
            num_samples=args.num_samples, num_warmup=args.num_warmup,
            num_chains=args.num_chains, progress_bar=True)
mcmc.run(mcmc_key, grouped_data=grouped_data, w0wa=w0wa, cmb=cmb, bao_obs=bao_obs,
         sigma_pec=args.sigma_pec)
mcmc.print_summary()
np.savez_compressed(npz_path, **mcmc.get_samples())
print(f"Saved: {npz_path}")
