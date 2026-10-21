"""Inference under non-Ia contamination (Sections 2.4 and 3.2).

One contaminated catalogue is simulated per (cosmo, seed); only the inference is
repeated. Three analyses of the SAME catalogue, selected by the per-object
inclusion vector passed to a compile-once model:

  all_objects       every selected object, contaminants included
  pure_ia           the true SNe Ia only (true labels)
  bayes_resample<m> modular Bayesian resampling: L_s^m ~ Bernoulli(C_s^Ia),
                    m = 0..M-1; pooled -> bayes_resample_pooled.npz (Eq. 26)

The classification scores are the exact conditional class probabilities
(flowsn2.contamination), whatever --engine is used for the fits.

    python scripts/infer_contam.py --cosmo cosmo5_cmb_bao --seed 1
    python scripts/infer_contam.py --cosmo cosmo1 --seed 0 --n_resamples 10
    python scripts/infer_contam.py --cosmo cosmo1_dust --seed 3 --dust_step --resamples_only
    python scripts/infer_contam.py --cosmo cosmo1 --seed 0 --n_scale 50 --resamples_only \\
        --chunk_size 20000

Every fit is its own .npz and is skipped if present, so runs resume and can be
split across jobs with --reps A-B. Pooling always uses every
bayes_resample<m>.npz found on disk. Output:
results/contam/<engine>/<cosmo>/seed<N>[_x<n_scale>]/
"""
import argparse
import json
import re

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.random as random
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_value

from flowsn2 import experiments as ex
from flowsn2.contamination import build_catalog, DEFAULTS
from flowsn2.paths import contam_dir

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--engine", choices=["flow", "analytical"], default="flow")
parser.add_argument("--cosmo", default="cosmo1")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_resamples", type=int, default=5, help="M, the number of label allocations")
parser.add_argument("--reps", default="all",
                    help="resamples to run now: 'all', 'A-B' (inclusive; all_objects and "
                         "pure_ia run iff A == 0) or 'none' (only re-pool)")
parser.add_argument("--resamples_only", action="store_true",
                    help="skip the all_objects / pure_ia reference fits")
parser.add_argument("--dust_step", action="store_true",
                    help="split tau and RB by host mass (model-comparison runs)")
parser.add_argument("--n_scale", type=float, default=1.0,
                    help="multiply every survey's size (scaling test: 10, 50)")
parser.add_argument("--chunk_size", type=int, default=None,
                    help="flow only: evaluate the likelihood in blocks of this many SNe")
parser.add_argument("--contam_frac", type=float, default=DEFAULTS["contam_frac"],
                    help="pre-selection contaminant fraction in the high-z survey")
parser.add_argument("--mag_offset", type=float, default=DEFAULTS["mag_offset"])
parser.add_argument("--contam_beta_rate", type=float, default=DEFAULTS["contam_beta_rate"])
parser.add_argument("--sigma_pec", type=float, default=300)
parser.add_argument("--num_samples", type=int, default=300)
parser.add_argument("--num_warmup", type=int, default=300)
parser.add_argument("--num_chains", type=int, default=4)
parser.add_argument("--max_tree_depth", type=int, default=6)
args = parser.parse_args()

numpyro.set_host_device_count(args.num_chains)
output_dir = contam_dir(args.engine, args.cosmo, args.seed, args.n_scale)
output_dir.mkdir(parents=True, exist_ok=True)
w0wa = ex.is_w0wa(args.cosmo)


def parse_reps(spec, n):
    """-> (resample indices to run, run the reference fits?)"""
    if spec == "all":
        return list(range(n)), True
    if spec == "none":
        return [], False
    m = re.fullmatch(r"(\d+)-(\d+)", spec)
    if not m:
        raise SystemExit(f"bad --reps '{spec}' (use all / A-B / none)")
    a, b = int(m.group(1)), int(m.group(2))
    return list(range(a, min(b, n - 1) + 1)), a == 0


def resample_index(path):
    return int(re.search(r"bayes_resample(\d+)\.npz$", path.name).group(1))


def pool_from_disk():
    files = sorted(output_dir.glob("bayes_resample[0-9]*.npz"), key=resample_index)
    if not files:
        print("  (no resamples on disk yet)")
        return
    ds = [dict(np.load(f)) for f in files]
    keys = set.intersection(*(set(d) for d in ds))
    np.savez_compressed(output_dir / "bayes_resample_pooled.npz",
                        **{k: np.concatenate([d[k] for d in ds]) for k in keys})
    print(f"  pooled M={len(files)} resamples -> bayes_resample_pooled.npz")


reps_to_run, do_reference = parse_reps(args.reps, args.n_resamples)
if args.resamples_only:
    do_reference = False

if do_reference or reps_to_run:
    grouped, mcmc_key, meta = build_catalog(
        seed=args.seed, cosmo=args.cosmo, contam_frac=args.contam_frac,
        mag_offset=args.mag_offset, contam_beta_rate=args.contam_beta_rate,
        sigma_pec=args.sigma_pec, n_scale=args.n_scale)
    scores = [np.asarray(d["class_score"]) for d in grouped]
    is_ia = [np.asarray(d["is_ia"]) for d in grouped]

    # CMB / BAO observations are shared by every analysis of a seed
    cmb, bao_obs, cmb_cov = ex.external_observations(args.cosmo, args.seed)
    init = ex.initial_values(args.cosmo, cmb, bao_obs, cmb_cov)

    run_kwargs = dict(grouped_data=grouped, w0wa=w0wa, sigma_pec=args.sigma_pec,
                      dust_step=args.dust_step, cmb=cmb, bao_obs=bao_obs)
    if args.engine == "analytical":
        from flowsn2.analytical_model import mcmc_model as model
    else:
        from flowsn2.flow_model import flow_mcmc_model as model
        from flowsn2.flow_likelihood import load_flows
        load_flows()
        run_kwargs["chunk_size"] = args.chunk_size

    mcmc = MCMC(NUTS(model, init_strategy=init_to_value(values=init),
                     max_tree_depth=args.max_tree_depth),
                num_samples=args.num_samples, num_warmup=args.num_warmup,
                num_chains=args.num_chains, progress_bar=True)

    def fit(name, include, key):
        path = output_dir / f"{name}.npz"
        if path.exists():
            print(f"  [{name}] exists -> skip")
            return
        n_in = int(sum(v.sum() for v in include))
        print(f"\n=== {name} ({n_in} objects included) ===")
        mcmc.run(key, include=include, **run_kwargs)
        mcmc.print_summary()
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in mcmc.get_samples().items()})

    if do_reference:
        fit("all_objects", [np.ones_like(q) for q in scores], random.fold_in(mcmc_key, 1))
        fit("pure_ia", is_ia, random.fold_in(mcmc_key, 2))

    for m in reps_to_run:
        rk = random.fold_in(mcmc_key, 1000 + m)
        include = [np.asarray(random.bernoulli(random.fold_in(rk, j), q)).astype(np.float64)
                   for j, q in enumerate(scores)]
        fit(f"bayes_resample{m}", include, rk)

    with open(output_dir / "meta.json", "w") as f:
        json.dump({k: (v if not isinstance(v, dict) else {kk: float(vv) for kk, vv in v.items()})
                   for k, v in meta.items()}, f, indent=2, default=float)

pool_from_disk()
print(f"\nDone [{args.engine} {args.cosmo} seed{args.seed}] -> {output_dir}")
