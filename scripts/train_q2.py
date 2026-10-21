"""Train the 2D selection flow q2 (Section 2.5.3) for one survey.

    python scripts/train_q2.py --survey highz
    python scripts/train_q2.py --survey lowz

q2 models the selected latent (Delta_mu, z), Delta_mu = mu_t - mu(z_t; C1),
conditioned on the 10 population parameters. Writes
flow_weights/q2_<survey>{.eqx,_std.npz,_arch.yml}. Defaults are the production
architecture (2 MADE blocks, 2 x 16 hidden units), 10 epochs.
"""
import argparse

from dataclasses import dataclass

import numpy as onp
import yaml
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx
import optax
import paramax

from flowjax.distributions import Normal
from flowjax.train.losses import MaximumLikelihoodLoss
from flowjax.train.train_utils import count_fruitless, get_batches, step, train_val_split

from flowsn2.flowjax_patch import patched_masked_autoregressive_flow as masked_autoregressive_flow
from flowsn2.cosmology import mu_func
from flowsn2.flow_likelihood import OM0_FID, W0_FID, M0_FID
from flowsn2.paths import TRAINING_DATA, WEIGHTS

_mu_fid_vmap = jax.jit(jax.vmap(lambda z: mu_func(z, OM0_FID, W0_FID)))
COND_DIM = 10   # population parameters only


@dataclass
class Config:
    name: str
    survey: str
    nn_width: int
    nn_depth: int
    no_flows: int
    max_pat: int
    val_prop: float
    batch_size: int
    epochs: int
    peak_lr: float
    end_lr: float


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--survey", choices=["highz", "lowz"], required=True)
    parser.add_argument("--name", default="q2", help="weights are saved as <name>_<survey>")
    parser.add_argument("--nn_width", type=int, default=16)
    parser.add_argument("--nn_depth", type=int, default=2)
    parser.add_argument("--no_flows", type=int, default=2)
    parser.add_argument("--max_pat", type=int, default=100)
    parser.add_argument("--val_prop", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8092)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--peak_lr", type=float, default=1e-3)
    parser.add_argument("--end_lr", type=float, default=2.5e-4, help="floor of the cosine decay")
    a = parser.parse_args()
    return Config(name=f"{a.name}_{a.survey}", survey=a.survey, nn_width=a.nn_width,
                  nn_depth=a.nn_depth, no_flows=a.no_flows, max_pat=a.max_pat,
                  val_prop=a.val_prop, batch_size=a.batch_size, epochs=a.epochs,
                  peak_lr=a.peak_lr, end_lr=a.end_lr)


@jax.jit
def standardize(X):
    mu  = jnp.mean(X, axis=0)
    std = jnp.std(X,  axis=0)
    return (X - mu) / std, mu, std


def safe_log(x):
    return jnp.log(jnp.clip(x, a_min=1e-8))


def load_data(path):
    X = onp.load(path)
    log_errs = jnp.column_stack((
        safe_log(X[:, 8]),
        safe_log(X[:, 9]),
        safe_log(X[:, 10]),
        X[:, 11],            # W_mc kept raw (can be negative)
    ))
    X = jnp.column_stack((X[:, :8], log_errs, X[:, 12:]))
    X = X.at[:, 14].set(safe_log(X[:, 14]))  # log(sigma_z)
    return X


def preprocess_data(X, key):
    X = jr.permutation(key, X, axis=0)
    # targets [m0 -> Delta_mu, latent_z] first, then the conditioning columns
    X = jnp.concatenate(
        [
            X[:, 4:5],   # m0 (replaced below with Delta_mu)
            X[:, 7:8],   # latent_z
            X[:, 5:7],   # c0_int, x0
            X[:, 8:],    # errors + population parameters
        ],
        axis=1,
    )
    # Delta_mu = m0 - mu(z; C1) - M0_fid, i.e. the N(0, sigma_mu) training smearing
    mu_fid = _mu_fid_vmap(X[:, 1])
    X = X.at[:, 0].set(X[:, 0] - mu_fid - M0_FID)
    X, mu, std = standardize(X)
    add_term = jnp.sum(safe_log(std[:2]))   # 2D targets
    return X, mu, std, add_term


def build_flow(cfg: Config, key, cond_dim):
    return masked_autoregressive_flow(
        key=key,
        base_dist=Normal(jnp.zeros(2)),
        cond_dim=cond_dim,
        nn_width=cfg.nn_width,
        nn_depth=cfg.nn_depth,
        flow_layers=cfg.no_flows,
    )


def make_schedule(peak_lr=1e-3, end_lr=2.5e-4):
    warmup = optax.linear_schedule(1e-6, peak_lr, 5000)
    cosine = optax.cosine_decay_schedule(peak_lr, 100000 - 5000, alpha=end_lr / peak_lr)
    return optax.join_schedules([warmup, cosine], [5000])


def save_model(params, static, name):
    eqx.tree_serialise_leaves(str(WEIGHTS / f"{name}.eqx"), eqx.combine(params, static))


def train(key, flow, X, cfg: Config, lr_schedule, add_term, save_path):
    params, static = eqx.partition(
        flow,
        eqx.is_inexact_array,
        is_leaf=lambda x: isinstance(x, paramax.NonTrainable),
    )

    opt       = optax.adamw(lr_schedule(0), weight_decay=1e-4)
    opt_state = opt.init(params)

    logs        = {"train": [], "val": [], "lr": []}
    best_params = params

    x = X[:, :2]
    # population parameters only (cols 2,3 + 11-18): z-independent
    cond = jnp.concatenate([X[:, 2:4], X[:, 11:]], axis=1)  # 10D

    key, split_key = jr.split(key)
    train_data, val_data = train_val_split(split_key, (x, cond), cfg.val_prop)

    n_train = train_data[0].shape[0]
    print(f"  Training samples : {n_train:,}  ({max(1, n_train // cfg.batch_size)} batches)")
    print(f"  Val samples      : {val_data[0].shape[0]:,}")
    print(f"  {'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'LR':>10}")

    loss_fn = MaximumLikelihoodLoss()
    counter = 0

    for epoch in range(cfg.epochs):
        key, k1, k2 = jr.split(key, 3)

        train_data = [jr.permutation(k1, a) for a in train_data]
        val_data   = [jr.permutation(k2, a) for a in val_data]

        train_losses = []
        for batch in zip(*get_batches(train_data, cfg.batch_size), strict=True):
            key, step_key = jr.split(key)
            params, opt_state, loss = step(
                params, static, *batch,
                optimizer=opt, opt_state=opt_state,
                loss_fn=loss_fn, key=step_key,
            )
            train_losses.append(loss + add_term)
            counter += 1
            if counter % 2000 == 0:
                opt = optax.adamw(lr_schedule(counter), weight_decay=1e-4)
        train_loss = jnp.mean(jnp.array(train_losses)).item()

        val_losses = []
        for batch in zip(*get_batches(val_data, cfg.batch_size)):
            key, val_key = jr.split(key)
            loss = loss_fn(params, static, *batch, key=val_key)
            val_losses.append(loss + add_term)
        val_loss = jnp.mean(jnp.array(val_losses)).item()

        logs["train"].append(train_loss)
        logs["val"].append(val_loss)
        logs["lr"].append(float(lr_schedule(counter)))

        is_best = val_loss <= min(logs["val"])
        if is_best:
            best_params = params
        fruitless = count_fruitless(logs["val"])
        print(f"  {epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
              f"{lr_schedule(counter):>10.2e}{'  <-- best' if is_best else ''}", flush=True)

        if fruitless > cfg.max_pat:
            print(f"  Early stopping at epoch {epoch}")
            break
        save_model(best_params, static, save_path)

    save_model(best_params, static, save_path)
    print(f"  Best val loss {min(logs['val']):.4f} -> {WEIGHTS / save_path}.eqx")
    return eqx.combine(best_params, static), logs


def main():
    cfg = parse_args()
    print(f"JAX devices: {jax.devices()}  |  run: {cfg.name}")
    key = jr.key(0)

    X_raw = load_data(str(TRAINING_DATA / f"{cfg.survey}_training_set.npy"))
    key, subkey = jr.split(key)
    X, mu, std, add_term = preprocess_data(X_raw, subkey)

    WEIGHTS.mkdir(exist_ok=True)
    # targets (2) + c0_int, x0 (2) + the 8 population columns
    onp.savez(str(WEIGHTS / f"{cfg.name}_std.npz"),
              mu=onp.concatenate([mu[:4], mu[11:]]), std=onp.concatenate([std[:4], std[11:]]))
    with open(WEIGHTS / f"{cfg.name}_arch.yml", "w") as f:
        yaml.dump({"nn_width": cfg.nn_width, "nn_depth": cfg.nn_depth,
                   "no_flows": cfg.no_flows, "cond_dim": COND_DIM}, f)

    key, subkey = jr.split(key)
    flow = build_flow(cfg, subkey, cond_dim=COND_DIM)
    train(key, flow, X, cfg, make_schedule(cfg.peak_lr, cfg.end_lr), add_term, cfg.name)


if __name__ == "__main__":
    main()
