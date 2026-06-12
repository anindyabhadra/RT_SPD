#!/usr/bin/env python3
"""
split_free_flow_rt_conditional_shape.py

Unrestricted triangular split RT flow matching for SPD matrices. The scalar
volume flow is trained marginally, and the shape flow is conditioned on the
(final, fixed) standardized log-volume. This implements
    q(v, z) = q(v) q(z | v)
while keeping all shape trajectories on the unit-determinant RT shape space.

The command-line interface is unchanged from split_free_flow_rt_shared.py, so
run_wishart_spd_baselines_all4_v8.py can call this file without modification.

Input .npz must contain:
  X_train : (n_train, p, p) SPD matrices
  X_test  : (n_test, p, p) SPD matrices

Output .npz contains:
  X_gen   : generated SPD matrices, same count as X_test unless --n-gen is set
  X_test  : copied test matrices
  v_gen, z_gen and diagnostics_json

Example:
  python split_free_flow_rt_shared.py \
    --data-npz results_wishart/wishart_p10_mix_data.npz \
    --out-npz results_wishart/wishart_p10_mix_split_free.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Allow this script to be run from the repo root or directly from flows/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basic_rt.rt_core import decode_rt, split_volume_shape, combine_volume_shape

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

    @classmethod
    def fit(cls, z: np.ndarray, eps: float = 1e-6):
        mean = z.mean(axis=0)
        std = z.std(axis=0)
        std = np.maximum(std, eps)
        return cls(mean, std)

    def transform(self, z: np.ndarray) -> np.ndarray:
        return (z - self.mean) / self.std

    def inverse_transform(self, z_std: np.ndarray) -> np.ndarray:
        return z_std * self.std + self.mean


# ============================================================
# Torch models
# ============================================================

class VolumeFlow(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.view(1)
        if t.shape[0] == 1 and v.shape[0] > 1:
            t = t.expand(v.shape[0])
        t = t.unsqueeze(-1)
        return self.net(torch.cat([v, t], dim=-1))


class ShapeFlow(nn.Module):
    def __init__(self, dim: int, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        v_condition: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if t.ndim == 0:
            t = t.view(1)
        if t.shape[0] == 1 and z.shape[0] > 1:
            t = t.expand(z.shape[0])
        t = t.unsqueeze(-1)

        if v_condition.ndim == 0:
            v_condition = v_condition.view(1, 1)
        elif v_condition.ndim == 1:
            v_condition = v_condition.unsqueeze(-1)
        if v_condition.shape[0] == 1 and z.shape[0] > 1:
            v_condition = v_condition.expand(z.shape[0], 1)
        if v_condition.shape != (z.shape[0], 1):
            raise ValueError(
                f"v_condition must have shape ({z.shape[0]}, 1); "
                f"got {tuple(v_condition.shape)}"
            )

        return self.net(torch.cat([z, v_condition, t], dim=-1))


def fm_loss_v(model: nn.Module, v_target: torch.Tensor) -> torch.Tensor:
    batch = v_target.shape[0]
    v0 = torch.randn_like(v_target)
    t = torch.rand(batch, device=v_target.device, dtype=v_target.dtype)
    vt = (1 - t).unsqueeze(-1) * v0 + t.unsqueeze(-1) * v_target
    u_target = v_target - v0
    return ((model(vt, t) - u_target) ** 2).mean()


def fm_loss_z(
    model: nn.Module,
    z_target: torch.Tensor,
    v_condition: torch.Tensor,
) -> torch.Tensor:
    batch = z_target.shape[0]
    z0 = torch.randn_like(z_target)
    t = torch.rand(batch, device=z_target.device, dtype=z_target.dtype)
    zt = (1 - t).unsqueeze(-1) * z0 + t.unsqueeze(-1) * z_target
    u_target = z_target - z0
    return ((model(zt, v_condition, t) - u_target) ** 2).mean()


@torch.no_grad()
def sample_volume(model: nn.Module, n: int, device, dtype, steps=100) -> torch.Tensor:
    dt = 1.0 / steps
    v = torch.randn(n, 1, device=device, dtype=dtype)
    t = 0.0
    for _ in range(steps):
        t0 = torch.full((n,), t, device=device, dtype=dtype)
        k1 = model(v, t0)
        v_euler = v + dt * k1
        t1 = torch.full((n,), min(t + dt, 1.0), device=device, dtype=dtype)
        k2 = model(v_euler, t1)
        v = v + 0.5 * dt * (k1 + k2)
        t += dt
    return v


@torch.no_grad()
def sample_shape(
    model: nn.Module,
    v_condition: torch.Tensor,
    n: int,
    dim: int,
    device,
    dtype,
    steps=100,
) -> torch.Tensor:
    if v_condition.shape != (n, 1):
        raise ValueError(
            f"v_condition must have shape ({n}, 1); got {tuple(v_condition.shape)}"
        )
    dt = 1.0 / steps
    z = torch.randn(n, dim, device=device, dtype=dtype)
    t = 0.0
    for _ in range(steps):
        t0 = torch.full((n,), t, device=device, dtype=dtype)
        k1 = model(z, v_condition, t0)
        z_euler = z + dt * k1
        t1 = torch.full((n,), min(t + dt, 1.0), device=device, dtype=dtype)
        k2 = model(z_euler, v_condition, t1)
        z = z + 0.5 * dt * (k1 + k2)
        t += dt
    return z


# ============================================================
# Diagnostics
# ============================================================

def compare_generated(X_test: np.ndarray, X_gen: np.ndarray) -> dict:
    n = min(len(X_test), len(X_gen))
    Xt = X_test[:n]
    Xg = X_gen[:n]
    pair_frob = np.linalg.norm(Xg - Xt, axis=(1, 2))
    mean_frob = np.linalg.norm(Xg.mean(axis=0) - Xt.mean(axis=0))
    logdet_t = np.array([np.linalg.slogdet(m)[1] for m in Xt])
    logdet_g = np.array([np.linalg.slogdet(m)[1] for m in Xg])
    return {
        "n_compare": int(n),
        "pair_frob_mean": float(pair_frob.mean()),
        "pair_frob_std": float(pair_frob.std()),
        "pair_frob_median": float(np.median(pair_frob)),
        "mean_matrix_frob": float(mean_frob),
        "logdet_true_mean": float(logdet_t.mean()),
        "logdet_true_std": float(logdet_t.std()),
        "logdet_gen_mean": float(logdet_g.mean()),
        "logdet_gen_std": float(logdet_g.std()),
        "logdet_abs_err_mean": float(np.abs(logdet_g - logdet_t).mean()),
        "logdet_abs_err_median": float(np.median(np.abs(logdet_g - logdet_t))),
    }


def print_diagnostics(name: str, d: dict) -> None:
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    for k, v in d.items():
        print(f"{k:24s}: {v: .6g}" if isinstance(v, float) else f"{k:24s}: {v}")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-npz", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-v", type=int, default=128)
    parser.add_argument("--hidden-z", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps-v", type=int, default=4000)
    parser.add_argument("--steps-z", type=int, default=3000)
    parser.add_argument("--lr-v", type=float, default=1e-3)
    parser.add_argument("--lr-z", type=float, default=1e-3)
    parser.add_argument("--sample-steps-v", type=int, default=100)
    parser.add_argument("--sample-steps-z", type=int, default=100)
    parser.add_argument("--n-gen", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=250)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    dtype = torch.float32

    data = np.load(args.data_npz)
    X_train = np.asarray(data["X_train"], dtype=np.float64)
    X_test = np.asarray(data["X_test"], dtype=np.float64)
    n_train, p, _ = X_train.shape
    n_gen = len(X_test) if args.n_gen is None else args.n_gen

    print(f"Loaded data: X_train={X_train.shape}, X_test={X_test.shape}; device={device}")

    v_train, z_train = [], []
    for th in X_train:
        vi, zi = split_volume_shape(th)
        v_train.append(vi)
        z_train.append(zi)
    v_train = np.asarray(v_train, dtype=np.float32).reshape(-1, 1)
    z_train = np.asarray(z_train, dtype=np.float32)

    z_scaler = Standardizer.fit(z_train)
    v_scaler = Standardizer.fit(v_train)
    z_train_std = z_scaler.transform(z_train).astype(np.float32)
    v_train_std = v_scaler.transform(v_train).astype(np.float32)

    shape_dim = z_train_std.shape[1]
    print(f"p={p}, shape_dim={shape_dim}")

    v_train_t = torch.tensor(v_train_std, dtype=dtype, device=device)
    z_train_t = torch.tensor(z_train_std, dtype=dtype, device=device)

    vol_model = VolumeFlow(hidden=args.hidden_v).to(device)
    shape_model = ShapeFlow(shape_dim, hidden=args.hidden_z).to(device)
    opt_v = optim.Adam(vol_model.parameters(), lr=args.lr_v)
    opt_z = optim.Adam(shape_model.parameters(), lr=args.lr_z)

    t0 = time.time()
    print("\nTraining RT split-free volume model...")
    for step in range(args.steps_v):
        idx = np.random.choice(n_train, size=args.batch_size, replace=False)
        loss = fm_loss_v(vol_model, v_train_t[idx])
        opt_v.zero_grad(); loss.backward(); opt_v.step()
        if args.print_every and (step + 1) % args.print_every == 0:
            print(f"  [v] step {step+1:5d} | loss={loss.item():.6f}")

    print("\nTraining RT split-free conditional shape model...")
    for step in range(args.steps_z):
        idx = np.random.choice(n_train, size=args.batch_size, replace=False)
        loss = fm_loss_z(shape_model, z_train_t[idx], v_train_t[idx])
        opt_z.zero_grad(); loss.backward(); opt_z.step()
        if args.print_every and (step + 1) % args.print_every == 0:
            print(f"  [z] step {step+1:5d} | loss={loss.item():.6f}")
    train_seconds = time.time() - t0

    print("\nSampling RT split-free model...")
    t1 = time.time()
    v_gen_std_t = sample_volume(
        vol_model, n_gen, device, dtype, args.sample_steps_v
    )
    z_gen_std_t = sample_shape(
        shape_model,
        v_gen_std_t,
        n_gen,
        shape_dim,
        device,
        dtype,
        args.sample_steps_z,
    )
    v_gen_std = v_gen_std_t.cpu().numpy()
    z_gen_std = z_gen_std_t.cpu().numpy()
    sample_seconds = time.time() - t1

    v_gen = v_scaler.inverse_transform(v_gen_std)
    z_gen = z_scaler.inverse_transform(z_gen_std)

    X_gen = np.zeros((n_gen, p, p), dtype=np.float64)
    for i in range(n_gen):
        xi = combine_volume_shape(float(v_gen[i, 0]), z_gen[i], p)
        X_gen[i] = decode_rt(xi, p)

    diagnostics = compare_generated(X_test, X_gen)
    diagnostics["method"] = "RT split-free conditional shape|volume"
    diagnostics["train_seconds"] = float(train_seconds)
    diagnostics["sample_seconds"] = float(sample_seconds)
    print_diagnostics("RT split-free conditional diagnostics", diagnostics)

    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        X_gen=X_gen,
        X_test=X_test,
        v_gen=v_gen,
        z_gen=z_gen,
        diagnostics_json=json.dumps(diagnostics),
    )
    print(f"\nSaved {args.out_npz}")


if __name__ == "__main__":
    main()
