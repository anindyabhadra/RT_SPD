#!/usr/bin/env python3
"""
rt_intrinsic_langevin_wishart_mainfig.py

Intrinsic RT Langevin sampler for a single Wishart target, with a main-text
style visualization comparing generated samples against independent true
Wishart samples.

Target
------
    Theta ~ Wishart_p(nu, I_p / nu), so E[Theta] = I_p.

The sampler is written in the intrinsic RT coordinates y=(v,d,beta) from the
paper, Lemma 4.1 / Eqs. (22)--(24). The target density used in the Langevin
drift is the density with respect to Fisher--Rao volume:

    log pi_FR(Theta) = const + (nu/2) log det(Theta) - (nu/2) tr(Theta),

because the ordinary Wishart Lebesgue density is

    pi_Leb(Theta) propto |Theta|^{(nu-p-1)/2} exp{-(nu/2) tr(Theta)}

and dVol_FR propto |Theta|^{-(p+1)/2} dTheta.

Figure
------
For p in --p-list, the script runs the intrinsic RT Langevin sampler and draws
independent true Wishart samples. It then produces a grid of plots with rows
corresponding to p and columns corresponding to:

    log det(Theta), trace(Theta), lambda_min(Theta), lambda_max(Theta).

In every panel:
    - generated RT Langevin samples are shown as an empirical histogram;
    - true Wishart samples are shown as a solid KDE curve.

Example
-------
python rt_intrinsic_langevin_wishart_mainfig.py \
  --p-list 20 50 \
  --df-extra 20 \
  --n-chains 128 \
  --n-steps 10000 \
  --burn-in 3000 \
  --thin 50 \
  --dt 1e-4 \
  --n-true 8000 \
  --out rt_langevin_wishart_mainfig.png
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

# Allow this script to be run from the repo root or directly from langevin/.
_REPO_ROOT = (
    Path(__file__).resolve().parents[1]
    if Path(__file__).resolve().parent.name == "langevin"
    else Path(__file__).resolve().parent
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from basic_rt.rt_core import (
    num_raw_offdiag,
    state_dim as rt_state_dim,
    encode_theta_to_y_beta,
    unpack_state_torch,
    compute_s_from_v_d_torch,
    build_L_from_state_torch,
    theta_from_state_torch,
)

try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def beta_dim(p: int) -> int:
    return num_raw_offdiag(p)


def state_dim(p: int) -> int:
    # y = (v, d_2,...,d_p, beta_2,...,beta_p)
    return rt_state_dim(p)


def nu_for_p(p: int, fixed_nu: int | None, df_extra: int) -> int:
    nu = int(fixed_nu) if fixed_nu is not None else int(p + df_extra)
    if nu <= p - 1:
        raise ValueError(f"nu={nu} must be > p-1={p-1} for Wishart_p.")
    return nu


# ============================================================
# True Wishart sampler and features
# ============================================================

def sample_wishart_identity_mean(p: int, nu: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample Wishart_p(nu, I/nu), so E[Theta]=I."""
    z = rng.standard_normal(size=(n, nu, p)) / math.sqrt(nu)
    return np.einsum("nki,nkj->nij", z, z, optimize=True)


def features_from_mats(mats: np.ndarray) -> Dict[str, np.ndarray]:
    n = mats.shape[0]
    logdet = np.empty(n, dtype=np.float64)
    trace = np.empty(n, dtype=np.float64)
    lam_min = np.empty(n, dtype=np.float64)
    lam_max = np.empty(n, dtype=np.float64)

    for i in range(n):
        M = 0.5 * (mats[i] + mats[i].T)
        sign, ld = np.linalg.slogdet(M)
        if sign <= 0:
            ld = np.nan
        vals = np.linalg.eigvalsh(M)
        logdet[i] = ld
        trace[i] = np.trace(M)
        lam_min[i] = vals[0]
        lam_max[i] = vals[-1]

    return {
        "logdet": logdet,
        "trace": trace,
        "lam_min": lam_min,
        "lam_max": lam_max,
    }


# ============================================================
# RT y=(v,d,beta) utilities from shared RT core
# ============================================================

def unpack_state(y: torch.Tensor, p: int) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """Split torch y=(v,d,beta) using the shared RT core convention."""
    return unpack_state_torch(y, p)


def compute_s_from_v_d(v: torch.Tensor, d_free: torch.Tensor, p: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert free d=(d_2,...,d_p) to full zero-sum d and s_j=d_j+v/p."""
    return compute_s_from_v_d_torch(v, d_free, p)


def build_L_from_state(y: torch.Tensor, p: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """Build lower triangular L such that Theta = L^T L from shared RT core."""
    return build_L_from_state_torch(y, p)


def theta_from_state(y: torch.Tensor, p: int) -> torch.Tensor:
    """Decode torch y=(v,d,beta) into an SPD matrix batch."""
    return theta_from_state_torch(y, p)


def encode_theta_to_state_np(theta: np.ndarray) -> np.ndarray:
    """Encode an SPD matrix into y=(v,d_2,...,d_p,beta_2,...,beta_p)."""
    return encode_theta_to_y_beta(theta)


# ============================================================
# Langevin drift and step
# ============================================================

@dataclass
class DriftParts:
    drift_v: torch.Tensor
    drift_d: torch.Tensor
    drift_beta_blocks: List[torch.Tensor]
    s: torch.Tensor
    L: torch.Tensor


def intrinsic_langevin_drift(
    y: torch.Tensor,
    p: int,
    nu: int,
    P: torch.Tensor,
) -> DriftParts:
    """
    Compute the drift terms from Lemma 4.1 / Eqs. (22)--(24):

        B_v      = p * partial_v log pi_FR,
        B_d      = P grad_d log pi_FR,
        B_beta_j = 0.5 exp(-s_j) Theta_{j-1} grad_beta_j log pi_FR,

    for the single Wishart target W_p(nu, I/nu), whose log density with respect
    to Fisher--Rao volume is

        log pi_FR = const + (nu/2) (v - tr Theta).
    """
    y_req = y.detach().clone().requires_grad_(True)
    v, d_free, beta_blocks = unpack_state(y_req, p)
    L, s, _, beta_blocks = build_L_from_state(y_req, p)

    # Since Theta = L^T L, tr(Theta)=||L||_F^2.
    trace_theta = (L * L).sum(dim=(1, 2))
    logp = 0.5 * nu * (v - trace_theta)

    grad = torch.autograd.grad(logp.sum(), y_req, create_graph=False)[0]
    grad_v, grad_d, grad_beta_blocks = unpack_state(grad, p)

    drift_v = p * grad_v
    drift_d = grad_d @ P.T

    drift_beta_blocks = []
    for j0 in range(1, p):
        # beta for variable j0+1 has dimension j0 and uses Theta_{j0}.
        g = grad_beta_blocks[j0 - 1]
        Lprev = L[:, :j0, :j0]

        # row-vector multiplication by Theta_prev = Lprev^T Lprev:
        # g Theta = (g Lprev^T) Lprev.
        tmp = torch.bmm(g.unsqueeze(1), Lprev.transpose(1, 2))
        theta_g = torch.bmm(tmp, Lprev).squeeze(1)

        sj = s[:, j0]
        drift_beta = 0.5 * torch.exp(-sj)[:, None] * theta_g
        drift_beta_blocks.append(drift_beta)

    return DriftParts(
        drift_v=drift_v.detach(),
        drift_d=drift_d.detach(),
        drift_beta_blocks=[b.detach() for b in drift_beta_blocks],
        s=s.detach(),
        L=L.detach(),
    )


def one_langevin_step(
    y: torch.Tensor,
    p: int,
    nu: int,
    dt: float,
    P: torch.Tensor,
    P_sqrt: torch.Tensor,
    kappa: torch.Tensor,
) -> torch.Tensor:
    """One Euler--Maruyama step for Lemma 4.1 under the single Wishart target."""
    B = y.shape[0]
    parts = intrinsic_langevin_drift(y, p, nu, P)
    v, d_free, beta_blocks = unpack_state(y, p)

    # v update: dv = B_v dt + sqrt(2p) dW.
    v_new = v + parts.drift_v * dt + math.sqrt(2.0 * p * dt) * torch.randn_like(v)

    # d update: dd = (B_d + 1/2 kappa) dt + sqrt(2) P^{1/2} dW.
    eps_d = torch.randn_like(d_free)
    noise_d = math.sqrt(2.0 * dt) * (eps_d @ P_sqrt.T)
    d_new = d_free + (parts.drift_d + 0.5 * kappa[None, :]) * dt + noise_d

    # beta updates.
    beta_new_blocks = []
    for j0 in range(1, p):
        beta = beta_blocks[j0 - 1]
        drift_beta = parts.drift_beta_blocks[j0 - 1]
        sj = parts.s[:, j0]
        Lprev = parts.L[:, :j0, :j0]

        eps = torch.randn(B, j0, dtype=y.dtype, device=y.device)
        # row eps @ Lprev has covariance Lprev^T Lprev = Theta_prev.
        raw_noise = torch.bmm(eps.unsqueeze(1), Lprev).squeeze(1)
        noise_beta = math.sqrt(dt) * torch.exp(-0.5 * sj)[:, None] * raw_noise
        beta_new_blocks.append(beta + drift_beta * dt + noise_beta)

    y_new = torch.cat([v_new[:, None], d_new] + beta_new_blocks, dim=1)
    return y_new.detach()


# ============================================================
# Sampling and conversion to features
# ============================================================

def make_P_and_kappa(p: int, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = p - 1
    one = torch.ones(q, q, dtype=dtype, device=device)
    P = torch.eye(q, dtype=dtype, device=device) - one / p
    P_sqrt = torch.linalg.cholesky(P)
    js = torch.arange(2, p + 1, dtype=dtype, device=device)
    kappa = 2.0 * js - p - 1.0
    return P, P_sqrt, kappa


def make_initial_state(
    p: int,
    nu: int,
    n_chains: int,
    init: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Initial state for chains.

    identity and mean are identical here because W_p(nu, I/nu) has mean I_p,
    which encodes to the zero RT state.
    """
    if init == "identity":
        return torch.zeros(n_chains, state_dim(p), dtype=dtype, device=device)

    if init != "mean":
        raise ValueError("init must be 'identity' or 'mean'.")

    mean_theta = np.eye(p, dtype=np.float64)
    y0 = encode_theta_to_state_np(mean_theta)
    y = np.repeat(y0[None, :], n_chains, axis=0)
    return torch.tensor(y, dtype=dtype, device=device)


def run_rt_langevin(
    p: int,
    nu: int,
    n_chains: int,
    n_steps: int,
    burn_in: int,
    thin: int,
    dt: float,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    init: str,
    max_collected: int | None = None,
    verbose: bool = True,
) -> torch.Tensor:
    """Run intrinsic RT Langevin and return collected states."""
    torch.manual_seed(seed)

    y = make_initial_state(p, nu, n_chains, init, dtype, device)
    P, P_sqrt, kappa = make_P_and_kappa(p, dtype, device)

    collected = []
    start = time.time()

    for step in range(1, n_steps + 1):
        y = one_langevin_step(y, p, nu, dt, P, P_sqrt, kappa)

        if not torch.isfinite(y).all():
            raise FloatingPointError(f"Nonfinite state at step {step}. Try smaller --dt.")

        if step > burn_in and ((step - burn_in) % thin == 0):
            collected.append(y.detach().cpu())
            if max_collected is not None:
                n_have = sum(chunk.shape[0] for chunk in collected)
                if n_have >= max_collected:
                    break

        if verbose and (step % max(1, n_steps // 10) == 0):
            elapsed = time.time() - start
            print(f"    p={p:3d} step {step:5d}/{n_steps} | elapsed {elapsed:7.1f}s")

    if not collected:
        raise RuntimeError("No states were collected. Check --n-steps, --burn-in, and --thin.")

    states = torch.cat(collected, dim=0)
    if max_collected is not None and states.shape[0] > max_collected:
        states = states[:max_collected]

    if verbose:
        print(f"    collected {states.shape[0]} RT states for p={p}")

    return states


def features_from_states(states: torch.Tensor, p: int, batch_size: int = 512) -> Dict[str, np.ndarray]:
    """Compute logdet, trace, lambda_min, lambda_max from collected RT states."""
    device = states.device
    dtype = states.dtype
    n = states.shape[0]

    logdet_list = []
    trace_list = []
    lmin_list = []
    lmax_list = []

    for start in range(0, n, batch_size):
        yb = states[start:start + batch_size].to(device=device, dtype=dtype)
        with torch.no_grad():
            v, _, _ = unpack_state(yb, p)
            theta = theta_from_state(yb, p)
            theta = 0.5 * (theta + theta.transpose(1, 2))
            eigvals = torch.linalg.eigvalsh(theta)
            tr = theta.diagonal(dim1=1, dim2=2).sum(dim=1)

        logdet_list.append(v.detach().cpu().numpy())
        trace_list.append(tr.detach().cpu().numpy())
        lmin_list.append(eigvals[:, 0].detach().cpu().numpy())
        lmax_list.append(eigvals[:, -1].detach().cpu().numpy())

    return {
        "logdet": np.concatenate(logdet_list),
        "trace": np.concatenate(trace_list),
        "lam_min": np.concatenate(lmin_list),
        "lam_max": np.concatenate(lmax_list),
    }


# ============================================================
# Plotting
# ============================================================

STAT_INFO = [
    ("logdet", r"$\log\det(\Theta)$"),
    ("trace", r"$\mathrm{tr}(\Theta)$"),
    ("lam_min", r"$\lambda_{\min}(\Theta)$"),
    ("lam_max", r"$\lambda_{\max}(\Theta)$"),
]


def robust_range(x: np.ndarray, y: np.ndarray, q_low: float = 0.005, q_high: float = 0.995) -> Tuple[float, float]:
    z = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
    lo, hi = np.quantile(z, [q_low, q_high])
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    pad = 0.05 * (hi - lo)
    return float(lo - pad), float(hi + pad)


def truth_kde_line(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = x[np.isfinite(x)]
    if HAS_SCIPY and len(x) > 5 and np.std(x) > 0:
        kde = gaussian_kde(x)
        return kde(grid)
    hist, edges = np.histogram(x, bins=80, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.interp(grid, centers, hist, left=0.0, right=0.0)


def make_main_figure(
    results: Dict[int, Dict[str, Dict[str, np.ndarray]]],
    out_path: str,
    bins: int = 45,
    dpi: int = 300,
) -> None:
    p_list = list(results.keys())
    n_rows = len(p_list)
    n_cols = len(STAT_INFO)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.0 * n_rows), squeeze=False)

    for r, p in enumerate(p_list):
        truth = results[p]["truth"]
        gen = results[p]["generated"]

        for c, (key, label) in enumerate(STAT_INFO):
            ax = axes[r, c]
            x_true = truth[key]
            x_gen = gen[key]
            lo, hi = robust_range(x_true, x_gen)
            grid = np.linspace(lo, hi, 400)

            ax.hist(
                x_gen,
                bins=bins,
                range=(lo, hi),
                density=True,
                alpha=0.35
            )

            dens_true = truth_kde_line(x_true, grid)
            ax.plot(grid, dens_true, linewidth=2.2)

            if r == 0:
                ax.set_title(label)
            if c == 0:
                ax.set_ylabel(fr"$p={p}$")
            ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


def print_summary_table(results: Dict[int, Dict[str, Dict[str, np.ndarray]]]) -> None:
    print("\n" + "=" * 100)
    print("SUMMARY: true Wishart vs intrinsic RT Langevin generated samples")
    print("=" * 100)
    header = f"{'p':>5s} {'stat':>12s} {'true_mean':>14s} {'gen_mean':>14s} {'true_sd':>14s} {'gen_sd':>14s}"
    print(header)
    print("-" * len(header))
    for p, out in results.items():
        for key, _ in STAT_INFO:
            xt = out["truth"][key]
            xg = out["generated"][key]
            print(
                f"{p:5d} {key:>12s} "
                f"{np.nanmean(xt):14.6g} {np.nanmean(xg):14.6g} "
                f"{np.nanstd(xt):14.6g} {np.nanstd(xg):14.6g}"
            )
    print("=" * 100)


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-list", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--df-extra", type=int, default=20,
                        help="Use nu=p+df_extra unless --nu is supplied.")
    parser.add_argument("--nu", type=int, default=None,
                        help="Optional fixed nu for all p. Must be > p-1.")
    parser.add_argument("--n-chains", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--burn-in", type=int, default=2500)
    parser.add_argument("--thin", type=int, default=50)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--n-true", type=int, default=8000)
    parser.add_argument("--max-collected", type=int, default=8000)
    parser.add_argument("--batch-size-features", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--init", choices=["identity", "mean"], default="identity")
    parser.add_argument("--out", type=str, default="rt_langevin_wishart_mainfig.png")
    parser.add_argument("--bins", type=int, default=45)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = device_from_arg(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    print("\nIntrinsic RT Langevin Wishart experiment")
    print("-" * 70)
    print("target       : W_p(nu, I_p / nu), E[Theta]=I_p")
    print(f"p-list       : {args.p_list}")
    print(f"nu           : fixed {args.nu} or p + {args.df_extra}")
    print(f"init         : {args.init}")
    print(f"n_chains     : {args.n_chains}")
    print(f"n_steps      : {args.n_steps}")
    print(f"burn_in      : {args.burn_in}")
    print(f"thin         : {args.thin}")
    print(f"dt           : {args.dt}")
    print(f"n_true       : {args.n_true}")
    print(f"device/dtype : {device}/{dtype}")
    print("-" * 70)

    rng = np.random.default_rng(args.seed + 12345)
    results: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}

    for p in args.p_list:
        nu = nu_for_p(p, args.nu, args.df_extra)
        print(f"\nRunning p={p}, nu={nu}")

        states = run_rt_langevin(
            p=p,
            nu=nu,
            n_chains=args.n_chains,
            n_steps=args.n_steps,
            burn_in=args.burn_in,
            thin=args.thin,
            dt=args.dt,
            dtype=dtype,
            device=device,
            seed=args.seed + p,
            init=args.init,
            max_collected=args.max_collected,
            verbose=not args.quiet,
        )

        print(f"  computing generated features for p={p}...")
        gen_features = features_from_states(states.to(device), p, batch_size=args.batch_size_features)

        print(f"  drawing true Wishart samples for p={p}...")
        true_mats = sample_wishart_identity_mean(p, nu, args.n_true, rng)
        true_features = features_from_mats(true_mats)

        results[p] = {
            "truth": true_features,
            "generated": gen_features,
        }

    print_summary_table(results)
    make_main_figure(results, args.out, bins=args.bins)


if __name__ == "__main__":
    main()
