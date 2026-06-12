#!/usr/bin/env python3
"""
rt_intrinsic_langevin_wishart_mixture_mainfig.py

Intrinsic RT Langevin sampler for a two-component Wishart mixture target,
with a main-text style visualization comparing generated samples against
independent exact mixture samples.

Target
------
For each dimension p, the target is

    0.5 * Wishart_p(delta_1, scale1)
  + 0.5 * Wishart_p(delta_2, scale2),

where in the default experiment

    scale1 = I_p,
    scale2 = I_p + 4*1 1^T / p.

where by default delta_k = p + df_extra_k.  The script uses the target density
with respect to Fisher--Rao volume, as required by the intrinsic RT Langevin
formula in Lemma 4.1 / Eqs. (22)--(24):

    B_v      = p * partial_v log pi_FR,
    B_d      = P grad_d log pi_FR,
    B_beta_j = 0.5 exp(-s_j) Theta_{j-1} grad_beta_j log pi_FR.

For a single Wishart component Wishart_p(delta, S), the density with respect to
Fisher--Rao volume is, up to a common constant independent of the component,

    log pi_FR(Theta)
      = const(delta,S) + (delta/2) log det(Theta)
        - 0.5 tr(S^{-1} Theta),

where

    const(delta,S) = log(weight) - (delta p/2) log 2
                     - (delta/2) log |S| - log Gamma_p(delta/2).

The mixture log-density is computed by log-sum-exp over the two components.

Figure
------
For p in --p-list, the script runs the intrinsic RT Langevin sampler and draws
independent exact samples from the known Wishart mixture. It then produces a
grid of plots with rows corresponding to p and columns corresponding to:

    log det(Theta), trace(Theta), lambda_min(Theta), lambda_max(Theta).

In every panel:
    - generated RT Langevin samples are shown as an empirical histogram;
    - exact mixture samples are shown as a solid KDE curve.

Example
-------
bash python rt_intrinsic_langevin_wishart_mixture_mainfig.py \
  --p-list 20 50 100\
  --df-extra1 20 \
  --df-extra2 20 \
  --n-chains 56 \
  --n-steps 20000 \
  --burn-in 10000 \
  --thin 50 \
  --dt 1e-4 \
  --n-true 8000 \
  --init mixture_means \
  --out rt_langevin_wishart_mixture_mainfig.png
  
Notes
-----
Use --init identity to test long-run mixing from the identity.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Allow this script to be run from the repo root or directly from langevin/.
_REPO_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "langevin" else Path(__file__).resolve().parent
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

def multigammaln(a: float, p: int) -> float:
    """log multivariate gamma Gamma_p(a). Requires a > (p-1)/2."""
    return (p * (p - 1) / 4.0) * math.log(math.pi) + sum(
        math.lgamma(a + (1.0 - j) / 2.0) for j in range(1, p + 1)
    )


def delta_for_p(p: int, fixed_delta: int | None, df_extra: int, name: str) -> int:
    delta = int(fixed_delta) if fixed_delta is not None else int(p + df_extra)
    if delta <= p - 1:
        raise ValueError(f"{name}={delta} must be > p-1={p-1} for Wishart_p.")
    return delta


# ============================================================
# Exact Wishart mixture sampler and features
# ============================================================

def sample_wishart_general(p: int, delta: int, scale: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample Wishart_p(delta, scale)."""
    chol = np.linalg.cholesky(scale)
    out = np.empty((n, p, p), dtype=np.float64)
    for i in range(n):
        z = rng.standard_normal(size=(delta, p)) @ chol.T
        out[i] = z.T @ z
    return out


def sample_wishart_mixture(
    p: int,
    delta1: int,
    delta2: int,
    scale1: np.ndarray,
    scale2: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample 0.5 W_p(delta1, scale1) + 0.5 W_p(delta2, scale2).

    The scale matrices are passed explicitly so the truth generation is always
    consistent with the target density used by the Langevin drift.
    """
    scale1 = np.asarray(scale1, dtype=np.float64)
    scale2 = np.asarray(scale2, dtype=np.float64)
    comp2 = rng.random(n) < 0.5
    n2 = int(comp2.sum())
    n1 = n - n2
    out = np.empty((n, p, p), dtype=np.float64)
    if n1 > 0:
        out[~comp2] = sample_wishart_general(p, delta1, scale1, n1, rng)
    if n2 > 0:
        out[comp2] = sample_wishart_general(p, delta2, scale2, n2, rng)
    return out


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
# RT y=(v,d,beta) utilities
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


def component_log_constants(
    p: int,
    delta1: int,
    delta2: int,
    scale1: np.ndarray,
    scale2: np.ndarray,
    weight1: float = 0.5,
    weight2: float = 0.5,
) -> Tuple[float, float]:
    """Component constants for the Fisher-volume Wishart mixture density.

    For W_p(delta, scale), the component density with respect to Fisher volume is
    proportional to

        exp{ c + (delta/2) log det(Theta)
                 - 0.5 tr(scale^{-1} Theta) },

    where c includes the mixture weight, the Wishart normalizing constant, and
    |scale|.  Any common constant independent of the component may be dropped;
    the expression below is sufficient for stable mixture responsibilities.
    """
    log2 = math.log(2.0)
    sign1, logdet_s1 = np.linalg.slogdet(scale1)
    sign2, logdet_s2 = np.linalg.slogdet(scale2)
    if sign1 <= 0 or sign2 <= 0:
        raise ValueError("scale1 and scale2 must be SPD.")

    c1 = (
        math.log(weight1)
        - 0.5 * delta1 * p * log2
        - 0.5 * delta1 * float(logdet_s1)
        - multigammaln(0.5 * delta1, p)
    )
    c2 = (
        math.log(weight2)
        - 0.5 * delta2 * p * log2
        - 0.5 * delta2 * float(logdet_s2)
        - multigammaln(0.5 * delta2, p)
    )
    return c1, c2


def intrinsic_langevin_drift_mixture(
    y: torch.Tensor,
    p: int,
    delta1: int,
    delta2: int,
    P: torch.Tensor,
    c1: float,
    c2: float,
    scale1_inv_t: torch.Tensor,
    scale2_inv_t: torch.Tensor,
) -> DriftParts:
    """
    Compute Eq. (22)--(24) drift for a generic two-component Wishart mixture.

    Component 1: W_p(delta1, scale1)
    Component 2: W_p(delta2, scale2)

    No rank-one or closed-form inverse identities are hardcoded here.  The terms
    tr(scale_k^{-1} Theta) are computed directly from the supplied inverse scale
    matrices, so changing scale1/scale2 in main() automatically changes both the
    exact reference sampler and the Langevin target.
    """
    y_req = y.detach().clone().requires_grad_(True)
    v, d_free, beta_blocks = unpack_state(y_req, p)
    L, s, _, beta_blocks = build_L_from_state(y_req, p)

    # Theta = L^T L.  For p up to around 50 this direct generic computation is
    # adequate and avoids hidden assumptions about the scale matrices.
    Theta = torch.bmm(L.transpose(1, 2), L)
    trace_s1_inv_theta = torch.einsum("bij,ji->b", Theta, scale1_inv_t)
    trace_s2_inv_theta = torch.einsum("bij,ji->b", Theta, scale2_inv_t)

    log_comp1 = c1 + 0.5 * delta1 * v - 0.5 * trace_s1_inv_theta
    log_comp2 = c2 + 0.5 * delta2 * v - 0.5 * trace_s2_inv_theta
    logp = torch.logsumexp(torch.stack([log_comp1, log_comp2], dim=0), dim=0)

    grad = torch.autograd.grad(logp.sum(), y_req, create_graph=False)[0]
    grad_v, grad_d, grad_beta_blocks = unpack_state(grad, p)

    drift_v = p * grad_v
    drift_d = grad_d @ P.T

    drift_beta_blocks = []
    for j0 in range(1, p):
        g = grad_beta_blocks[j0 - 1]
        Lprev = L[:, :j0, :j0]
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
    delta1: int,
    delta2: int,
    dt: float,
    P: torch.Tensor,
    P_sqrt: torch.Tensor,
    kappa: torch.Tensor,
    c1: float,
    c2: float,
    scale1_inv_t: torch.Tensor,
    scale2_inv_t: torch.Tensor,
) -> torch.Tensor:
    """One Euler--Maruyama step for Lemma 4.1 under the mixture target."""
    B = y.shape[0]
    parts = intrinsic_langevin_drift_mixture(
        y, p, delta1, delta2, P, c1, c2, scale1_inv_t, scale2_inv_t
    )
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
    delta1: int,
    delta2: int,
    scale1: np.ndarray,
    scale2: np.ndarray,
    n_chains: int,
    init: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if init == "identity":
        return torch.zeros(n_chains, state_dim(p), dtype=dtype, device=device)

    if init != "mixture_means":
        raise ValueError("init must be 'identity' or 'mixture_means'.")

    mean1 = delta1 * scale1
    mean2 = delta2 * scale2
    y1 = encode_theta_to_state_np(mean1)
    y2 = encode_theta_to_state_np(mean2)

    y = np.zeros((n_chains, state_dim(p)), dtype=np.float64)
    half = n_chains // 2
    y[:half] = y1
    y[half:] = y2
    return torch.tensor(y, dtype=dtype, device=device)


def run_rt_langevin(
    p: int,
    delta1: int,
    delta2: int,
    scale1: np.ndarray,
    scale2: np.ndarray,
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
    scale1 = np.asarray(scale1, dtype=np.float64)
    scale2 = np.asarray(scale2, dtype=np.float64)
    scale1_inv = np.linalg.inv(scale1)
    scale2_inv = np.linalg.inv(scale2)

    y = make_initial_state(p, delta1, delta2, scale1, scale2, n_chains, init, dtype, device)
    P, P_sqrt, kappa = make_P_and_kappa(p, dtype, device)
    c1, c2 = component_log_constants(p, delta1, delta2, scale1, scale2)
    scale1_inv_t = torch.tensor(scale1_inv, dtype=dtype, device=device)
    scale2_inv_t = torch.tensor(scale2_inv, dtype=dtype, device=device)

    collected = []
    start = time.time()

    for step in range(1, n_steps + 1):
        y = one_langevin_step(
            y, p, delta1, delta2, dt, P, P_sqrt, kappa, c1, c2,
            scale1_inv_t, scale2_inv_t,
        )

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
                alpha=0.35,
                label="RT Langevin samples",
            )

            dens_true = truth_kde_line(x_true, grid)
            ax.plot(grid, dens_true, linewidth=2.2, label="exact mixture reference")

            if r == 0:
                ax.set_title(label)
            if c == 0:
                ax.set_ylabel(fr"$p={p}$")
            else:
                ax.set_ylabel("density")
            ax.grid(alpha=0.25)

            m_true = np.nanmean(x_true)
            m_gen = np.nanmean(x_gen)
                     
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


def print_summary_table(results: Dict[int, Dict[str, Dict[str, np.ndarray]]]) -> None:
    print("\n" + "=" * 100)
    print("SUMMARY: exact Wishart mixture vs intrinsic RT Langevin generated samples")
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
    parser.add_argument("--df-extra1", type=int, default=20, help="Use delta1=p+df_extra1 unless --delta1 is supplied.")
    parser.add_argument("--df-extra2", type=int, default=20, help="Use delta2=p+df_extra2 unless --delta2 is supplied.")
    parser.add_argument("--delta1", type=int, default=None, help="Optional fixed delta1 for all p. Must be > p-1.")
    parser.add_argument("--delta2", type=int, default=None, help="Optional fixed delta2 for all p. Must be > p-1.")
    parser.add_argument("--n-chains", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=8000)
    parser.add_argument("--burn-in", type=int, default=4000)
    parser.add_argument("--thin", type=int, default=80)
    parser.add_argument("--dt", type=float, default=2e-5)
    parser.add_argument("--n-true", type=int, default=8000)
    parser.add_argument("--max-collected", type=int, default=8000)
    parser.add_argument("--batch-size-features", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--init", choices=["identity", "mixture_means"], default="mixture_means")
    parser.add_argument("--out", type=str, default="rt_langevin_wishart_mixture_mainfig.png")
    parser.add_argument("--bins", type=int, default=45)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = device_from_arg(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    print("\nIntrinsic RT Langevin Wishart mixture experiment")
    print("-" * 70)
    print(f"target       : 0.5 W(delta1,scale1) + 0.5 W(delta2,scale2)")
    print(f"scale1       : I_p")
    print(f"scale2       : I_p + 411^T / p")
    print(f"p-list       : {args.p_list}")
    print(f"delta1       : fixed {args.delta1} or p + {args.df_extra1}")
    print(f"delta2       : fixed {args.delta2} or p + {args.df_extra2}")
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
        delta1 = delta_for_p(p, args.delta1, args.df_extra1, "delta1")
        delta2 = delta_for_p(p, args.delta2, args.df_extra2, "delta2")
        scale1 = np.eye(p, dtype=np.float64)
        scale2 = np.eye(p, dtype=np.float64) + 4.0 * np.ones((p, p), dtype=np.float64) / p

        print(f"\nRunning p={p}, delta1={delta1}, delta2={delta2}")

        states = run_rt_langevin(
            p=p,
            delta1=delta1,
            delta2=delta2,
            scale1=scale1,
            scale2=scale2,
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

        print(f"  drawing exact mixture samples for p={p}...")
        true_mats = sample_wishart_mixture(p, delta1, delta2, scale1, scale2, args.n_true, rng)
        true_features = features_from_mats(true_mats)

        results[p] = {
            "truth": true_features,
            "generated": gen_features,
        }

    print_summary_table(results)
    make_main_figure(results, args.out, bins=args.bins)


if __name__ == "__main__":
    main()
