#!/usr/bin/env python3
"""
Core Reverse Telescoping (RT) routines for SPD matrices.

Coordinates
-----------
Raw RT coordinates are stored as a single vector

    x = (v, d_2,...,d_p, r_2,...,r_p),

where v = log det(Theta), d_j = log(tau_j) - v/p with sum_j d_j = 0,
and r_j is the raw Schur off-diagonal block at stage j.

Normalized/product coordinates are

    (v, zeta),  zeta = (d_2,...,d_p, rho_2,...,rho_p),

where rho_j = exp(-v/p) r_j are the raw RT off-diagonals of the
unit-determinant matrix exp(-v/p) Theta.

Intrinsic/regression coordinates are

    y = (v, d_2,...,d_p, beta_2,...,beta_p),

where beta_j = r_j / tau_j.

This module deliberately has no dependency on the flow or Langevin scripts.
Those scripts should import the routines here rather than duplicating them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

import numpy as np

try:  # torch is optional for the pure NumPy RT routines.
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class RTEncoding:
    """Full output of the RT encoder."""

    x: np.ndarray
    tilde: np.ndarray
    pivots: np.ndarray

    @property
    def v(self) -> float:
        return float(self.x[0])

    @property
    def p(self) -> int:
        return int(self.tilde.shape[0])

    @property
    def d(self) -> np.ndarray:
        return self.x[1:self.p]

    @property
    def r(self) -> np.ndarray:
        return self.x[self.p:]


def symmetrize(theta: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""
    theta = np.asarray(theta, dtype=np.float64)
    return 0.5 * (theta + theta.T)


def check_square(theta: np.ndarray) -> int:
    theta = np.asarray(theta)
    if theta.ndim != 2 or theta.shape[0] != theta.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {theta.shape}.")
    return int(theta.shape[0])


def num_raw_offdiag(p: int) -> int:
    return p * (p - 1) // 2


def state_dim(p: int) -> int:
    return 1 + (p - 1) + num_raw_offdiag(p)


def infer_p_from_vector(x: np.ndarray) -> int:
    n = int(np.asarray(x).size)
    p = int((math.sqrt(8 * n + 1) - 1) / 2)
    if state_dim(p) != n:
        raise ValueError(f"Vector length {n} is not p(p+1)/2 for any integer p.")
    return p


def _pivots_from_v_d(v: float, d_free: np.ndarray, p: int) -> np.ndarray:
    d_free = np.asarray(d_free, dtype=np.float64)
    if d_free.shape != (p - 1,):
        raise ValueError(f"d_free must have shape ({p-1},), got {d_free.shape}.")
    d_full = np.empty(p, dtype=np.float64)
    d_full[1:] = d_free
    d_full[0] = -float(np.sum(d_free))
    s = d_full + float(v) / p
    return np.exp(s)


def vector_to_parts(x: np.ndarray, p: Optional[int] = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """Split raw RT vector x into (v, d_free, r_flat)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if p is None:
        p = infer_p_from_vector(x)
    expected = state_dim(p)
    if x.size != expected:
        raise ValueError(f"Expected x length {expected} for p={p}, got {x.size}.")
    return float(x[0]), x[1:p].copy(), x[p:].copy()


def r_flat_to_blocks(r_flat: np.ndarray, p: int) -> List[np.ndarray]:
    """Split flattened off-diagonal blocks into [r_2, ..., r_p]."""
    r_flat = np.asarray(r_flat, dtype=np.float64).reshape(-1)
    if r_flat.size != num_raw_offdiag(p):
        raise ValueError(f"Expected {num_raw_offdiag(p)} off-diagonal entries, got {r_flat.size}.")
    blocks: List[np.ndarray] = []
    idx = 0
    for j in range(1, p):
        blocks.append(r_flat[idx:idx + j].copy())
        idx += j
    return blocks


def blocks_to_r_flat(blocks: List[np.ndarray], p: int) -> np.ndarray:
    if len(blocks) != p - 1:
        raise ValueError(f"Expected {p-1} blocks, got {len(blocks)}.")
    if not blocks:
        return np.empty(0, dtype=np.float64)
    return np.concatenate([np.asarray(b, dtype=np.float64).reshape(-1) for b in blocks])


def tilde_from_x(x: np.ndarray, p: Optional[int] = None) -> np.ndarray:
    """Construct the intermediate RT matrix \tilde{Theta} from raw x."""
    v, d_free, r_flat = vector_to_parts(x, p)
    if p is None:
        p = infer_p_from_vector(x)
    pivots = _pivots_from_v_d(v, d_free, p)
    tilde = np.zeros((p, p), dtype=np.float64)
    np.fill_diagonal(tilde, pivots)
    idx = 0
    for j in range(1, p):
        rj = r_flat[idx:idx + j]
        idx += j
        tilde[:j, j] = rj
        tilde[j, :j] = rj
    return tilde


def encode_rt_full(theta: np.ndarray, symmetrize_input: bool = True) -> RTEncoding:
    """Encode an SPD matrix into raw RT coordinates and the intermediate tilde matrix."""
    theta = np.asarray(theta, dtype=np.float64)
    p = check_square(theta)
    if symmetrize_input:
        theta = symmetrize(theta)

    omega = theta.copy()
    pivots = np.zeros(p, dtype=np.float64)
    r_blocks_reversed: List[np.ndarray] = []
    tilde = np.zeros((p, p), dtype=np.float64)

    for j in range(p - 1, 0, -1):
        rj = omega[:j, j].copy()
        tau = float(omega[j, j])
        if tau <= 0 or not np.isfinite(tau):
            raise ValueError("Encountered a non-positive Schur pivot; input may not be SPD.")
        pivots[j] = tau
        tilde[:j, j] = rj
        tilde[j, :j] = rj
        tilde[j, j] = tau
        r_blocks_reversed.append(rj)
        omega = omega[:j, :j] - np.outer(rj, rj) / tau

    tau0 = float(omega[0, 0])
    if tau0 <= 0 or not np.isfinite(tau0):
        raise ValueError("Encountered a non-positive Schur pivot; input may not be SPD.")
    pivots[0] = tau0
    tilde[0, 0] = tau0

    r_blocks = list(reversed(r_blocks_reversed))
    s = np.log(pivots)
    v = float(np.sum(s))
    d_full = s - v / p
    d_free = d_full[1:]
    r_flat = blocks_to_r_flat(r_blocks, p)
    x = np.concatenate([[v], d_free, r_flat])
    return RTEncoding(x=x, tilde=tilde, pivots=pivots)


def encode_rt(theta: np.ndarray) -> np.ndarray:
    """Encode an SPD matrix and return only the raw RT vector x."""
    return encode_rt_full(theta).x


def decode_rt(x: np.ndarray, p: Optional[int] = None) -> np.ndarray:
    """Decode raw RT coordinates x into Theta."""
    v, d_free, r_flat = vector_to_parts(x, p)
    if p is None:
        p = infer_p_from_vector(x)
    pivots = _pivots_from_v_d(v, d_free, p)
    theta = np.array([[pivots[0]]], dtype=np.float64)
    idx = 0
    for j in range(1, p):
        rj = r_flat[idx:idx + j]
        idx += j
        new_theta = np.zeros((j + 1, j + 1), dtype=np.float64)
        new_theta[:j, :j] = theta + np.outer(rj, rj) / pivots[j]
        new_theta[:j, j] = rj
        new_theta[j, :j] = rj
        new_theta[j, j] = pivots[j]
        theta = new_theta
    return theta


def decode_rt_inv(x: np.ndarray, p: Optional[int] = None) -> np.ndarray:
    """Decode raw RT coordinates x directly into Sigma = Theta^{-1}."""
    v, d_free, r_flat = vector_to_parts(x, p)
    if p is None:
        p = infer_p_from_vector(x)
    pivots = _pivots_from_v_d(v, d_free, p)
    r_blocks = r_flat_to_blocks(r_flat, p)

    sigma = np.array([[1.0 / pivots[0]]], dtype=np.float64)
    for j in range(1, p):
        rj = r_blocks[j - 1]
        beta = rj / pivots[j]
        v_col = -sigma @ beta
        sigma_ii = 1.0 / pivots[j] + beta @ sigma @ beta
        new_sigma = np.zeros((j + 1, j + 1), dtype=np.float64)
        new_sigma[:j, :j] = sigma
        new_sigma[:j, j] = v_col
        new_sigma[j, :j] = v_col
        new_sigma[j, j] = sigma_ii
        sigma = new_sigma
    return sigma


def decode_sqrt_theta(x: np.ndarray, p: Optional[int] = None) -> np.ndarray:
    """Decode an upper triangular T such that T @ T.T == Theta."""
    v, d_free, r_flat = vector_to_parts(x, p)
    if p is None:
        p = infer_p_from_vector(x)
    pivots = _pivots_from_v_d(v, d_free, p)
    r_blocks = r_flat_to_blocks(r_flat, p)
    T = np.zeros((p, p), dtype=np.float64)
    np.fill_diagonal(T, np.sqrt(pivots))
    for j in range(1, p):
        T[:j, j] = r_blocks[j - 1] / math.sqrt(pivots[j])
    return T


def decode_sqrt_precision(x: np.ndarray, p: Optional[int] = None) -> np.ndarray:
    """Decode a lower triangular M such that M @ M.T == Theta^{-1}."""
    v, d_free, r_flat = vector_to_parts(x, p)
    if p is None:
        p = infer_p_from_vector(x)
    pivots = _pivots_from_v_d(v, d_free, p)
    r_blocks = r_flat_to_blocks(r_flat, p)
    M = np.zeros((p, p), dtype=np.float64)
    M[0, 0] = 1.0 / math.sqrt(pivots[0])
    for j in range(1, p):
        beta = r_blocks[j - 1] / pivots[j]
        M[j, :j] = -beta @ M[:j, :j]
        M[j, j] = 1.0 / math.sqrt(pivots[j])
    return M


def split_volume_shape(theta: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return v=logdet(theta) and normalized shape zeta=(d,rho)."""
    theta = symmetrize(theta)
    p = theta.shape[0]
    sign, logdet = np.linalg.slogdet(theta)
    if sign <= 0:
        raise ValueError("Matrix is not SPD.")
    scale = math.exp(float(logdet) / p)
    x_unit = encode_rt(theta / scale)
    zeta = x_unit[1:]  # v_unit should be zero up to numerical error.
    return float(logdet), zeta


def combine_volume_shape(v: float, zeta: np.ndarray, p: int) -> np.ndarray:
    """Combine v and normalized shape zeta=(d,rho) into raw RT vector x=(v,d,r)."""
    zeta = np.asarray(zeta, dtype=np.float64).reshape(-1)
    expected = (p - 1) + num_raw_offdiag(p)
    if zeta.size != expected:
        raise ValueError(f"Expected zeta length {expected} for p={p}, got {zeta.size}.")
    d = zeta[:p - 1]
    rho = zeta[p - 1:]
    r = math.exp(float(v) / p) * rho
    return np.concatenate([[float(v)], d, r])


def decode_volume_shape(v: float, zeta: np.ndarray, p: int) -> np.ndarray:
    """Decode normalized/product RT coordinates (v,zeta) into Theta."""
    return decode_rt(combine_volume_shape(v, zeta, p), p)


def rt_volume_shape_coordinates(mats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Batch version of split_volume_shape for distributional diagnostics."""
    mats = np.asarray(mats, dtype=np.float64)
    if mats.ndim != 3 or mats.shape[1] != mats.shape[2]:
        raise ValueError(f"Expected shape (n,p,p), got {mats.shape}.")
    n = mats.shape[0]
    v = np.empty((n, 1), dtype=np.float64)
    zetas = []
    for i in range(n):
        vi, zi = split_volume_shape(mats[i])
        v[i, 0] = vi
        zetas.append(zi)
    return v, np.asarray(zetas, dtype=np.float64)


def encode_theta_to_y_beta(theta: np.ndarray) -> np.ndarray:
    """Encode Theta into intrinsic coordinates y=(v,d_free,beta)."""
    enc = encode_rt_full(theta)
    p = enc.p
    r_blocks = r_flat_to_blocks(enc.r, p)
    beta_blocks = [r_blocks[j - 1] / enc.pivots[j] for j in range(1, p)]
    beta_flat = blocks_to_r_flat(beta_blocks, p)
    return np.concatenate([[enc.v], enc.d, beta_flat])


def y_beta_to_x_raw(y: np.ndarray, p: Optional[int] = None) -> np.ndarray:
    """Convert y=(v,d,beta) to raw RT x=(v,d,r)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if p is None:
        p = infer_p_from_vector(y)
    v = float(y[0])
    d = y[1:p]
    beta_flat = y[p:]
    pivots = _pivots_from_v_d(v, d, p)
    beta_blocks = r_flat_to_blocks(beta_flat, p)
    r_blocks = [pivots[j] * beta_blocks[j - 1] for j in range(1, p)]
    return np.concatenate([[v], d, blocks_to_r_flat(r_blocks, p)])


def decode_y_beta(y: np.ndarray, p: Optional[int] = None) -> np.ndarray:
    """Decode intrinsic coordinates y=(v,d,beta) into Theta."""
    return decode_rt(y_beta_to_x_raw(y, p), p)


# -------------------------------------------------------------------------
# Torch helpers for intrinsic Langevin. These mirror the NumPy y=(v,d,beta)
# convention and are kept here so the Langevin script has no private RT logic.
# -------------------------------------------------------------------------

def require_torch():
    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required for the torch RT helpers.")


def unpack_state_torch(y, p: int):
    """Split torch tensor y into v, d_free, [beta_2,...,beta_p]."""
    require_torch()
    v = y[:, 0]
    d_free = y[:, 1:p]
    beta_flat = y[:, p:]
    blocks = []
    idx = 0
    for j in range(2, p + 1):
        m = j - 1
        blocks.append(beta_flat[:, idx:idx + m])
        idx += m
    return v, d_free, blocks


def compute_s_from_v_d_torch(v, d_free, p: int):
    """Convert free d=(d_2,...,d_p) to full zero-sum d and s_j=d_j+v/p."""
    require_torch()
    B = v.shape[0]
    d_full = torch.zeros(B, p, dtype=v.dtype, device=v.device)
    d_full[:, 1:] = d_free
    d_full[:, 0] = -d_free.sum(dim=1)
    s = d_full + v[:, None] / p
    return s, d_full


def build_L_from_state_torch(y, p: int):
    """
    Build lower triangular L such that Theta = L.T @ L from y=(v,d,beta).

    With tau_j=exp(s_j) and beta_j=r_j/tau_j,
        L[j,j]  = sqrt(tau_j),
        L[j,:j] = sqrt(tau_j) beta_{j+1}
    in zero-based indexing.
    """
    require_torch()
    v, d_free, beta_blocks = unpack_state_torch(y, p)
    s, d_full = compute_s_from_v_d_torch(v, d_free, p)
    sqrt_tau = torch.exp(0.5 * s)
    B = y.shape[0]
    L = torch.zeros(B, p, p, dtype=y.dtype, device=y.device)
    for j0 in range(p):
        L[:, j0, j0] = sqrt_tau[:, j0]
        if j0 > 0:
            L[:, j0, :j0] = sqrt_tau[:, j0:j0 + 1] * beta_blocks[j0 - 1]
    return L, s, d_full, beta_blocks


def theta_from_state_torch(y, p: int):
    """Decode torch y=(v,d,beta) into Theta batch."""
    L, _, _, _ = build_L_from_state_torch(y, p)
    return torch.bmm(L.transpose(1, 2), L)


def verify_basic_properties(p: int = 8, seed: int = 22, atol: float = 1e-8) -> dict:
    """Run a small numerical verification suite for the core RT identities."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((p + 3, p))
    A = Z.T @ Z + 0.25 * np.eye(p)
    enc = encode_rt_full(A)
    x = enc.x
    A_rec = decode_rt(x, p)
    A_inv_rec = decode_rt_inv(x, p)
    T = decode_sqrt_theta(x, p)
    M = decode_sqrt_precision(x, p)

    # Unit determinant path check in normalized coordinates.
    Zb = rng.standard_normal((p + 4, p))
    B = Zb.T @ Zb + 0.25 * np.eye(p)
    vA, zA = split_volume_shape(A)
    vB, zB = split_volume_shape(B)
    t = 0.37
    zC = (1.0 - t) * zA + t * zB
    C_unit = decode_volume_shape(0.0, zC, p)

    out = {
        "reconstruction_error": float(np.linalg.norm(A_rec - A)),
        "inverse_error": float(np.linalg.norm(A_inv_rec - np.linalg.inv(A))),
        "sqrt_theta_error": float(np.linalg.norm(T @ T.T - A)),
        "sqrt_precision_error": float(np.linalg.norm(M @ M.T - np.linalg.inv(A))),
        "logdet_coordinate_error": float(abs(np.linalg.slogdet(A)[1] - x[0])),
        "tilde_pivot_det_error": float(abs(np.prod(enc.pivots) - np.linalg.det(A))),
        "unit_shape_path_det_error": float(abs(np.linalg.det(C_unit) - 1.0)),
        "volume_shape_reconstruction_error": float(np.linalg.norm(decode_volume_shape(vA, zA, p) - A)),
        "y_beta_reconstruction_error": float(np.linalg.norm(decode_y_beta(encode_theta_to_y_beta(A), p) - A)),
    }
    out["passed"] = all(value < atol for key, value in out.items() if key.endswith("error"))
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(verify_basic_properties(), indent=2))
