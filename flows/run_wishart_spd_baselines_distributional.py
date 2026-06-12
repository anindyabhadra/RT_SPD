#!/usr/bin/env python3
"""
Generate a shared Wishart training/test set and run four SPD generative methods
on exactly the same data:

  1. RT split-free flow                 (subprocess: split_free_flow_rt_shared.py)
  2. RT Hamiltonian/divergence-free flow (subprocess: split_hamiltonian_flow_rt_shared.py)
  3. DiffeoCFM                          (from the DiffeoCFM repo)
  4. Riemannian SPD-CFM                 (from the DiffeoCFM repo)

Run this script from the RT repo root. The two RT flow scripts live in flows/.
Optional DiffeoCFM baseline files are expected under third_party/diffeocfm/.

Outputs are saved in --out-dir. The main table prints:
  - Maximum Mean Discrepancy (MMD) with an RBF kernel
  - Sliced Wasserstein-2
  - Classifier 2-sample test AUC
  - Run time
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

# Allow this script to be run from the repo root or directly from flows/.
_REPO_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "flows" else Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]

DIFFEOCFM_DIR = REPO_ROOT / "third_party" / "diffeocfm"
if DIFFEOCFM_DIR.exists():
    sys.path.insert(0, str(DIFFEOCFM_DIR))    

from basic_rt.rt_core import rt_volume_shape_coordinates as _rt_volume_shape_coordinates


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Wishart simulation
# ============================================================

def sample_wishart(df: int, scale: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    p = scale.shape[0]
    chol = np.linalg.cholesky(scale)
    out = np.zeros((n, p, p), dtype=np.float64)
    for i in range(n):
        z = rng.standard_normal(size=(df, p)) @ chol.T
        out[i] = z.T @ z
    return out


def sample_wishart_mixture(
    n: int,
    df: int,
    scale_1: np.ndarray,
    scale_2: np.ndarray,
    mix_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    p = scale_1.shape[0]
    out = np.zeros((n, p, p), dtype=np.float64)
    comp = rng.random(n) < mix_prob
    n1 = int(comp.sum())
    n2 = n - n1
    if n1 > 0:
        out[comp] = sample_wishart(df, scale_1, n1, rng)
    if n2 > 0:
        out[~comp] = sample_wishart(df, scale_2, n2, rng)
    return out


def make_wishart_data(
    p: int,
    df: int,
    n_train: int,
    n_test: int,
    n_holdout: int,
    target: str,
    seed: int,
    mix_prob: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)

    if target == "single":
        diag_vals = 0.5 + np.exp(rng.standard_normal(p) * 0.25)
        scale = np.diag(diag_vals)
        X_train = sample_wishart(df, scale, n_train, rng)
        X_test = sample_wishart(df, scale, n_test, rng)
        X_holdout = sample_wishart(df, scale, n_holdout, rng)
        meta = {
            "target": target,
            "p": p,
            "df": df,
            "scale_diag": diag_vals.tolist(),
        }

    elif target == "mixture":
        diag_vals_1 = np.linspace(0.8, 1.2, p)
        diag_vals_2 = np.linspace(1.5, 2.5, p)[::-1]
        scale_1 = np.diag(diag_vals_1)
        scale_2 = np.diag(diag_vals_2)
        X_train = sample_wishart_mixture(n_train, df, scale_1, scale_2, mix_prob, rng)
        X_test = sample_wishart_mixture(n_test, df, scale_1, scale_2, mix_prob, rng)
        X_holdout = sample_wishart_mixture(n_holdout, df, scale_1, scale_2, mix_prob, rng)
        meta = {
            "target": target,
            "p": p,
            "df": df,
            "mix_prob": mix_prob,
            "scale_1_diag": diag_vals_1.tolist(),
            "scale_2_diag": diag_vals_2.tolist(),
        }
    else:
        raise ValueError("target must be 'single' or 'mixture'.")

    return X_train, X_test, X_holdout, meta


# ============================================================
# Diagnostics
# ============================================================

def spd_validity(mats: np.ndarray, eps: float = 1e-10) -> dict:
    mins = []
    signs = []
    for M in mats:
        M = 0.5 * (M + M.T)
        vals = np.linalg.eigvalsh(M)
        mins.append(float(vals[0]))
        signs.append(int(np.linalg.slogdet(M)[0] > 0))
    mins = np.asarray(mins)
    signs = np.asarray(signs)
    return {
        "min_eig_min": float(mins.min()),
        "min_eig_median": float(np.median(mins)),
        "frac_spd_eps": float(np.mean(mins > eps)),
        "frac_positive_det": float(np.mean(signs)),
    }


def project_spd_numerically(mats: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    out = np.zeros_like(mats, dtype=np.float64)
    for i, M in enumerate(mats):
        M = 0.5 * (M + M.T)
        vals, vecs = np.linalg.eigh(M)
        vals = np.maximum(vals, eps)
        out[i] = vecs @ np.diag(vals) @ vecs.T
    return out


def compare_generated(X_test: np.ndarray, X_gen: np.ndarray) -> dict:
    n = min(len(X_test), len(X_gen))
    Xt = np.asarray(X_test[:n], dtype=np.float64)
    Xg = np.asarray(X_gen[:n], dtype=np.float64)
    pair_frob = np.linalg.norm(Xg - Xt, axis=(1, 2))
    mean_frob = np.linalg.norm(Xg.mean(axis=0) - Xt.mean(axis=0))
    logdet_t = np.array([np.linalg.slogdet(M)[1] for M in Xt])
    logdet_g = np.array([np.linalg.slogdet(M)[1] for M in Xg])
    trace_t = np.trace(Xt, axis1=1, axis2=2)
    trace_g = np.trace(Xg, axis1=1, axis2=2)
    return {
        "n_compare": int(n),
        "pair_frob_mean": float(pair_frob.mean()),
        "pair_frob_std": float(pair_frob.std()),
        "pair_frob_median": float(np.median(pair_frob)),
        "pair_frob_q90": float(np.quantile(pair_frob, 0.90)),
        "pair_frob_iqr": float(np.quantile(pair_frob, 0.75)-np.quantile(pair_frob, 0.25)),
        "mean_matrix_frob": float(mean_frob),
        "mean_matrix_frob_div_p": float(mean_frob / Xt.shape[1]),
        "logdet_true_mean": float(logdet_t.mean()),
        "logdet_true_median": float(np.median(logdet_t)),
        "logdet_true_std": float(logdet_t.std()),
        "logdet_gen_mean": float(logdet_g.mean()),
        "logdet_gen_median": float(np.median(logdet_g)),
        "logdet_gen_std": float(logdet_g.std()),
        "logdet_gen_iqr": float(np.quantile(logdet_g, 0.75)-np.quantile(logdet_g, 0.25)),
        "logdet_abs_err_mean": float(np.abs(logdet_g - logdet_t).mean()),
        "logdet_abs_err_median": float(np.median(np.abs(logdet_g - logdet_t))),
        "trace_true_mean": float(trace_t.mean()),
        "trace_true_std": float(trace_t.std()),
        "trace_gen_mean": float(trace_g.mean()),
        "trace_gen_std": float(trace_g.std()),
    }



def rt_volume_shape_coordinates(mats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return log-volume v and unit-determinant RT shape coordinates zeta.

    This wrapper delegates to basic_rt.rt_core so the simulation metrics and
    the learned RT flows use exactly the same encoder.
    """
    return _rt_volume_shape_coordinates(mats)

def fit_joint_feature_map(X_train: np.ndarray, n_shape_pcs: int = 5) -> dict:
    """
    Fit the common joint feature map [standardized log-volume, whitened shape PCs]
    using the training matrices only.
    """
    v_train, zeta_train = rt_volume_shape_coordinates(X_train)

    v_mean = v_train.mean(axis=0)
    v_std = np.maximum(v_train.std(axis=0), 1e-8)
    zeta_mean = zeta_train.mean(axis=0)
    zeta_std = np.maximum(zeta_train.std(axis=0), 1e-8)
    zeta_train_std = (zeta_train - zeta_mean) / zeta_std

    n_components = min(
        int(n_shape_pcs),
        zeta_train_std.shape[1],
        max(1, zeta_train_std.shape[0] - 1),
    )
    pca = PCA(n_components=n_components, whiten=True, random_state=0)
    pca.fit(zeta_train_std)

    return {
        "v_mean": v_mean,
        "v_std": v_std,
        "zeta_mean": zeta_mean,
        "zeta_std": zeta_std,
        "pca": pca,
    }


def joint_volume_shape_features(X: np.ndarray, feature_map: dict) -> np.ndarray:
    """Transform SPD matrices into the common dependence-sensitive feature space."""
    v, zeta = rt_volume_shape_coordinates(X)
    v_std = (v - feature_map["v_mean"]) / feature_map["v_std"]
    zeta_std = (zeta - feature_map["zeta_mean"]) / feature_map["zeta_std"]
    shape_pc = feature_map["pca"].transform(zeta_std)
    return np.concatenate([v_std, shape_pc], axis=1)


def _subsample_rows(X: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if len(X) <= max_rows:
        return X
    idx = rng.choice(len(X), size=max_rows, replace=False)
    return X[idx]


def _pairwise_sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    XX = np.sum(X * X, axis=1, keepdims=True)
    YY = np.sum(Y * Y, axis=1, keepdims=True).T
    return np.maximum(XX + YY - 2.0 * X @ Y.T, 0.0)


def rbf_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int,
    max_samples: int = 1500,
) -> float:
    """Biased RBF-kernel MMD with a pooled median-distance bandwidth."""
    rng = np.random.default_rng(seed)
    X = _subsample_rows(np.asarray(X, dtype=np.float64), max_samples, rng)
    Y = _subsample_rows(np.asarray(Y, dtype=np.float64), max_samples, rng)

    pooled = np.vstack([X, Y])
    bandwidth_sample = _subsample_rows(pooled, 1000, rng)
    d2_band = _pairwise_sq_dists(bandwidth_sample, bandwidth_sample)
    upper = d2_band[np.triu_indices_from(d2_band, k=1)]
    upper = upper[upper > 0]
    sigma2 = float(np.median(upper)) if upper.size else 1.0
    sigma2 = max(sigma2, 1e-12)

    Kxx = np.exp(-_pairwise_sq_dists(X, X) / (2.0 * sigma2))
    Kyy = np.exp(-_pairwise_sq_dists(Y, Y) / (2.0 * sigma2))
    Kxy = np.exp(-_pairwise_sq_dists(X, Y) / (2.0 * sigma2))
    mmd2 = float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())
    return float(np.sqrt(max(mmd2, 0.0)))


def sliced_wasserstein_2(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int,
    n_projections: int = 200,
    max_samples: int = 2000,
) -> float:
    """Monte Carlo sliced Wasserstein-2 distance in the common feature space."""
    rng = np.random.default_rng(seed)
    X = _subsample_rows(np.asarray(X, dtype=np.float64), max_samples, rng)
    Y = _subsample_rows(np.asarray(Y, dtype=np.float64), max_samples, rng)

    n = min(len(X), len(Y))
    if len(X) != n:
        X = X[rng.choice(len(X), size=n, replace=False)]
    if len(Y) != n:
        Y = Y[rng.choice(len(Y), size=n, replace=False)]

    directions = rng.standard_normal((X.shape[1], n_projections))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    proj_x = np.sort(X @ directions, axis=0)
    proj_y = np.sort(Y @ directions, axis=0)
    return float(np.sqrt(np.mean((proj_x - proj_y) ** 2)))


def c2st_auc(X: np.ndarray, Y: np.ndarray, seed: int) -> float:
    """Five-fold logistic classifier two-sample test; 0.5 is ideal."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    features = np.vstack([X, Y])
    labels = np.concatenate([
        np.zeros(len(X), dtype=np.int64),
        np.ones(len(Y), dtype=np.int64),
    ])

    classifier = make_pipeline(
        SklearnStandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs"),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    probabilities = cross_val_predict(
        classifier,
        features,
        labels,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    auc = float(roc_auc_score(labels, probabilities))
    return max(auc, 1.0 - auc)


def distributional_metrics(
    test_features: np.ndarray,
    generated_features: np.ndarray,
    seed: int,
) -> dict:
    """Dependence-sensitive metrics used in the final compact table."""
    return {
        "mmd": rbf_mmd(test_features, generated_features, seed=seed),
        "sliced_wasserstein_2": sliced_wasserstein_2(
            test_features,
            generated_features,
            seed=seed + 1,
        ),
        "c2st_auc": c2st_auc(test_features, generated_features, seed=seed + 2),
    }

def metric_arrays_for_panels(X_test: np.ndarray, X_gen: np.ndarray) -> dict:
    """
    Arrays used for the compact histogram diagnostics.

    For each method we store the pairwise Frobenius distances against X_test
    and the generated log determinants. The exact holdout row is obtained by
    passing X_gen = X_holdout.
    """
    n = min(len(X_test), len(X_gen))
    Xt = np.asarray(X_test[:n], dtype=np.float64)
    Xg = np.asarray(X_gen[:n], dtype=np.float64)

    pair_frob = np.linalg.norm(Xg - Xt, axis=(1, 2))
    logdet_g = np.array([np.linalg.slogdet(M)[1] for M in Xg], dtype=np.float64)

    return {
        "pair_frob": pair_frob,
        "logdet": logdet_g,
    }


def make_panel_histograms(hist_data: dict, out_path: Path, bins: int = 45) -> None:
    """
    Make a compact 2 x 4 panel histogram figure.

    Rows:
      1. pairwise Frobenius distances
      2. log determinants

    Columns:
      RT split-free, RT Hamiltonian, DiffeoCFM, Riemannian SPD-CFM.

    In each panel, the exact holdout distribution is shown as a gray reference
    histogram and the method distribution is shown as a blue histogram.
    """
    method_order = [
        "RT split-free",
        "RT Hamiltonian",
        "DiffeoCFM",
        "Riemannian SPD-CFM",
    ]
    method_order = [m for m in method_order if m in hist_data]
    if "Exact holdout" not in hist_data:
        raise ValueError("hist_data must contain an 'Exact holdout' entry")

    metric_info = [
        ("pair_frob", "pair Frobenius"),
        ("logdet", "log determinant"),
    ]

    n_rows = len(metric_info)
    n_cols = len(method_order)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    # Common x-limits within each row, so columns are directly comparable.
    xlims = {}
    for key, _ in metric_info:
        vals = []
        for name in ["Exact holdout"] + method_order:
            x = np.asarray(hist_data[name][key], dtype=float)
            x = x[np.isfinite(x)]
            if len(x):
                vals.append(x)
        z = np.concatenate(vals)
        lo, hi = np.quantile(z, [0.005, 0.995])
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        pad = 0.05 * (hi - lo)
        xlims[key] = (float(lo - pad), float(hi + pad))

    holdout_color = "#808080"
    method_color = "#1f77b4"

    for j, method in enumerate(method_order):
        for i, (key, row_title) in enumerate(metric_info):
            ax = axes[i, j]
            x_ref = np.asarray(hist_data["Exact holdout"][key], dtype=float)
            x_ref = x_ref[np.isfinite(x_ref)]
            x_met = np.asarray(hist_data[method][key], dtype=float)
            x_met = x_met[np.isfinite(x_met)]
            lo, hi = xlims[key]

            ax.hist(
                x_ref,
                bins=bins,
                range=(lo, hi),
                density=True,
                histtype="step",
                linewidth=1.8,
                color=holdout_color,
                label="Exact holdout",
            )
            ax.hist(
                x_met,
                bins=bins,
                range=(lo, hi),
                density=True,
                alpha=0.55,
                color=method_color,
                label=method,
            )
            ax.set_xlim(lo, hi)
            ax.grid(alpha=0.25)

            if i == 0:
                ax.set_title(method)
            if j == 0:
                ax.set_ylabel(row_title + "\ndensity")
            else:
                ax.set_ylabel("density")


    fig.suptitle(" ", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved panel histograms: {out_path.resolve()}")


def print_diagnostics(name: str, diagnostics: dict) -> None:
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    for k, v in diagnostics.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"  {kk:26s}: {vv: .6g}")
        elif isinstance(v, float):
            print(f"{k:28s}: {v: .6g}")
        else:
            print(f"{k:28s}: {v}")


def to_numpy_array(x):
    """Convert repo outputs that may be Torch tensors or NumPy arrays to NumPy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def last_time_slice(samples) -> np.ndarray:
    samples = to_numpy_array(samples)
    if samples.ndim == 4:
        return samples[-1]
    if samples.ndim == 3:
        return samples
    raise ValueError(f"Unexpected sample shape: {samples.shape}")


# ============================================================
# DiffeoCFM baselines
# ============================================================

def make_hidden_list(width: int, layers: int) -> list[int]:
    return [int(width)] * int(layers)


def make_common_config(args, device: str) -> dict:
    return {
        "FM_TYPE": args.fm_type,
        "WARMUP_EPOCHS": args.warmup_epochs,
        "FACTOR_LR": args.factor_lr,
        "LR": args.lr,
        "BATCH_SIZE": args.batch_size,
        "EPOCHS": args.epochs,
        "HIDDEN_DIM": make_hidden_list(args.hidden_dim, args.hidden_layers),
        "PRINT_EVERY": args.print_every,
        "T_GRID": torch.linspace(0, 1, args.sample_steps, device=device, dtype=torch.float64),
        "DEVICE": device,
        "RNG": np.random.RandomState(args.seed),
    }


def run_diffeocfm(X_train: np.ndarray, X_test: np.ndarray, args, device: str) -> tuple[np.ndarray, dict]:
    from fm import DiffeoCFM

    config = make_common_config(args, device)
    config["DIFFEO"] = args.diffeo
    model = DiffeoCFM(config)
    y_train = np.zeros(len(X_train), dtype=np.int64)
    y_gen = np.zeros(len(X_test), dtype=np.int64)

    t0 = time.time()
    info = model.fit(X_train, y_train)
    train_seconds = time.time() - t0

    t1 = time.time()
    samples = model.sample(y_gen)
    sample_seconds = time.time() - t1

    X_gen_raw = last_time_slice(samples)
    validity_raw = spd_validity(X_gen_raw)
    X_gen = project_spd_numerically(X_gen_raw) if args.project_baseline_output else X_gen_raw

    diagnostics = compare_generated(X_test, X_gen)
    diagnostics["method"] = "DiffeoCFM"
    diagnostics["diffeo"] = args.diffeo
    diagnostics["train_seconds"] = float(train_seconds)
    diagnostics["sample_seconds"] = float(sample_seconds)
    diagnostics["raw_spd_validity"] = validity_raw
    if isinstance(info, dict) and "train_loss" in info:
        diagnostics["final_train_loss"] = float(np.asarray(info["train_loss"])[-1])
    if isinstance(info, dict) and "val_loss" in info:
        diagnostics["final_val_loss"] = float(np.asarray(info["val_loss"])[-1])
    return X_gen, diagnostics




def patch_spd_vectorize_for_numpy() -> None:
    """
    Patch the DiffeoCFM repo's SPD.vectorize method so that SPD-CFM.fit can
    accept NumPy arrays consistently.

    The upstream SPD-CFM code calls
        X = man.vectorize(X)
    and then later
        torch.from_numpy(X_train)
    so vectorize must return a NumPy array when given a NumPy array.  However,
    spd.py's vectorize implementation expects a Torch tensor because it uses
    A.device.  This patch converts NumPy input to Torch internally and converts
    the vectorized output back to NumPy.  Torch input is left unchanged.
    """
    from spd import SPD

    if getattr(SPD, "_rt_numpy_vectorize_patch", False):
        return

    original_vectorize = SPD.vectorize

    def vectorize_numpy_aware(self, A):
        input_is_numpy = isinstance(A, np.ndarray)
        if input_is_numpy:
            A_t = torch.tensor(A, dtype=torch.float64)
            out = original_vectorize(self, A_t)
            if isinstance(out, torch.Tensor):
                return out.detach().cpu().numpy()
            return np.asarray(out)
        return original_vectorize(self, A)

    SPD.vectorize = vectorize_numpy_aware
    SPD._rt_numpy_vectorize_patch = True


def run_spd_cfm(X_train: np.ndarray, X_test: np.ndarray, args, device: str) -> tuple[np.ndarray, dict]:
    from spd_fm import SPDConditionalFlowMatching

    patch_spd_vectorize_for_numpy()

    config = make_common_config(args, device)
    # The Riemannian SPD-CFM implementation in the repo is usually more stable
    # with a smaller LR and deeper MLP. Keep command-line overrides simple.
    config["LR"] = args.spd_cfm_lr
    config["HIDDEN_DIM"] = make_hidden_list(args.spd_cfm_hidden_dim, args.spd_cfm_hidden_layers)
    config["WARMUP_EPOCHS"] = args.spd_cfm_warmup_epochs

    model = SPDConditionalFlowMatching(config)

    # Upstream SPD-CFM internally uses train_test_split and torch.from_numpy,
    # so pass NumPy arrays here.  The vectorize monkeypatch above only fixes the
    # internal vectorization step that otherwise expects Torch input.
    y_train = np.zeros(len(X_train), dtype=np.int64)
    y_gen = np.zeros(len(X_test), dtype=np.int64)

    t0 = time.time()
    info = model.fit(X_train, y_train)
    train_seconds = time.time() - t0

    t1 = time.time()
    samples = model.sample(y_gen)
    sample_seconds = time.time() - t1

    X_gen_raw = last_time_slice(samples)
    validity_raw = spd_validity(X_gen_raw)
    X_gen = project_spd_numerically(X_gen_raw) if args.project_baseline_output else X_gen_raw

    diagnostics = compare_generated(X_test, X_gen)
    diagnostics["method"] = "Riemannian SPD-CFM"
    diagnostics["train_seconds"] = float(train_seconds)
    diagnostics["sample_seconds"] = float(sample_seconds)
    diagnostics["raw_spd_validity"] = validity_raw
    if isinstance(info, dict) and "train_loss" in info:
        diagnostics["final_train_loss"] = float(np.asarray(info["train_loss"])[-1])
    if isinstance(info, dict) and "val_loss" in info:
        diagnostics["final_val_loss"] = float(np.asarray(info["val_loss"])[-1])
    return X_gen, diagnostics


# ============================================================
# RT subprocess runners
# ============================================================

def run_rt_subprocess(script_path: Path, data_npz: Path, out_npz: Path, args, method_name: str) -> tuple[np.ndarray, dict]:
    cmd = [
        sys.executable, str(script_path),
        "--data-npz", str(data_npz),
        "--out-npz", str(out_npz),
        "--seed", str(args.seed),
        "--device", args.device,
        "--hidden-v", str(args.rt_hidden_v),
        "--hidden-z", str(args.rt_hidden_z),
        "--batch-size", str(args.rt_batch_size),
        "--steps-v", str(args.rt_steps_v),
        "--steps-z", str(args.rt_steps_z),
        "--lr-v", str(args.rt_lr_v),
        "--lr-z", str(args.rt_lr_z),
        "--sample-steps-v", str(args.rt_sample_steps_v),
        "--sample-steps-z", str(args.rt_sample_steps_z),
        "--print-every", str(args.rt_print_every),
    ]
    print("\n" + "#" * 80)
    print(f"Running {method_name}")
    print(" ".join(cmd))
    print("#" * 80)
    subprocess.run(cmd, check=True)

    out = np.load(out_npz, allow_pickle=False)
    X_gen = np.asarray(out["X_gen"], dtype=np.float64)
    if "diagnostics_json" in out:
        diagnostics = json.loads(str(out["diagnostics_json"]))
    else:
        diagnostics = {}
    return X_gen, diagnostics


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--target", choices=["single", "mixture"], default="mixture")
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--df", type=int, default=70)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--n-holdout", type=int, default=None,
                        help="Number of independent exact holdout samples for truth baseline; defaults to n-test.")
    parser.add_argument("--mix-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", type=str, default="results_wishart")
    parser.add_argument("--out-prefix", type=str, default="wishart_p10_mix")

    # Method selection
    parser.add_argument("--skip-split-free", action="store_true")
    parser.add_argument("--skip-split-hamiltonian", action="store_true")
    parser.add_argument("--skip-diffeocfm", action="store_true")
    parser.add_argument("--skip-spd-cfm", action="store_true")

    # Script locations
    parser.add_argument("--split-free-script", type=str, default="flows/split_free_flow_rt_shared.py")
    parser.add_argument("--split-hamiltonian-script", type=str, default="flows/split_hamiltonian_flow_rt_shared.py")

    # Device
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    # RT options
    parser.add_argument("--rt-hidden-v", type=int, default=128)
    parser.add_argument("--rt-hidden-z", type=int, default=128)
    parser.add_argument("--rt-batch-size", type=int, default=256)
    parser.add_argument("--rt-steps-v", type=int, default=4000)
    parser.add_argument("--rt-steps-z", type=int, default=3000)
    parser.add_argument("--rt-lr-v", type=float, default=1e-3)
    parser.add_argument("--rt-lr-z", type=float, default=1e-3)
    parser.add_argument("--rt-sample-steps-v", type=int, default=100)
    parser.add_argument("--rt-sample-steps-z", type=int, default=100)
    parser.add_argument("--rt-print-every", type=int, default=250)

    # DiffeoCFM / SPD-CFM options
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--factor-lr", type=float, default=0.1)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--sample-steps", type=int, default=101)
    parser.add_argument("--diffeo", choices=["logeuclidean", "logcholesky"], default="logeuclidean")
    parser.add_argument("--fm-type", choices=["classic", "ot", "schrodinger_bridge", "variance_preserving"], default="classic")

    # SPD-CFM defaults from the repo are heavier; exposed separately.
    parser.add_argument("--spd-cfm-lr", type=float, default=1e-4)
    parser.add_argument("--spd-cfm-hidden-dim", type=int, default=128)
    parser.add_argument("--spd-cfm-hidden-layers", type=int, default=6)
    parser.add_argument("--spd-cfm-warmup-epochs", type=int, default=50)

    parser.add_argument("--no-project-baseline-output", dest="project_baseline_output", action="store_false")
    parser.set_defaults(project_baseline_output=True)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    root = Path.cwd()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nConfiguration:")
    print(json.dumps(vars(args), indent=2))
    print(f"Device for DiffeoCFM/SPD-CFM: {device}")
    print(f"Working directory: {root}")

    # Generate and save one shared train/test set, plus an independent exact holdout
    # sample used only for the "truth" baseline in the final compact table.
    n_holdout = args.n_test if args.n_holdout is None else args.n_holdout
    X_train, X_test, X_holdout, meta = make_wishart_data(
        p=args.p,
        df=args.df,
        n_train=args.n_train,
        n_test=args.n_test,
        n_holdout=n_holdout,
        target=args.target,
        seed=args.seed,
        mix_prob=args.mix_prob,
    )
    data_npz = out_dir / f"{args.out_prefix}_data.npz"
    np.savez_compressed(
        data_npz,
        X_train=X_train,
        X_test=X_test,
        X_holdout=X_holdout,
        meta_json=json.dumps(meta),
        args_json=json.dumps(vars(args)),
    )
    print(f"\nSaved shared Wishart data: {data_npz.resolve()}")
    print(f"X_train={X_train.shape}, X_test={X_test.shape}, X_holdout={X_holdout.shape}")
    print_diagnostics("Test-set SPD validity", spd_validity(X_test))
    print_diagnostics("Independent holdout SPD validity", spd_validity(X_holdout))

    # Fit one common joint volume-shape feature map on the training set.
    # All distributional comparisons below use [standardized v, five whitened
    # RT shape PCs], so the metrics are directly comparable across methods.
    joint_feature_map = fit_joint_feature_map(X_train, n_shape_pcs=5)
    test_joint_features = joint_volume_shape_features(X_test, joint_feature_map)
    holdout_joint_features = joint_volume_shape_features(X_holdout, joint_feature_map)

    # Real-vs-real baseline: compare an independent exact holdout sample to the test set
    # using the same diagnostics used for generated samples. This is the Monte Carlo
    # sampling-noise target printed in parentheses in the compact table.
    truth_baseline = compare_generated(X_test, X_holdout)
    truth_baseline.update(
        distributional_metrics(
            test_joint_features,
            holdout_joint_features,
            seed=args.seed,
        )
    )
    truth_baseline["method"] = "Exact holdout"
    print_diagnostics("TRUTH BASELINE: independent exact holdout vs test", truth_baseline)

    hist_data = {
        "Exact holdout": metric_arrays_for_panels(X_test, X_holdout),
    }

    all_diags = {
        "args": vars(args),
        "meta": meta,
        "data_npz": str(data_npz),
        "truth_baseline": truth_baseline,
    }

    # 1. RT split-free
    if not args.skip_split_free:
        script = Path(args.split_free_script)
        out_npz = out_dir / f"{args.out_prefix}_split_free.npz"
        X_gen, diag = run_rt_subprocess(script, data_npz, out_npz, args, "RT split-free flow")
        diag = compare_generated(X_test, X_gen) | {"method": "RT split-free", **diag}
        diag.update(
            distributional_metrics(
                test_joint_features,
                joint_volume_shape_features(X_gen, joint_feature_map),
                seed=args.seed,
            )
        )
        hist_data["RT split-free"] = metric_arrays_for_panels(X_test, X_gen)
        print_diagnostics("SUMMARY: RT split-free", diag)
        all_diags["split_free"] = diag

    # 2. RT Hamiltonian
    if not args.skip_split_hamiltonian:
        script = Path(args.split_hamiltonian_script)
        out_npz = out_dir / f"{args.out_prefix}_split_hamiltonian.npz"
        X_gen, diag = run_rt_subprocess(script, data_npz, out_npz, args, "RT Hamiltonian split flow")
        diag = compare_generated(X_test, X_gen) | {"method": "RT Hamiltonian", **diag}
        diag.update(
            distributional_metrics(
                test_joint_features,
                joint_volume_shape_features(X_gen, joint_feature_map),
                seed=args.seed,
            )
        )
        hist_data["RT Hamiltonian"] = metric_arrays_for_panels(X_test, X_gen)
        print_diagnostics("SUMMARY: RT Hamiltonian", diag)
        all_diags["split_hamiltonian"] = diag

    # 3. DiffeoCFM
    if not args.skip_diffeocfm:
        print("\n" + "#" * 80)
        print("Running DiffeoCFM")
        print("#" * 80)
        X_gen, diag = run_diffeocfm(X_train, X_test, args, device)
        diag.update(
            distributional_metrics(
                test_joint_features,
                joint_volume_shape_features(X_gen, joint_feature_map),
                seed=args.seed,
            )
        )
        hist_data["DiffeoCFM"] = metric_arrays_for_panels(X_test, X_gen)
        out_npz = out_dir / f"{args.out_prefix}_diffeocfm.npz"
        np.savez_compressed(out_npz, X_gen=X_gen, X_test=X_test, diagnostics_json=json.dumps(diag))
        print_diagnostics("SUMMARY: DiffeoCFM", diag)
        all_diags["diffeocfm"] = diag

    # 4. Riemannian SPD-CFM
    if not args.skip_spd_cfm:
        print("\n" + "#" * 80)
        print("Running Riemannian SPD-CFM")
        print("#" * 80)
        X_gen, diag = run_spd_cfm(X_train, X_test, args, device)
        diag.update(
            distributional_metrics(
                test_joint_features,
                joint_volume_shape_features(X_gen, joint_feature_map),
                seed=args.seed,
            )
        )
        hist_data["Riemannian SPD-CFM"] = metric_arrays_for_panels(X_test, X_gen)
        out_npz = out_dir / f"{args.out_prefix}_spd_cfm.npz"
        np.savez_compressed(out_npz, X_gen=X_gen, X_test=X_test, diagnostics_json=json.dumps(diag))
        print_diagnostics("SUMMARY: Riemannian SPD-CFM", diag)
        all_diags["spd_cfm"] = diag

    diag_path = out_dir / f"{args.out_prefix}_all_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(all_diags, f, indent=2)
    print(f"\nSaved combined diagnostics: {diag_path.resolve()}")

    # Compact table for manuscript-style reading.
    # MMD, sliced Wasserstein-2, and C2ST AUC are computed in a common
    # dependence-sensitive feature space consisting of standardized log-volume
    # and five whitened RT shape principal components.
    print("\n" + "=" * 100)
    print("COMPACT COMPARISON TABLE")
    print("First row is independent exact holdout-vs-test baseline.")
    print("Lower is better; ideal C2ST AUC is 0.5.")
    print("=" * 100)

    def time_cell(d):
        train = float(d.get("train_seconds", 0.0))
        sample = float(d.get("sample_seconds", 0.0))
        total = train + sample
        return f"{total:.4g}"

    def print_table_row(method_name, d, time_str):
        print(
            f"{method_name:24s} "
            f"{d['mmd']:12.5g} "
            f"{d['sliced_wasserstein_2']:24.5g} "
            f"{d['c2st_auc']:12.5g} "
            f"{time_str:>10s}"
        )

    header = (
        f"{'method':24s} "
        f"{'MMD':>12s} "
        f"{'Sliced-Wasserstein-2':>24s} "
        f"{'C2ST AUC':>12s} "
        f"{'time(s)':>10s}"
    )
    print(header)
    print("-" * len(header))

    tb = all_diags["truth_baseline"]
    print_table_row("Exact holdout", tb, "--")

    for key in ["split_free", "split_hamiltonian", "diffeocfm", "spd_cfm"]:
        if key not in all_diags:
            continue
        d = all_diags[key]
        print_table_row(d.get("method", key), d, time_cell(d))

    hist_path = out_dir / f"{args.out_prefix}_panel_histograms.png"
    make_panel_histograms(hist_data, hist_path)


if __name__ == "__main__":
    main()
