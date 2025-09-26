"""Python file to reproduce the flat-likelihood directions figure."""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# possibly change working directory to the root of the repository
# os.chdir(...)

from utils import (
    load_config_and_key,
    setup_loaders,
)

# %%
# Run experiments and specify paths
CONFIG_ONE_FIVE_ONE_RELU = 'experiments/configs/flat_likelihood/one_five_one_relu_bias.yaml' 
CONFIG_ONE_HUNDRED_ONE_RELU = 'experiments/configs/flat_likelihood/one_hundred_one_relu_bias.yaml' 

CONFIG_FIVE_FIVE_ONE_RELU = 'experiments/configs/flat_likelihood/five_five_one_relu_bias.yaml'
CONFIG_FIVE_HUNDRED_ONE_RELU = 'experiments/configs/flat_likelihood/five_hundred_one_relu_bias.yaml'

OUTPUT_PATH = 'figures/likelihood_flat_directions'

# %%
# Load configs for all three experiments
config_one_five_one, key_one_five_one = load_config_and_key(CONFIG_ONE_FIVE_ONE_RELU)
config_one_hundred_one, key_one_hundred_one = load_config_and_key(CONFIG_ONE_HUNDRED_ONE_RELU)

config_five_five_one, key_five_five_one = load_config_and_key(CONFIG_FIVE_FIVE_ONE_RELU)
config_five_hundred_one, key_five_hundred_one = load_config_and_key(CONFIG_FIVE_HUNDRED_ONE_RELU)

# Setup loaders
sample_loader_one_five_one, samples_one_five_one, data_loader_one_five_one = setup_loaders(config_one_five_one, key_one_five_one)
sample_loader_one_hundred_one, samples_one_hundred_one, data_loader_one_hundred_one = setup_loaders(config_one_hundred_one, key_one_hundred_one)

sample_loader_five_five_one, samples_five_five_one, data_loader_five_five_one = setup_loaders(config_five_five_one, key_five_five_one)
sample_loader_five_hundred_one, samples_five_hundred_one, data_loader_five_hundred_one = setup_loaders(config_five_hundred_one, key_five_hundred_one)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def get_params_for_sample(network_trace: Dict, chain_idx: int, sample_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        layer0 = network_trace['params']['fcn']['layer0']['kernel']
        layer1 = network_trace['params']['fcn']['layer1']['kernel']
        b1_arr = network_trace['params']['fcn']['layer0']['bias']

        if layer0.ndim == 4 and layer1.ndim == 4:
            w1 = layer0[chain_idx, sample_idx]  # (in_dim, hidden)
            v  = layer1[chain_idx, sample_idx]  # (hidden, out_dim)
            b1 = b1_arr[chain_idx, sample_idx]  # (hidden,)
            if w1.shape[0] == 1:
                w1 = w1[0]      # (m,)
            if v.shape[-1] == 1:
                v = v[:, 0]     # (m,)
            return w1, v, b1
        else:
            w1 = network_trace['params']['fcn']['layer0']['kernel'][chain_idx, sample_idx, 0, :]
            v  = network_trace['params']['fcn']['layer1']['kernel'][chain_idx, sample_idx, :, 0]
            b1 = network_trace['params']['fcn']['layer0']['bias'][chain_idx, sample_idx, :]
            return w1, v, b1
    except Exception as e:
        print(f"Parameter extraction error: {e}")
        return None, None, None

def gauge_fix_relu(w1: np.ndarray, b1: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if w1.ndim == 1:
        scales = np.sqrt(w1**2 + b1**2)
        scales = np.where(scales < eps, 1.0, scales)
        return w1 / scales, b1 / scales, v * scales
    elif w1.ndim == 2:
        scales = np.sqrt(np.sum(w1**2, axis=0) + b1**2)
        scales = np.where(scales < eps, 1.0, scales)
        return w1 / scales[None, :], b1 / scales, v * scales
    else:
        raise ValueError("w1 must be 1D (m,) or 2D (d,m).")

def compute_activation_matrix(X: np.ndarray, w1: np.ndarray, b1: np.ndarray, act: str = "relu") -> np.ndarray:
    if X.ndim == 1:
        Xn = X.reshape(-1)
    else:
        Xn = X
    if w1.ndim == 1:
        z = np.outer(Xn, w1) + b1
    elif w1.ndim == 2:
        z = Xn @ w1 + b1
    else:
        raise ValueError("w1 must be (m,) or (d,m).")
    if act == "relu":
        return relu(z)
    elif act == "tanh":
        return tanh(z)
    else:
        raise ValueError("act must be 'relu' or 'tanh'.")

def standardize_columns(A: np.ndarray, var_eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    A_centered = A - np.mean(A, axis=0, keepdims=True)
    norms = np.linalg.norm(A_centered, axis=0)
    active = norms > var_eps
    norms_safe = np.where(active, norms, 1.0)
    A_std = A_centered / norms_safe
    return A_std[:, active], active

def cosine_similarity_matrix(A_std: np.ndarray) -> np.ndarray:
    return A_std.T @ A_std

def nearest_neighbor_cosines(C: np.ndarray) -> np.ndarray:
    C2 = C.copy()
    np.fill_diagonal(C2, -np.inf)
    return np.max(C2, axis=1)

def cluster_from_threshold(C: np.ndarray, tau: float) -> Tuple[List[np.ndarray], int]:
    m = C.shape[0]
    adj = (C > tau).astype(np.int8)
    np.fill_diagonal(adj, 0)
    graph = csr_matrix(adj)
    n_comp, labels = connected_components(graph, directed=False, return_labels=True)
    clusters = [np.where(labels == k)[0] for k in range(n_comp)]
    D = int(np.sum([len(c) - 1 for c in clusters if len(c) > 0]))
    return clusters, D


# ---------------- Tau calibration against a null model (called rho in the paper) ---------------- #

def calibrate_tau(n: int, m: int, alpha: float = 0.01) -> float:
    """Analytic high-dim approximation: C_{jk} ~ N(0, 1/n); control family-wise error.
    tau ≈ sqrt( 2 log( m choose 2 / alpha ) / n ).
    Clipped to < 0.999 for numerical stability.
    """
    m = max(int(m), 2)
    pairs = m * (m - 1) / 2.0
    pairs = max(pairs, 1.0)
    tau = np.sqrt(2.0 * np.log(pairs / max(alpha, 1e-12)) / max(n, 1))
    return float(min(tau, 0.999))

def calibrate_tau_mc(n: int, m: int, alpha: float = 0.01, trials: int = 200, rng: Optional[np.random.Generator] = None) -> float:
    """Monte Carlo null: columns ~ iid uniform on S^{n-1}. tau = (1-alpha)-quantile of max off-diagonal cosine.
    trials is modest to keep runtime reasonable; increase for tighter estimates.
    """
    if rng is None:
        rng = np.random.default_rng()
    m = max(int(m), 2)
    max_cos = []
    for _ in range(trials):
        Z = rng.normal(size=(n, m))
        Z /= np.linalg.norm(Z, axis=0, keepdims=True) + 1e-12
        C = Z.T @ Z
        iu = np.triu_indices(m, k=1)
        max_cos.append(np.max(C[iu]))
    return float(np.quantile(np.array(max_cos), 1.0 - alpha))

# ---------- Per-sample diagnostics ----------

@dataclass
class SampleSummary:
    degeneracy: float
    rho_median: float
    rho_mean: float
    dup_fraction: float
    m_active: int
    rhos: np.ndarray
    tau_used: float

def diagnostics_for_sample(network_trace: Dict,
                           X: np.ndarray, 
                           chain_idx: int,
                           sample_idx: int,
                           act: str = "relu",
                           tau: Optional[float] = None,
                           tau_policy: str = "analytic",
                           alpha: float = 0.01,
                           gauge_fix: bool = True) -> Tuple[SampleSummary, np.ndarray, np.ndarray]:
    """Return (summary, A_std, A_raw). If tau is None, calibrate per-sample based on m_active."""
    w1, v, b1 = get_params_for_sample(network_trace, chain_idx, sample_idx)
    if gauge_fix and act == "relu":
        w1, b1, v = gauge_fix_relu(w1, b1, v)

    A = compute_activation_matrix(X, w1, b1, act=act)
    A_std, active = standardize_columns(A)

    m_active = A_std.shape[1]
    if tau is None:
        if tau_policy == "analytic":
            tau_used = calibrate_tau(n=A_std.shape[0], m=m_active, alpha=alpha) if m_active >= 2 else 0.0
        elif tau_policy == "mc":
            tau_used = calibrate_tau_mc(n=A_std.shape[0], m=m_active, alpha=alpha) if m_active >= 2 else 0.0
        else:
            raise ValueError("tau_policy must be 'analytic' or 'mc'.")
    else:
        tau_used = float(tau)

    if m_active == 0:
        summary = SampleSummary(degeneracy=0.0, rho_median=np.nan, rho_mean=np.nan,
                                dup_fraction=0.0, r_eff=0.0, m_active=0, rhos=np.array([]), tau_used=tau_used)
        return summary, A_std, A

    C = cosine_similarity_matrix(A_std)
    rhos = nearest_neighbor_cosines(C)
    dup_fraction = float(np.mean(rhos > tau_used))
    clusters, D = cluster_from_threshold(C, tau=tau_used)

    summary = SampleSummary(degeneracy=float(D),
                            rho_median=float(np.median(rhos)),
                            rho_mean=float(np.mean(rhos)),
                            dup_fraction=dup_fraction,
                            m_active=m_active,
                            rhos=rhos,
                            tau_used=tau_used)
    return summary, A_std, A

# ---------- Aggregation across a chain ----------

@dataclass
class ChainAggregate:
    summaries: List[SampleSummary]
    rho_all: np.ndarray
    r_eff_all: np.ndarray
    r_eff_all_normalized: np.ndarray
    D_all: np.ndarray
    dup_frac_all: np.ndarray
    m_active_all: np.ndarray
    A_pool: np.ndarray              # pooled standardized activations across samples (n x total_active)
    sample_slices: List[slice]      # to map pooled columns back to samples
    activation_rate: float          # fraction of active neurons: (n_samples * m_active) / (n_samples * m) = mean(m_active) / m

def aggregate_chain(network_trace: Dict,
                    X: np.ndarray,
                    chain_idx: int = 0,
                    sample_indices: Optional[List[int]] = None,
                    act: str = "relu",
                    tau: float = None,
                    tau_policy: str = "analytic",
                    gauge_fix: bool = True) -> ChainAggregate:
    if sample_indices is None:
        total_samples = network_trace['params']['fcn']['layer0']['kernel'].shape[1]
        sample_indices = list(range(total_samples))

    summaries: List[SampleSummary] = []
    pooled_cols = []
    slices: List[slice] = []

    col_start = 0
    for s in sample_indices:
        result = diagnostics_for_sample(network_trace, X, chain_idx, s, act=act, tau=tau, tau_policy=tau_policy, gauge_fix=gauge_fix)
        if result is None:
            continue
        summary, A_std, A = result
        summaries.append(summary)
        if A_std.shape[1] > 0:
            pooled_cols.append(A_std)
            col_end = col_start + A_std.shape[1]
            slices.append(slice(col_start, col_end))
            col_start = col_end

    if pooled_cols:
        A_pool = np.concatenate(pooled_cols, axis=1)  # (n x total_active)
    else:
        A_pool = np.zeros((X.shape[0], 0))

    rho_all = np.concatenate([s.rhos for s in summaries if s.rhos.size > 0]) if summaries else np.array([])
    r_eff_all = np.array([s.r_eff for s in summaries], dtype=float)
    r_eff_all_normalized = r_eff_all / A.shape[1] if A.shape[1] > 0 else r_eff_all
    D_all = np.array([s.degeneracy for s in summaries], dtype=float)
    dup_frac_all = np.array([s.dup_fraction for s in summaries], dtype=float)
    m_active_all = np.array([s.m_active for s in summaries], dtype=int)

    return ChainAggregate(summaries=summaries, rho_all=rho_all, 
                          r_eff_all_normalized=r_eff_all_normalized, r_eff_all=r_eff_all,
                          D_all=D_all, dup_frac_all=dup_frac_all, m_active_all=m_active_all,
                          A_pool=A_pool, sample_slices=slices,
                          activation_rate=A_pool.shape[1] / (len(summaries) * A_pool.shape[0]) if len(summaries) > 0 else 0.0,
                          )

# ---------------- Likelihood-flat directions ---------------- #

def _orthonormal_basis_one_perp(k: int) -> np.ndarray:
    """Return Q (k x (k-1)): columns form an orthonormal basis of {t in R^k : 1^T t = 0}."""
    if k < 2:
        return np.zeros((k, 0))
    e = np.eye(k)
    # Gram-Schmidt on columns of I minus mean
    U = e - np.ones((k, k))/k
    # Orthonormalize via SVD (stable)
    Uu, _, _ = np.linalg.svd(U, full_matrices=False)
    # Last column corresponds to the all-ones direction; drop it
    return Uu[:, :k-1]

def mixing_sensitivity_from_cluster(A: np.ndarray, idx: np.ndarray) -> float:
    """Given raw activation matrix A (n x m) and a cluster index set idx (size k>=2),
    compute S_cluster = sigma_min( A[:, idx] @ Q ), where Q spans 1^\perp in R^k.
    Smaller is 'flatter' (more mixing invariance).
    """
    k = len(idx)
    if k < 2:
        return np.nan
    Q = _orthonormal_basis_one_perp(k)  # (k x (k-1))
    B = A[:, idx] @ Q                   # (n x (k-1))
    # Smallest singular value of B
    s = np.linalg.svd(B, compute_uv=False)
    return float(np.min(s)) if s.size > 0 else np.nan

@dataclass
class MixingSensitivitySummary:
    s_all: np.ndarray         # sensitivities across clusters in this sample
    s_median: float
    s_mean: float
    cluster_sizes: np.ndarray

def mixing_sensitivity_for_sample(trace,
                                  X: np.ndarray,
                                  chain_index: int,
                                  sample_indices: int,
                                  act: str = "relu",
                                  tau: Optional[float] = None,
                                  tau_policy: str = "analytic",
                                  alpha: float = 0.01,
                                  gauge_fix: bool = True) -> Tuple[MixingSensitivitySummary, List[np.ndarray]]:
    """Detect clusters (via standardized A, threshold tau) and compute output-mixing sensitivity S per cluster on raw A."""
    summary, A_std, A_raw = diagnostics_for_sample(
        trace, X, chain_index, sample_idx=sample_indices, act=act, tau=tau, tau_policy=tau_policy, alpha=alpha, gauge_fix=gauge_fix
    )
    m_active = A_std.shape[1]
    if m_active < 2 or summary.degeneracy == 0:
        return MixingSensitivitySummary(s_all=np.array([]), s_median=np.nan, s_mean=np.nan, cluster_sizes=np.array([])), []

    # Recompute similarity and clusters with the same tau_used
    C = cosine_similarity_matrix(A_std)
    clusters, _ = cluster_from_threshold(C, tau=summary.tau_used)
    # Keep only clusters of size >= 2
    clusters = [c for c in clusters if len(c) >= 2]

    svals = []
    sizes = []
    for c in clusters:
        s = mixing_sensitivity_from_cluster(A_raw[:, :m_active], c)  # A_raw columns in same order as A_std active columns
        if np.isfinite(s):
            svals.append(s)
            sizes.append(len(c))

    svals = np.array(svals, dtype=float) if svals else np.array([])
    sizes = np.array(sizes, dtype=int) if sizes else np.array([])
    ms = MixingSensitivitySummary(s_all=svals,
                                  s_median=float(np.median(svals)) if svals.size > 0 else np.nan,
                                  s_mean=float(np.mean(svals)) if svals.size > 0 else np.nan,
                                  cluster_sizes=sizes)
    return ms, clusters

# ---------------- Aggregation across a chain (likelihood-flat directions) ---------------- #

@dataclass
class ChainSensitivityAggregate:
    s_all: np.ndarray         # pooled sensitivities across samples
    s_median_by_sample: np.ndarray
    s_mean_by_sample: np.ndarray
    cluster_sizes_all: np.ndarray

def aggregate_mixing_sensitivity(X: np.ndarray,
                                 trace: Dict,
                                 chain_idx: int = 0,
                                 sample_indices: Optional[List[int]] = None,
                                 act: str = "relu",
                                 tau: Optional[float] = None,
                                 tau_policy: str = "analytic",
                                 alpha: float = 0.01,
                                 gauge_fix: bool = True) -> ChainSensitivityAggregate:
    if sample_indices is None:
        total_samples = trace['params']['fcn']['layer0']['kernel'].shape[1]
        sample_indices = list(range(total_samples))

    s_all = []
    s_med = []
    s_mean = []
    sizes_all = []
    for s in sample_indices:
        ms, clusters = mixing_sensitivity_for_sample(
            trace, X, chain_idx, sample_indices=s, act=act, tau=tau, tau_policy=tau_policy, alpha=alpha, gauge_fix=gauge_fix
        )
        if ms.s_all.size > 0:
            s_all.append(ms.s_all)
            s_med.append(ms.s_median)
            s_mean.append(ms.s_mean)
            sizes_all.append(ms.cluster_sizes)

    s_all = np.concatenate(s_all) if s_all else np.array([])
    s_med = np.array(s_med) if s_med else np.array([])
    s_mean = np.array(s_mean) if s_mean else np.array([])
    sizes_all = np.concatenate(sizes_all) if sizes_all else np.array([], dtype=int)

    return ChainSensitivityAggregate(s_all=s_all, s_median_by_sample=s_med, s_mean_by_sample=s_mean, cluster_sizes_all=sizes_all)


color_start = "#348ABD"
color_mid = "#467821"
color_end = "#A60628"

scatter_colors = ['#FFD92F'] # Orange
textsize = 12
scattersize = 40
plot_prior_draws = True

X, y = data_loader_one_five_one.data
X_airfoil, y_airfoil = data_loader_five_five_one.data

# Mixing sensitivity aggregation for all experiments
agg_sens_one_five_one = aggregate_mixing_sensitivity(X, samples_one_five_one, chain_idx=0, act="relu", tau=None, tau_policy="analytic", alpha=0.01, gauge_fix=True)
agg_sens_one_hundred_one = aggregate_mixing_sensitivity(X, samples_one_hundred_one, chain_idx=0, act="relu", tau=None, tau_policy="analytic", alpha=0.01, gauge_fix=True)
agg_sens_five_five_one = aggregate_mixing_sensitivity(X_airfoil, samples_five_five_one, chain_idx=0, act="relu", tau=None, tau_policy="analytic", alpha=0.01, gauge_fix=True)
agg_sens_five_hundred_one = aggregate_mixing_sensitivity(X_airfoil, samples_five_hundred_one, chain_idx=0, act="relu", tau=None, tau_policy="analytic", alpha=0.01, gauge_fix=True)

def plot_mixing_sensitivity_diagnostics(agg_sens_list, titles, save_path=OUTPUT_PATH):
    """Plot likelihood-flat direction diagnostics for multiple experiments."""
    n_experiments = len(agg_sens_list)
    fig, axs = plt.subplots(2, n_experiments, figsize=(10, 3))
    
    # Ensure axs is 2D even for single experiment
    if n_experiments == 1:
        axs = axs.reshape(1, -1)        
    
    for i, (agg_sens, title) in enumerate(zip(agg_sens_list, titles)):
        # Plot histogram of all sensitivities (top row)
        if agg_sens.s_all.size > 0:
            sns.histplot(agg_sens.s_all, ax=axs[0, i], stat='density', fill=True, element='poly', bins=50, alpha=0.5)
            axs[0, i].set_ylabel("")
            axs[0, i].set_xlabel("")
            axs[0, i].text(0.8, 0.95, f"{title}", transform=axs[0, i].transAxes, fontsize=textsize+1, va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
            axs[0, i].set_xlabel(r"$S$ (sigma_min)", fontsize=textsize)
            median_s = np.median(agg_sens.s_all[np.isfinite(agg_sens.s_all)])
            axs[0, i].axvline(median_s, color='#D54E4E', linestyle='-', linewidth=2)

            if i == 0:
                axs[0, i].set_ylabel("Density", fontsize=textsize)
        
        # Plot median sensitivity per sample (bottom row)
        if agg_sens.s_median_by_sample.size > 0:
            axs[1, i].plot(np.arange(len(agg_sens.s_median_by_sample)), agg_sens.s_median_by_sample, alpha=0.6, lw=0.2)
            median_of_medians = np.median(agg_sens.s_median_by_sample[np.isfinite(agg_sens.s_median_by_sample)])
            axs[1, i].axhline(median_of_medians, color='#D54E4E', linestyle='-', linewidth=2)
            axs[1, i].text(0.8, 0.95, f"{title}", transform=axs[1, i].transAxes, fontsize=textsize+1, va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
            axs[1, i].set_xlabel("Sample Index", fontsize=textsize)

            if i == 0:
                axs[1, i].set_ylabel(r"Median $S$", fontsize=textsize)

        # Apply consistent styling
        for j in range(2):
            axs[j, i].spines['top'].set_visible(False)
            axs[j, i].spines['right'].set_visible(False)
            axs[j, i].spines['left'].set_visible(False)
            axs[j, i].spines['bottom'].set_visible(False)
            axs[j, i].grid(color='grey', linestyle='-', linewidth=0.4, alpha=0.5)
            axs[j, i].tick_params(axis='both', which='major', labelsize=textsize)
            axs[j, i].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{save_path}.pdf', dpi=300)

# %%
agg_sens_list = [agg_sens_one_five_one, agg_sens_one_hundred_one, agg_sens_five_five_one, agg_sens_five_hundred_one]
titles = ["1-5-1", "1-100-1", "5-5-1", "5-100-1"]
plot_mixing_sensitivity_diagnostics(agg_sens_list, titles)
