"""Python file to reproduce the 1D under- and overparameterization figure."""
# %%
import os
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# possibly change working directory to the root of the repository
# os.chdir(...)

from utils import (
    load_config_and_key,
    setup_loaders,
    get_train_plan_and_batch_size,
    setup_evaluators,
    get_predictions,
)

from src.sai.dataset.tabular import TabularLoader

# %%
# Run experiments and specify paths
IZ_UN = 'experiments/configs/under_overparam/1d_underparam.yaml'
IZ_OV = 'experiments/configs/under_overparam/1d_overparam.yaml'
IZ_OV_2ND = 'experiments/configs/under_overparam/1d_overparam_2nd.yaml'

OUTPUT_PATH = 'figures/1d_under_overparam_figure'

# %%
# Load configs for all three experiments
config_iz_un, key_iz_un = load_config_and_key(IZ_UN)
config_iz_ov, key_iz_ov = load_config_and_key(IZ_OV)
config_iz_ov_2nd, key_iz_ov_2nd = load_config_and_key(IZ_OV_2ND)

# Setup loaders
sample_loader_iz_un, samples_iz_un, data_loader_iz_un = setup_loaders(config_iz_un, key_iz_un)
sample_loader_iz_ov, samples_iz_ov, data_loader_iz_ov = setup_loaders(config_iz_ov, key_iz_ov)
sample_loader_iz_ov_2nd, samples_iz_ov_2nd, data_loader_iz_ov_2nd = setup_loaders(config_iz_ov_2nd, key_iz_ov_2nd)

# Put x grid into data_test of both data loaders
x_grid = jnp.linspace(-3, 3, 400).reshape(-1, 1)
y_dummy = jnp.zeros_like(x_grid).squeeze()  # dummy y values, won't be used

x_range_loader_cfg = config_iz_un.data
x_range_loader_cfg.valid_split = 0.0
x_range_loader_cfg.test_split = 1.0  # use all data for test
x_range_loader_cfg.train_split = 0.0
x_range_loader = TabularLoader(config=data_loader_iz_un.config, rng_key=data_loader_iz_un.key, n_chains=data_loader_iz_un.n_chains)
x_range_loader.data = (x_grid, y_dummy)

# Get predictions
train_plan_iz_un, batch_size_test_iz_un = get_train_plan_and_batch_size(config_iz_un, data_loader_iz_un)
evaluators_iz_un = setup_evaluators(config_iz_un)
predictions_iz_un = get_predictions(
    evaluators_iz_un,
    train_plan_iz_un,
    sample_loader_iz_un,
    config_iz_un,
    x_range_loader,
    batch_size_test_iz_un,
)

train_plan_iz_ov, batch_size_test_iz_ov = get_train_plan_and_batch_size(config_iz_ov, data_loader_iz_ov)
evaluators_iz_ov = setup_evaluators(config_iz_ov)
predictions_iz_ov = get_predictions(
    evaluators_iz_ov,
    train_plan_iz_ov,
    sample_loader_iz_ov,
    config_iz_ov,
    x_range_loader,
    batch_size_test_iz_ov,
)

train_plan_iz_ov_2nd, batch_size_test_iz_ov_2nd = get_train_plan_and_batch_size(config_iz_ov_2nd, data_loader_iz_ov_2nd)
evaluators_iz_ov_2nd = setup_evaluators(config_iz_ov_2nd)
predictions_iz_ov_2nd = get_predictions(
    evaluators_iz_ov_2nd,
    train_plan_iz_ov_2nd,
    sample_loader_iz_ov_2nd,
    config_iz_ov_2nd,
    x_range_loader,
    batch_size_test_iz_ov_2nd,
)   

# Load unnormalized test data
dataset = np.genfromtxt("data/izmailov.data", delimiter=" ")
x, y = dataset[:, 0:1], dataset[:, 1]
mean_x = x.mean()
std_x = x.std()
mean_y = y.mean()
std_y = y.std()

# transform test data back to original scale
x_train_orig_scale = data_loader_iz_un.data_train[0] * std_x + mean_x
y_train_orig_scale = data_loader_iz_un.data_train[1] * std_y + mean_y
# x_test_orig_scale = data_loader_iz_un.data_test[0] * std_x + mean_x
# y_test_orig_scale = data_loader_iz_un.data_test[1] * std_y + mean_y

def _to1d(a):
    """Squeeze to 1D numpy array."""
    return np.asarray(a).squeeze()

def plot_pred_panel(ax, x_train, y_train, x_grid, chains, title):
    x_train = _to1d(x_train); y_train = _to1d(y_train); xg = _to1d(x_grid)
    sns.scatterplot(ax=ax, x=x_train, y=y_train, label='Training Data', s=20, alpha=0.9, color='black')
    palette = sns.color_palette(n_colors=len(chains))
    for i, (label, pred) in enumerate(chains.items()):
        yg = _to1d(pred)
        sns.scatterplot(ax=ax, x=xg, y=yg, label=label, s=16, alpha=0.6, color=palette[i])
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(ncol=2, fontsize=8, frameon=True)

def plot_indiv_sample_preds(ax, chains_to_plot, color, x_train, y_train, x_grid, preds, sample_limit=1000):
    x_train = _to1d(x_train); y_train = _to1d(y_train); xg = _to1d(x_grid)
    sns.scatterplot(ax=ax, x=x_train, y=y_train, s=20, alpha=0.9, color='black')
    num_chains, num_samples, _, _ = preds['sampling'].shape
    all_preds = []
    for sample_idx in range(num_samples):
        yg_sample = _to1d(preds['sampling'][np.array(chains_to_plot), sample_idx, :, 0]) * std_y + mean_y
        all_preds.append(yg_sample)
    all_preds = np.array(all_preds)
    all_preds = all_preds.reshape(all_preds.shape[0] * all_preds.shape[1], all_preds.shape[-1])
    y_min = np.min(all_preds, axis=0)
    y_max = np.max(all_preds, axis=0)
    ax.fill_between(xg, y_min, y_max, color=color, alpha=0.5)

KDE_DEFAULTS = dict(
    levels=6,        
    fill=True,       
    linewidths=1.0,
    alpha=0.8,
    bw_method=0.2,
)

def kde2d(ax, x, y, **overrides):
    """
    Single wrapper around seaborn.kdeplot used everywhere.
    Edit KDE_DEFAULTS or this function to alter ALL KDEs globally.
    """
    params = {**KDE_DEFAULTS, **overrides}
    return sns.kdeplot(ax=ax, x=x, y=y, **params)

def plot_weight_pair_kde(ax, samples, chains, *,
                         labels=None, colors=['#1f77b4', '#BD6734'],
                         idx_l0=(0, 0), idx_l1=(0, 0),
                         kde_overrides=None):
    """
    Plot KDEs for (layer0.kernel[..., idx_l0], layer1.kernel[..., idx_l1]) across selected chains.
    - samples: nested dict as in your code
    - chains: list of chain indices to plot
    - labels/colors: optional per-chain aesthetics
    - idx_l0/idx_l1: (i, j) indices into the last 2 dims of each kernel tensor
    - kde_overrides: kwargs that override KDE_DEFAULTS for THIS panel
    """
    kde_overrides = kde_overrides or {}
    x = samples['params']['fcn']['layer0']['kernel'][np.array(chains), :, idx_l0[0], idx_l0[1]].ravel()
    y = samples['params']['fcn']['layer1']['kernel'][np.array(chains), :, idx_l1[0], idx_l1[1]].ravel()
    extras = {}
    sns.scatterplot(ax=ax, x=x, y=y, s=15, alpha=0.3, edgecolors='none', **{**kde_overrides, **extras})

def stylize_panel(ax, xlabel, ylabel, xticks=None, yticks=None, ylim=None):
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.grid(color='grey', linestyle='-', linewidth=0.4, alpha=0.5)
    ax.tick_params(axis='both', which='major')
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    ax.legend().set_visible(False)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.tick_params(axis='both', which='major', labelsize=18)
    if ylim is not None:
        ax.set_ylim(ylim)

# %%
fig, ax = plt.subplots(1, 4, figsize=(16, 4))

chains_to_plot = [3, 4]  # example: plot for chains 5th and 11th (0-based)

chain_colors = sns.color_palette(n_colors=3)


# (0,0): KDE underparameterized
ax00 = ax[0]
plot_weight_pair_kde(
    ax=ax00,
    samples=samples_iz_un,
    chains=chains_to_plot,
    kde_overrides={"color": chain_colors[0]},
)
stylize_panel(
    ax00,
    xlabel='First Layer Weight',
    ylabel='Second Layer Weight',
    xticks=[0, -2, 2],
    yticks=[0, -2, 2],
)

ax[0].set_title('1-7-1', fontsize=18, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5), zorder=20, loc='left')


# (1,0): KDE overparameterized
ax10 = ax[1]
plot_weight_pair_kde(
    ax=ax10,
    samples=samples_iz_ov,
    chains=chains_to_plot,
    kde_overrides={"color": chain_colors[1]},
)
# Add legend only if labels were provided
ax10.legend()
stylize_panel(
    ax10,
    xlabel='First Layer Weight',
    ylabel='',
    xticks=[0, -2, 2],
    yticks=[0, -2, 2],
)

ax[1].set_title('1-100-1', fontsize=18, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5), zorder=20, loc='left')


# (2,0): Predictions underparameterized
ax20 = ax[2]
plot_weight_pair_kde(
    ax=ax20,
    samples=samples_iz_ov_2nd,
    chains=chains_to_plot,
    kde_overrides={"color": chain_colors[2]},
)
stylize_panel(
    ax20,
    xlabel='First Layer Weight',
    ylabel='',
    xticks=[0, -2, 2],
    yticks=[0, -2, 2],
)

ax[2].set_title('1-500-1', fontsize=18, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5), zorder=20, loc='left')


# (3,0): Predictions overparameterized

plot_indiv_sample_preds(
    ax=ax[3],
    chains_to_plot=chains_to_plot,
    color=chain_colors[2],
    x_train=x_train_orig_scale,
    y_train=y_train_orig_scale,
    x_grid=x_grid * std_x + mean_x,
    preds=predictions_iz_ov_2nd,
    sample_limit=1000, 
)
plot_indiv_sample_preds(
    ax=ax[3],
    chains_to_plot=chains_to_plot,
    color=chain_colors[1],
    x_train=x_train_orig_scale,
    y_train=y_train_orig_scale,
    x_grid=x_grid * std_x + mean_x,
    preds=predictions_iz_ov,
    sample_limit=1000, 
)
plot_indiv_sample_preds(
    ax=ax[3],
    chains_to_plot=chains_to_plot,
    color=chain_colors[0],
    x_train=x_train_orig_scale,
    y_train=y_train_orig_scale,
    x_grid=x_grid * std_x + mean_x,
    preds=predictions_iz_un,
    sample_limit=1000,
)

stylize_panel(
    ax[3],
    xlabel='x',
    ylabel='y',
    xticks=[-2, 0, 2],
    yticks=[-12, -6, 0, 6, 12, 18],
    ylim=(-13, 19),
)

plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}.pdf', dpi=300)
plt.show()
