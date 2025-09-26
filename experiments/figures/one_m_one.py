"""Python file to reproduce the 1-M-1 marginal posteriors figure."""
# %%
import os
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
)

# Run experiments and specify paths
CONFIG_ONE_ONE_ONE = 'experiments/configs/marginals_1-M-1/one_one_one.yaml'
CONFIG_ONE_FIVE_ONE = 'experiments/configs/marginals_1-M-1/one_five_one.yaml'
CONFIG_ONE_HUNDRED_ONE = 'experiments/configs/marginals_1-M-1/one_hundred_one.yaml'

CONFIG_ONE_ONE_ONE_01 = 'experiments/configs/marginals_1-M-1/one_one_one_01.yaml'
CONFIG_ONE_FIVE_ONE_01 = 'experiments/configs/marginals_1-M-1/one_five_one_01.yaml'
CONFIG_ONE_HUNDRED_ONE_01 = 'experiments/configs/marginals_1-M-1/one_hundred_one_01.yaml'

CONFIG_ONE_ONE_ONE_03 = 'experiments/configs/marginals_1-M-1/one_one_one_03.yaml'
CONFIG_ONE_FIVE_ONE_03 = 'experiments/configs/marginals_1-M-1/one_five_one_03.yaml'
CONFIG_ONE_HUNDRED_ONE_03 = 'experiments/configs/marginals_1-M-1/one_hundred_one_03.yaml'

OUTPUT_PATH = 'figures/one_m_one_figure'


# %%
# Load configs for all three experiments
config_one_one_one, key_one_one_one = load_config_and_key(CONFIG_ONE_ONE_ONE)
config_one_five_one, key_one_five_one = load_config_and_key(CONFIG_ONE_FIVE_ONE)
config_one_hundred_one, key_one_hundred_one = load_config_and_key(CONFIG_ONE_HUNDRED_ONE)

config_one_one_one_01, key_one_one_one_01 = load_config_and_key(CONFIG_ONE_ONE_ONE_01)
config_one_five_one_01, key_one_five_one_01 = load_config_and_key(CONFIG_ONE_FIVE_ONE_01)
config_one_hundred_one_01, key_one_hundred_one_01 = load_config_and_key(CONFIG_ONE_HUNDRED_ONE_01)

config_one_one_one_03, key_one_one_one_03 = load_config_and_key(CONFIG_ONE_ONE_ONE_03)
config_one_five_one_03, key_one_five_one_03 = load_config_and_key(CONFIG_ONE_FIVE_ONE_03)
config_one_hundred_one_03, key_one_hundred_one_03 = load_config_and_key(CONFIG_ONE_HUNDRED_ONE_03)

# Setup loaders
sample_loader_one_one_one, samples_one_one_one, data_loader_one_one_one = setup_loaders(config_one_one_one, key_one_one_one)
sample_loader_one_five_one, samples_one_five_one, data_loader_one_five_one = setup_loaders(config_one_five_one, key_one_five_one)
sample_loader_one_hundred_one, samples_one_hundred_one, data_loader_one_hundred_one = setup_loaders(config_one_hundred_one, key_one_hundred_one)

sample_loader_one_one_one_01, samples_one_one_one_01, data_loader_one_one_one_01 = setup_loaders(config_one_one_one_01, key_one_one_one_01)
sample_loader_one_five_one_01, samples_one_five_one_01, data_loader_one_five_one_01 = setup_loaders(config_one_five_one_01, key_one_five_one_01)
sample_loader_one_hundred_one_01, samples_one_hundred_one_01, data_loader_one_hundred_one_01 = setup_loaders(config_one_hundred_one_01, key_one_hundred_one_01)

sample_loader_one_one_one_03, samples_one_one_one_03, data_loader_one_one_one_03 = setup_loaders(config_one_one_one_03, key_one_one_one_03)
sample_loader_one_five_one_03, samples_one_five_one_03, data_loader_one_five_one_03 = setup_loaders(config_one_five_one_03, key_one_five_one_03)
sample_loader_one_hundred_one_03, samples_one_hundred_one_03, data_loader_one_hundred_one_03 = setup_loaders(config_one_hundred_one_03, key_one_hundred_one_03)

color_start = "#348ABD"
color_mid = "#467821"
color_end = "#A60628"

textsize = 22
scattersize = 44
plot_prior_draws = True

# Create figure with 3x3 subplots
fig, axs = plt.subplots(3, 3, figsize=(12, 12))

banana_x = np.linspace(-4, 4, 1000)
banana_y = 1/banana_x
banana_y[abs(banana_y) > 1e2] = np.nan

if plot_prior_draws:
    w1_1 = np.random.normal(0, 1.0, size=1000)
    w2_1 = np.random.normal(0, 1.0, size=1000)

    w1_03 = np.random.normal(0, 0.31622776601683794, size=1000)
    w2_03 = np.random.normal(0, 0.31622776601683794, size=1000)

    w1_01 = np.random.normal(0, 0.1, size=1000)
    w2_01 = np.random.normal(0, 0.1, size=1000)

    # Plot prior distributions as KDE plots instead of scatter plots
    sns.kdeplot(x=w1_1, y=w2_1, ax=axs[0, 0], color='gray', alpha=0.3, levels=6, label='Prior')
    sns.kdeplot(x=w1_1, y=w2_1, ax=axs[0, 1], color='gray', alpha=0.3, levels=6)
    sns.kdeplot(x=w1_1, y=w2_1, ax=axs[0, 2], color='gray', alpha=0.3, levels=6)
    sns.kdeplot(x=w1_03, y=w2_03, ax=axs[1, 0], color='gray', alpha=0.3, levels=6)
    sns.kdeplot(x=w1_03, y=w2_03, ax=axs[1, 1], color='gray', alpha=0.3, levels=6)
    sns.kdeplot(x=w1_03, y=w2_03, ax=axs[1, 2], color='gray', alpha=0.3, levels=6)
    sns.kdeplot(x=w1_01, y=w2_01, ax=axs[2, 0], color='gray', alpha=0.3, levels=6)
    sns.kdeplot(x=w1_01, y=w2_01, ax=axs[2, 1], color='gray', alpha=0.3, levels=6)
    sns.kdeplot(x=w1_01, y=w2_01, ax=axs[2, 2], color='gray', alpha=0.3, levels=6)


def plot_kde(samples, ax, title, bw_method=0.12):
    sns.kdeplot(
        x=samples['params']['fcn']['layer0']['kernel'][:, :, 0, 0].flatten(),
        y=samples['params']['fcn']['layer1']['kernel'][:, :, 0, 0].flatten(),
        ax=ax,
        levels=6,
        color=color_start,
        alpha=0.8,
        label='NUTS',
        bw_method=bw_method,
        fill=True,
    )

    if ax == axs[2, 0] or ax == axs[2, 1] or ax == axs[2, 2]:  # Bottom row
        ax.set_xlabel('First Layer Weight', fontsize=textsize)
    if ax == axs[0, 0] or ax == axs[1, 0] or ax == axs[2, 0]:  # Left column
        ax.set_ylabel('Second Layer Weight', fontsize=textsize)
    ax.set_title(title, fontsize=textsize, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5), zorder=20)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(color='grey', linestyle='-', linewidth=0.4, alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=textsize - 2)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    ax.set_xticks([0, -4, -2, 2, 4])
    ax.set_yticks([0, -4, -2, 2, 4])

    ax.plot(banana_x, banana_y, 'r--', alpha=0.7, linewidth=2, c=color_mid)
    ax.legend(fontsize=14)

    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-3.1, 3.1)

    ax.legend().set_visible(False)

# %%
## First row
# First subplot
plot_kde(samples_one_one_one, axs[0, 0], r'1-1-1 FCN, $\tau=10^{0}$')

# Second subplot
plot_kde(samples_one_five_one, axs[0, 1], r'1-5-1 FCN, $\tau=10^{0}$')

# Third subplot
plot_kde(samples_one_hundred_one, axs[0, 2], r'1-100-1 FCN, $\tau=10^{0}$')


## Second row
# First subplot
plot_kde(samples_one_one_one_03, axs[1, 0], r'1-1-1 FCN, $\tau=10^{-0.5}$')

# Second subplot
plot_kde(samples_one_five_one_03, axs[1, 1], r'1-5-1 FCN, $\tau=10^{-0.5}$')

# Third subplot
plot_kde(samples_one_hundred_one_03, axs[1, 2], r'1-100-1 FCN, $\tau=10^{-0.5}$')

## Third row
# First subplot
plot_kde(samples_one_one_one_01, axs[2, 0], r'1-1-1 FCN, $\tau=10^{-1}$')

# Second subplot
plot_kde(samples_one_five_one_01, axs[2, 1], r'1-5-1 FCN, $\tau=10^{-1}$')

# Third subplot
plot_kde(samples_one_hundred_one_01, axs[2, 2], r'1-100-1 FCN, $\tau=10^{-1}$')

plt.tight_layout()

plt.savefig(f'{OUTPUT_PATH}.pdf', bbox_inches='tight')
