"""Python file to reproduce the covariance and correlation matrix figure."""
# %%
import os
import numpy as np
import matplotlib.pyplot as plt

import sys
from matplotlib.patches import Rectangle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# possibly change working directory to the root of the repository
# os.chdir(...)

from utils import (
    load_config_and_key,
    setup_loaders,
)

# %%
CONFIG_SMALL = 'experiments/configs/covariance_correlation/corr_5-16-1_airfoil.yaml'
CONFIG_LARGE = 'experiments/configs/covariance_correlation/corr_5-16-16-16-1_airfoil.yaml'

OUTPUT_PATH = 'figures/covariance_correlation'

# %%

config_small_airfoil, key_small_airfoil = load_config_and_key(CONFIG_SMALL)
config_large_airfoil, key_large_airfoil = load_config_and_key(CONFIG_LARGE)

sample_loader_small_airfoil, samples_small_airfoil, data_loader_small_airfoil = setup_loaders(config_small_airfoil, key_small_airfoil)
sample_loader_large_airfoil, samples_large_airfoil, data_loader_large_airfoil = setup_loaders(config_large_airfoil, key_large_airfoil)

# covariance matrix
def plot_covariance_matrix(samples, num_samples=200, clip_val_im=0.11, textsize=21):
    # Extract weights from the first layer
    # Extract all weights and biases from all layers
    all_weights = []
    
    # Get all parameters from the network
    params = samples['params']['fcn']
    
    # Iterate through all layers and collect weights and biases
    for layer_name, layer_params in params.items():
        print(f"Layer: {layer_name}, Params: {list(layer_params.keys())}")
        if 'kernel' in layer_params:
            kernel_weights = np.stack([weight.flatten() for weight in layer_params['kernel'][0, :num_samples, :, :]]) # shape (n_samples, n_weights)
            all_weights.append(kernel_weights)
            print(f"Kernel weights shape: {kernel_weights.shape}")
        if 'bias' in layer_params:
            bias_weights = np.stack([weight.flatten() for weight in layer_params['bias'][0, :num_samples, :]]) # shape (n_samples, n_biases)
            all_weights.append(bias_weights)
            print(f"Bias weights shape: {bias_weights.shape}")

    # Concatenate all weights
    weights = np.concatenate(all_weights, axis=1)  # Shape: (n_samples, total_n_weights)
    print(f"{weights.shape}")
    weights = np.array(weights.reshape(num_samples, -1))  # Shape: (n_samples, n_weights)
    print(f"{weights.shape}")
    # Compute covariance matrix
    cov_matrix = np.cov(weights, rowvar=False)  # Shape: (n_weights, n_weights)
    corr_matrix = np.corrcoef(weights, rowvar=False)
    
    # Determine common color scale limits to ensure zero has the same color
    cov_max = np.max(np.abs(cov_matrix))
    corr_max = np.max(np.abs(corr_matrix))

    # calculate percentile for meaningful comparison of covariances
    clip_val = np.percentile(np.abs(cov_matrix), 98)

    needs_inlay = cov_matrix.shape[0] > 300 or cov_matrix.shape[1] > 300
    
    # Plot covariance matrix and correlation matrix
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    im = ax1.imshow(cov_matrix, cmap='viridis', vmin=-clip_val_im, vmax=clip_val_im)
    cbar1 = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=textsize)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(axis='both', which='major', labelsize=textsize)
    
    # Add inlay for covariance if needed
    if needs_inlay:
        inlay_size = 40
        start_idx = 150
        
        inlay_ax1 = ax1.inset_axes([0.5, 0.5, 0.45, 0.45])
        inlay_ax1.imshow(cov_matrix[start_idx:start_idx+inlay_size, start_idx:start_idx+inlay_size], cmap='viridis', vmin=-clip_val, vmax=clip_val)
        inlay_ax1.set_xticks([])
        inlay_ax1.set_yticks([])
        rect = Rectangle((start_idx, start_idx), inlay_size, inlay_size, 
                linewidth=1.0, edgecolor='black', facecolor='none', alpha=0.8)
        ax1.add_patch(rect)

    ax2 = plt.subplot(1, 2, 2)
    im = ax2.imshow(corr_matrix, cmap='viridis', vmin=-corr_max, vmax=corr_max)
    cbar2 = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=textsize)  # Control colorbar label size
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    cbar1.set_ticks([-0.1, 0, 0.1])
    cbar2.set_ticks([-1.0, 0, 1.0])
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params(axis='both', which='major', labelsize=textsize)
    
    # Add inlay for correlation if needed
    if needs_inlay:
        inlay_ax2 = ax2.inset_axes([0.5, 0.5, 0.45, 0.45])
        inlay_ax2.imshow(corr_matrix[start_idx:start_idx+inlay_size, start_idx:start_idx+inlay_size], cmap='viridis', vmin=-corr_max, vmax=corr_max)
        inlay_ax2.set_xticks([])
        inlay_ax2.set_yticks([])
        rect = Rectangle((start_idx, start_idx), inlay_size, inlay_size, 
                linewidth=1.0, edgecolor='black', facecolor='none', alpha=0.8)
        ax2.add_patch(rect)

    plt.tight_layout()
    is_small = 'small' if cov_matrix.shape[0] < 300 else 'large'
    plt.savefig(f'{OUTPUT_PATH}_{is_small}.pdf', dpi=300)

    return clip_val

# %%
clip_val_small = plot_covariance_matrix(samples_small_airfoil)
clip_val_large = plot_covariance_matrix(samples_large_airfoil)
