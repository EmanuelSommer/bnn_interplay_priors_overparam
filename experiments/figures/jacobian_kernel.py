"""Python file to reproduce the Jacobian projections figure."""
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# possibly change working directory to the root of the repository
# os.chdir(...)

from utils import (
    load_config_and_key,
    setup_loaders,
)
# %%

ONE_TWO_ONE_CONFIG = 'experiments/configs/jacobian_one_two_one_relu.yaml'
OUTPUT_PATH = 'figures/jacobian_projections_figure'

# %%
config, key = load_config_and_key(ONE_TWO_ONE_CONFIG)
sample_loader, samples, data_loader = setup_loaders(config, key)

all_weights = []

# Get all parameters from the network
params = samples['params']['fcn']

# Iterate through all layers and collect weights and biases
for layer_name, layer_params in params.items():
    print(f"Layer: {layer_name}, Params: {list(layer_params.keys())}")
    if 'kernel' in layer_params:
        # shape (n_samples, n_weights)
        kernel_weights = np.stack([weight.flatten() for weight in layer_params['kernel'][1, :, :, :]])
        all_weights.append(kernel_weights)

weights = np.concatenate(all_weights, axis=1)  # Shape: (n_samples, total_n_weights)

df = np.genfromtxt('data/synthetic_regression_1x.data', delimiter=' ')
x = df[:, 0]
y = df[:, 1]


def relu(z):
    """Rectified Linear Unit function."""
    return np.maximum(0, z)

def jacobian_f(theta, x_data):
    """
    Computes the Jacobian of f(x) = a*ReLU(b*x) + c*ReLU(d*x)
    with respect to theta = (a, c, b, d).
    """
    a, c, b, d = theta
    
    # Partial derivatives (vectorized)
    df_da = relu(b * x_data)
    df_db = a * (b * x_data > 0) * x_data
    df_dc = relu(d * x_data)
    df_dd = c * (d * x_data > 0) * x_data
    
    # Stack into Jacobian matrix (n_data x 4)
    return np.stack([df_da, df_dc, df_db, df_dd], axis=1)

# Calculate Jacobian at the posterior mean
theta_mean = weights.mean(axis=0)
J = jacobian_f(theta_mean, x).reshape(-1, 4)  # Shape: (n_data, n_params)

# Perform Singular Value Decomposition (SVD)
U, s, Vh = np.linalg.svd(J, full_matrices=True)
V = Vh.T

# Determine the effective rank
tol = 1e-8
rank = np.sum(s > tol)
print(f"Jacobian shape: {J.shape}")
print(f"Effective rank: {rank}")

# Get basis vectors for image and null spaces
V_img = V[:, :rank]   # Image space basis (first 'rank' columns)
V_null = V[:, rank:]  # Null space basis (remaining columns)

# Center weights
D = weights - theta_mean
# Project weights in the SVD basis
Z = D @ V  

Z_img = Z[:, :rank]   # Image space projections
Z_null = Z[:, rank:]  # Null space projections

Theta_img = Z_img @ V_img.T
Theta_null = Z_null @ V_null.T

# Prepare data for plotting in a pandas DataFrame
df_list = []

# Original parameter space (a vs b)
df_list.append(pd.DataFrame({
    'x_coord': weights[:, 0],
    'y_coord': weights[:, 2],
    'space': 'Original (a vs b)'
}))

# Image space (first two dimensions)
if V_img.shape[1] >= 2:
    df_list.append(pd.DataFrame({
        'x_coord': Theta_img[:, 0],
        'y_coord': Theta_img[:, 1],
        'space': 'Image Space (dims 1-2)'
    }))

# Null space (first two dimensions)
if V_null.shape[1] >= 2:
    df_list.append(pd.DataFrame({
        'x_coord': Theta_null[:, 0],
        'y_coord': Theta_null[:, 1],
        'space': 'Null Space (dims 1-2)'
    }))

df_all = pd.concat(df_list, ignore_index=True)

# Create the plot
g = sns.FacetGrid(df_all, col="space", sharex=False, sharey=False, height=4)
g.map_dataframe(sns.scatterplot, x="x_coord", y="y_coord", alpha=0.3, s=24, linewidth=0)
g.set_axis_labels("", "")
g.set_titles(col_template="")
for ax in g.axes.flat:
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.grid(color='grey', linestyle='-', linewidth=0.4, alpha=0.5)
    ax.tick_params(axis='both', which='major')
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    ax.legend().set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=18)

# Set custom ticks for all subplots
for ax in g.axes.flat:
    if ax == g.axes.flat[0]:  # Original space
        ax.set_xticks([0, 1.5, 3])
        ax.set_yticks([0, 1.5, 3])
        ax.set_xlim(-0.2, 3.2)
        ax.set_ylim(-0.2, 3.2)
        ax.axhline(y=0, color='black', linewidth=2.0, alpha=0.8)
        ax.axvline(x=0, color='black', linewidth=2.0, alpha=0.8)
    elif ax == g.axes.flat[1]: # Image space
        ax.set_xticks([0, 0.5])
        ax.set_yticks([0, -0.5, -1])
    elif ax == g.axes.flat[2]:  # Null space
        ax.set_xticks([-1.5, 0, 1.5])
        ax.set_yticks([-1.5, 0, 1.5])
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.axhline(y=0, color='black', linewidth=2.0, alpha=0.8)
        ax.axvline(x=0, color='black', linewidth=2.0, alpha=0.8)

plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}.pdf', dpi=300)