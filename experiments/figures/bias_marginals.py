"""Python file to reproduce the empirical marginal bias distribution figure."""
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
BIAS_UNIFORM = 'experiments/configs/marginals_bias/bias_uniform_airfoil.yaml'
BIAS_PENALIZED = 'experiments/configs/marginals_bias/bias_penalized_airfoil.yaml'

# %%
config_bias_uniform, key_bias_uniform = load_config_and_key(BIAS_UNIFORM)
config_bias_penalized, key_bias_penalized = load_config_and_key(BIAS_PENALIZED)

# Setup loaders
sample_loader_bias_uniform, samples_bias_uniform, data_loader_bias_uniform = setup_loaders(config_bias_uniform, key_bias_uniform)
sample_loader_bias_penalized, samples_bias_penalized, data_loader_bias_penalized = setup_loaders(config_bias_penalized, key_bias_penalized)

# create function that shows ridgeline plot of bias distributions per layer for both experiments next to each other
def plot_bias_ridgeline(samples_bias_uniform, samples_bias_penalized):
    """
    Generates a two-column ridge plot of bias distributions for different layers.

    Args:
        samples_bias_uniform (dict): Dictionary containing bias data for the 'Uniform' experiment.
        samples_bias_penalized (dict): Dictionary containing bias data for the 'Penalized' experiment.
    """
    
    bias_data = []

    def process_samples(samples, experiment_name):
        """Helper function to process and append bias data."""
        if 'params' in samples and 'fcn' in samples['params']:
            for layer_name, layer_params in samples['params']['fcn'].items():
                if 'bias' in layer_params:
                    # Flatten the bias values and append them
                    bias_values = layer_params['bias'].flatten()
                    for val in bias_values:
                        bias_data.append({
                            'experiment': experiment_name,
                            'layer': layer_name,
                            'bias': np.array(val, dtype=np.float32)
                        })

    # Process both experiment data
    process_samples(samples_bias_uniform, 'Uniform')
    process_samples(samples_bias_penalized, 'Penalized')
    
    # Create DataFrame
    if not bias_data:
        print("No bias data found to plot.")
        return
        
    df = pd.DataFrame(bias_data)
    
    # Sort layers for consistent plotting order (e.g., layer_1, layer_2, ...)
    sorted_layers = sorted(df['layer'].unique(), key=lambda x: int(x[-1]))
    df['layer'] = pd.Categorical(df['layer'], categories=sorted_layers, ordered=True)
    df.sort_values('layer', inplace=True)
    df['bias'] = df['bias'].astype(float)

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    num_layers = len(sorted_layers)
    blue_palette = sns.light_palette("#3498db", n_colors=num_layers, reverse=True) # Blue palette
    orange_palette = sns.light_palette("#e67e22", n_colors=num_layers, reverse=True) # Orange palette

    g = sns.FacetGrid(df, row="layer", col="experiment", hue="layer", aspect=5, height=1, 
                      sharex=True, sharey=False, legend_out=True,
                      col_order=['Penalized', 'Uniform'])

    g.map_dataframe(sns.kdeplot, x="bias", fill=True, alpha=0.1)
    g.map_dataframe(sns.kdeplot, x="bias", color='white', fill=False, lw=1.5, alpha=0.5)

    for i, ax_row in enumerate(g.axes):
        # Column 0: Uniform (Blue)
        if ax_row[0].has_data():
            plt.setp(ax_row[0].collections, facecolor=blue_palette[i], edgecolor='white')
        # Column 1: Penalized (Orange)
        if ax_row[1].has_data():
            plt.setp(ax_row[1].collections, facecolor=orange_palette[i], edgecolor='white')

    g.map(plt.axhline, y=0, lw=1, clip_on=False, color='black')
    g.set(xlim=(-12, 8))

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        # Get the current row and column position
        row_name = ax.get_subplotspec().get_topmost_subplotspec().rowspan.start
        col_name = ax.get_subplotspec().get_topmost_subplotspec().colspan.start
        
        custom_labels = {
            (0, 0): f"1st Layer",
            (1, 0): f"2nd Layer",
            (2, 0): f"3rd Layer",
            (3, 0): f"4th Layer",
        }
        
        # Use custom labels
        layer_label = custom_labels.get((row_name, col_name), "")
        
        ax.text(0, .2, layer_label, color='black', fontsize=15,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "bias")
    g.figure.subplots_adjust(hspace=-0.5)
    g.set_titles("")
    g.set(yticks=[], ylabel="", xlabel="")
    g.despine(bottom=True, left=True)
    for ax in g.axes.flat:
        ax.tick_params(axis='x', labelsize=15)
    plt.savefig('bias_ridgeline_plot.pdf', dpi=300, bbox_inches='tight')


def plot_bias_ridgeline_2x4(samples_bias_uniform, samples_bias_penalized):
    """
    Generates a ridge plot of bias distributions arranged as 2 rows (experiments)
    by 4 columns (layers), preserving prior styling.
    """
    bias_data = []

    def process_samples(samples, experiment_name):
        if 'params' in samples and 'fcn' in samples['params']:
            for layer_name, layer_params in samples['params']['fcn'].items():
                if 'bias' in layer_params:
                    bias_values = layer_params['bias'].flatten()
                    for val in bias_values:
                        bias_data.append({
                            'experiment': experiment_name,
                            'layer': layer_name,
                            'bias': np.array(val, dtype=np.float32)
                        })

    process_samples(samples_bias_uniform, 'Uniform')
    process_samples(samples_bias_penalized, 'Penalized')

    if not bias_data:
        print("No bias data found to plot.")
        return

    df = pd.DataFrame(bias_data)

    # Sort layers like: layer_1, layer_2, ...
    sorted_layers = sorted(df['layer'].unique(), key=lambda x: int(x[-1]))
    df['layer'] = pd.Categorical(df['layer'], categories=sorted_layers, ordered=True)
    df.sort_values('layer', inplace=True)
    df['bias'] = df['bias'].astype(float)
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    num_layers = len(sorted_layers)
    blue_palette = sns.light_palette("#3498db", n_colors=num_layers+2, reverse=True)   # Uniform
    orange_palette = sns.light_palette("#e67e22", n_colors=num_layers+2, reverse=True) # Penalized
    g = sns.FacetGrid(
        df,
        row="experiment",              # CHANGED: experiments on rows
        col="layer",                   # CHANGED: layers on columns
        hue="layer",
        aspect=4, height=1,            # keep prior sizing/styling
        sharex=True, sharey=False, legend_out=True,
        row_order=['Uniform', 'Penalized'],   # preserves old left→right order in top→bottom
        col_order=sorted_layers
    )
    g.map_dataframe(sns.kdeplot, x="bias", fill=True, alpha=0.1)
    g.map_dataframe(sns.kdeplot, x="bias", color='white', fill=False, lw=1.5, alpha=0.5)

    for j, _layer in enumerate(sorted_layers):
        # Row 0: Penalized
        ax_top = g.axes[0, j]
        if ax_top.has_data():
            plt.setp(ax_top.collections, facecolor=orange_palette[j], edgecolor='white')
        # Row 1: Uniform
        ax_bottom = g.axes[1, j]
        if ax_bottom.has_data():
            plt.setp(ax_bottom.collections, facecolor=blue_palette[j], edgecolor='white')

    g.map(plt.axhline, y=0, lw=1, clip_on=False, color='black')
    g.set(xlim=(-12, 5.5))

    def label(x, color, label):
        ax = plt.gca()
        row_idx = ax.get_subplotspec().rowspan.start
        col_idx = ax.get_subplotspec().colspan.start

        if row_idx != 0:
            return

        ordinal = {0: "1st", 1: "2nd", 2: "3rd"}
        ordinal_label = ordinal.get(col_idx, f"{col_idx+1}th")
        ax.text(0, .2, f"{ordinal_label} Layer", color='black', fontsize=15,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "bias")

    g.figure.subplots_adjust(hspace=-0.5)
    g.figure.subplots_adjust(wspace=0.05)
    g.set_titles("")
    g.set(yticks=[], ylabel="", xlabel="")
    g.despine(bottom=True, left=True)
    for ax in g.axes.flat:
        ax.tick_params(axis='x', labelsize=15)
        ax.set_xticks([-10, -5, 0, 5])

    plt.savefig('bias_ridgeline_plot_2x4.pdf', dpi=300, bbox_inches='tight')
    plt.show()

plot_bias_ridgeline_2x4(samples_bias_uniform, samples_bias_penalized)
