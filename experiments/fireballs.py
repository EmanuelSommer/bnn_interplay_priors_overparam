# %%
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from datashader.mpl_ext import dsshow
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import jax.numpy as jnp 
import numpy as np
import os

color_start = "#7d7580"
color_mid = "#050df5"
color_end = "#ff00aa"
linear_cmap = LinearSegmentedColormap.from_list("custom_cmap", [color_start, color_mid, color_end])

def plot_bivar_density(
    w1, w2,
    xlab="w1", ylab="w2",
    title="", textsize=16,
    group=None,
    kde=False,
    kde_bandwidth=0.25,
    limiter=4,
    save_path="bivar_density.png",
):
    """
    Plots bivariate density for large vectors using datashader.
    """
    # Prepare the data
    df = pd.DataFrame({'W1': w1, 'W2': w2})
    if group is not None:
        df['group'] = pd.Categorical(group)
    else:
        df['group'] = None

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    if group is not None:
        # Categorical data
        if not kde:
            dsartist = dsshow(
                df,
                ds.Point("W1", "W2"),
                ds.count_cat("group"),
                shade_hook=tf.dynspread,
                width_scale=0.8,
                height_scale=0.8,
                norm="eq_hist",
                aspect="equal",
                ax=ax,
            )
        else:
            sns.kdeplot(
                data=df, x="W1", y="W2", hue="group", fill=True, bw_adjust=kde_bandwidth,
                common_norm=False, palette="tab10", ax=ax, legend=False
            )
    else:
        if not kde:
            dsartist = dsshow(
                df,
                ds.Point("W1", "W2"),
                ds.count(),
                # shade_hook=tf.dynspread,
                width_scale=3,
                height_scale=3,
                norm="eq_hist",
                aspect="equal",
                ax=ax,
                # blues as the colormap
                cmap=linear_cmap,
            )
        else:
            sns.kdeplot(
                data=df, x="W1", y="W2", fill=True, bw_adjust=kde_bandwidth,
                common_norm=False, palette="tab10", ax=ax, legend=False)

    ax.set_xlabel(xlab, fontsize=textsize)
    ax.set_ylabel(ylab, fontsize=textsize)
    ax.set_title(title, fontsize=textsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(color='grey', linestyle='-', linewidth=0.4, alpha=0.5)
    ax.set_xticks([0, -2, 2])
    ax.set_yticks([0, -2, 2])
    ax.tick_params(axis='both', which='major', labelsize=textsize - 2)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    ax.set_xlim(-limiter, limiter)
    ax.set_ylim(-limiter, limiter)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# %%
# Test the function
w1 = jnp.load("../data/fireball/mile_bike/traces/layer0_bias0.npz")["bias"]
w2 = jnp.load("../data/fireball/mile_bike/traces/layer0_bias1.npz")["bias"]
filter_groups = 1000
group = False
if filter_groups is not None:
    w1 = w1[:filter_groups]
    w2 = w2[:filter_groups]
if not group:
    # flatten the arrays
    w1 = w1.flatten()
    w2 = w2.flatten()
    group = None
else:
    group = jnp.repeat(jnp.arange(w1.shape[0]), w1.shape[1])
    group = np.array(group)
    w1 = w1.flatten()
    w2 = w2.flatten()


# %%
# Call the function
os.makedirs("../results/figs/fireballs", exist_ok=True)
plot_bivar_density(
    w1, w2, 
    xlab="$w_1$", 
    ylab="$w_2$", 
    group=group,
    textsize=28,
    title=f"Kernel weights of hidden layer 1",
    kde=False,
    save_path="..results/figs/fireballs/bivar_density.pdf",
)
# %%
# Now let's loop over all the layers and layer types
layers = [i for i in range(4)]
layer_types = ["kernel", "bias"]
sampler = ["mile", "nuts"]
save_id_base = "fireball_bike"

os.makedirs("../results/figs/fireballs", exist_ok=True)

for samp in tqdm(sampler):
    for layer in layers:
        for layer_type in layer_types:
            # load data
            w1 = jnp.load(f"../data/fireball/{samp}_bike/traces/layer{layer}_{layer_type}0.npz")[layer_type]
            w2 = jnp.load(f"../data/fireball/{samp}_bike/traces/layer{layer}_{layer_type}1.npz")[layer_type]
            nchains = w1.shape[0]

            # first plot the full fireball
            group = None
            plot_bivar_density(
                w1.flatten(), w2.flatten(), 
                xlab="$w_1$", 
                ylab="$w_2$", 
                title=f"{nchains} Chains",
                group=group,
                textsize=28,
                kde=False,
                save_path=f"../results/figs/fireballs/{save_id_base}_full_{samp}_layer{layer}_{layer_type}.pdf",
            )

            # now plot a subset of 10 chains with KDE
            filter_groups = 10
            w1 = w1[:filter_groups]
            w2 = w2[:filter_groups]
            group = jnp.repeat(jnp.arange(w1.shape[0]), w1.shape[1])
            group = np.array(group)

            plot_bivar_density(
                w1.flatten(), w2.flatten(), 
                xlab="$w_1$", 
                ylab="$w_2$", 
                title=f"{filter_groups} Chains",
                group=group,
                textsize=28,
                kde=True,
                kde_bandwidth=0.25 if samp == "mile" else 1.0,
                save_path=f"../results/figs/fireballs/{save_id_base}_subset_{samp}_layer{layer}_{layer_type}.pdf",
            )



