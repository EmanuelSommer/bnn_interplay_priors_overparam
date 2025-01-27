# %%
import plotnine as p9
import pandas as pd
import jax.numpy as jnp

import pandas as pd
import plotnine as p9

def plot_cumulative_lppd(
    running_lppd: list,
    iterations: int,
    output_path: str,
    x_label: str = "#Chains",
    y_label: str = "Cumulative LPPD",
    text_size: int = 20,
    axis_text_size: int = 16,
):
    """Plot and save the cumulative LPPD over iterations.

    Args:
        running_lppd (list): Values of the running LPPD.
        iterations (int): Number of iterations.
        output_path (str): Path to save the plot.
        x_label (str): Label for the x-axis. Default is '#Chains'.
        y_label (str): Label for the y-axis. Default is 'Cumulative LPPD'.
        text_size (int): Font size for text elements. Default is 20.
        axis_text_size (int): Font size for axis text. Default is 16.
    """
    df = pd.DataFrame({
        "running_lppd": running_lppd,
        "iteration": range(1, iterations + 1)
    })

    p = (
        p9.ggplot(df) + 
        p9.geom_line(p9.aes(x="iteration", y="running_lppd"), size=0.9) +
        p9.scale_x_log10() +
        p9.labs(x=x_label, y=y_label) +
        p9.theme_minimal() +
        p9.theme(
            panel_background=p9.element_rect(fill="white"),
            text=p9.element_text(size=text_size),
            axis_text=p9.element_text(size=axis_text_size),
        )
    )

    p.save(output_path)

# %%
# Evaluate the cumulative LPPD over chains.
cumulative_lppd_miles = jnp.load("../data/fireball/mile_bike/Lppd_sampling.npz")
# %%
running_lppd = cumulative_lppd_miles["running_lppd"]
running_lppd.shape
# %%
plot_cumulative_lppd(
    running_lppd=running_lppd,
    iterations=running_lppd.shape[0],
    output_path="../results/cumulative_lppd_fireballs/cumulative_lppd_miles.pdf",
)
# %%
# repeat for NUTS
cumulative_lppd_nuts = jnp.load("../data/fireball/nuts_bike/Lppd_sampling.npz")
running_lppd = cumulative_lppd_nuts["running_lppd"][::5] # as we have 200 chains only
plot_cumulative_lppd(
    running_lppd=running_lppd,
    iterations=running_lppd.shape[0],
    output_path="../results/cumulative_lppd_fireballs/cumulative_lppd_nuts.pdf",
)
# %%
