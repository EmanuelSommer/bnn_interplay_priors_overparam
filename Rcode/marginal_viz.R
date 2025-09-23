library(tidyverse)
# exp 1
exp <- "tabular_regr_mile_concrete"
weight <- "layer1_kernel0.csv"
title <- "8-16-1 FCN (162)"
colorcode <- "#58c458"
# exp 2
exp <- "tabular_regr_mile_concrete_over1"
weight <- "layer1_kernel0.csv"
title <- "8-16-16-16-16-1 FCN (978)"
colorcode <- "#58b2c4"
# exp 3
exp <- "tabular_regr_mile_concrete_over2"
weight <- "layer1_kernel0.csv"
title <- "8-64-64-1 FCN (4802)"
colorcode <- "#5867c4"

df <- read_csv(paste0(
  "tracebasepath", # adapt to your trace base path obtained from the save_traces.py util
  exp, "/traces/", weight
), col_names = FALSE)

colnames(df) <- as.character(1:ncol(df))

# keep the row index as chain column
df_long <- df |>
  mutate(chain = row_number()) |>
  pivot_longer(-chain, names_to = "param", values_to = "value")

p <- ggplot(df_long, aes(x = value, group = chain)) +
  geom_density(aes(y = ..density..), fill = colorcode, alpha = 0.2, color=NA) +
  geom_density(data = df_long, aes(x = value, y = ..density..), color = "black", size = 1, inherit.aes = F) +
  theme_minimal() +
  labs(x = "Random Kernel Weight", y = "Density", title = title) +
  theme(
    text = element_text(size = 17),
    axis.text = element_text(size = 16),
    title = element_text(size = 17),
    # increase size of numbers on the axes
    panel.grid.minor = element_blank(),
    # place title in the center
    plot.title = element_text(hjust = 0.5)
  )
p

ggsave(paste0("marginal_", exp, "_", weight, ".pdf"), plot = p, width = 5, height = 4)