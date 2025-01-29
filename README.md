# SAI

This repository contains the code for the **2025 ICML submission Approximate Posteriors in Neural Networks: A Sampling Perspective**.


## Project Structure

```
.
├── experiments     Python scripts for experiments and figures.
    └── configs     Config files for experiments.
    └── figures     Python scripts to reproduce figures.
├── data            Data files.
├── src             Python modules.
└── results         Git ignored directory for storing results.
```

## Python Setup (using poetry)

```bash
git clone
cd sabi
# Possibly create a fresh virtual environment and activate it
poetry install --no-root
```

All python scripts to generate figures and results can be found in the `experiments` directory. Most scripts can be run interactively
using VSCode inline magic `#%%`.

## Run Experiments

Create a (gitignored) `results/` folder to store the results:

```bash
mkdir results
```

The individual experiments can be executed in parallel across CPU or GPU cores (with appropriate ENV variables set for your hardware), using the flag `-d`:

```bash
python python -m src.sai -d 12 -c experiments/configs/permuted_warmstarts.yaml
```

When a whole grid of experiments is to be run, the `-s` argument providing a the grid is used:

```bash
python -m src.sai -d 10 -c experiments/configs/mile_mean_regr_uci.yaml -s experiments/configs/tabular_search.yaml
```

The results in the respective subfolder of `results/` contain:

- The `config.yaml` configuration file describing the experiment.
- The trained warm-start models inside the `warmstart/` folder.
- The posterior samples in the `samples/` folder.
- Diagnostics and training logs.

**Utils:** We also provide utility CLI tools that 1) aggregate multiple experiments (`experiments/pool_results.py`) and 2) save a subset of traces for large experiments (`experiments/save_traces.py`).