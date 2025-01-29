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
poetry install
```

All python scripts to generate figures and results can be found in the `experiments` directory. Most scripts can be run interactively
using VSCode inline magic `#%%`.

## Run Experiments

Create a (gitignored) `results/` folder to store the results:

```bash
mkdir results
```

The individual experiments can be executed in parallel across CPU cores, using the flag `-d`:

```bash
python python -m src/sai -d 12 -c experiments/configs/permuted_warmstarts.yaml
```

The results in the respective subfolder of `results/` contain:

- The `config.yaml` configuration file describing the experiment.
- The trained warm-start models inside the `warmstart/` folder.
- The posterior samples in the `samples/` folder.
- Diagnostics and training logs.