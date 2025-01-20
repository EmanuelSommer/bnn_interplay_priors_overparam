# SAbI

This repository contains the code for the **2025 ICML submission "TBD"**.


## Project Structure

```
.
├── sabi     Python code
├── Rcode    R code
├── data     Data files
└── results  Git ignored directory for storing results
```


## Python Setup (using poetry)

```bash
git clone
cd sabi
# Activate your virtual environment
poetry install
```

All python code can be found in the `sabi` directory. Most scripts can be run interactively
using VSCode inline magic `#%%`.

## R Setup

The R code can be found in the `Rcode` directory.

We use R version `4.4.1` and the following packages:

- `ggplot2`
- `dplyr`
- ...