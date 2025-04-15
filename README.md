# Decomposed diffusion

## Installation
In the following, Linux is assumed as the OS but the installation on Windows should be similar.

We recommend to install the required python packages (see `requirements.yml`) via a conda environment (e.g. using [miniconda](https://docs.conda.io/en/latest/miniconda.html)), but it may be possible to directly install them with *pip* (e.g. via *venv* for a separate environment) as well.
```shell
conda env create -f requirements.yml
conda activate ACDM
```
In the following, all commands should be run from the root directory of this source code.

## Directory Structure and Basic Usage
The directory `src/turbpred` contains the general code base of this project. The `src/lsim` directory contains the [LSiM metric](https://github.com/tum-pbs/LSIM) that is used for evaluations. The `data` directory contains data generation scripts, and downloaded or generated data sets should end up there as well. The `models` directory contains pretrained model checkpoints once they are downloaded (see below). The `runs` directory contains model checkpoints as well as further log files when training models from scratch. The `results` directory contains the results from the sampling, evaluation, and plotting scripts. Sampled model predictions are written to this directory as compressed numpy arrays. These arrays are read by the plotting scripts, which in turn write the resulting plots to the `results` directory as well.

The scripts in `src` contain the main training, sampling, and plotting functionality. Each script contains various configuration options and architecture selections at the beginning of the file. All files should be run directly with Python according to the following pattern:
```shell
python src/training_*.py
python src/sample_models_*.py
python src/plot_*.py
```

