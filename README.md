# PyTorch Utils

Utilities for building PyTorch pipelines and models

## List of tools

* `transformation/normalize.py`
  * `Normalize`: register data statistics (numpy, xarray, tensors) for (de-)noramlization, generate torch
    normalization layers, save and restore.
* `layers/lossfn.py`
  * `RegressionLoss`: L1 and L2 loss functions for missing values in target.
* `pipeline/tshandling.py`:
  * `SeqScheme`: Create training tuples for sequential data with different strategies and deal with missing values

## Installation

`pip install git+https://github.com/bask0/pytorch_utils.git@master`

## Collaboration

Feel free to add your helper tools here or suggest improvements.

I plan to add tools to deal with [xarray](https://xarray.pydata.org/en/stable/) data (building pipelines, sampling,
etc.), any contributions welcome!
