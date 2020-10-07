# data_pipeline

## `loader_mat.py`

Contains the class `Loader` which has methods for loading `.mat` files containing output MATLAB `Struct`'s generated during simulation and from the SARUS scanner.

## `train.py`

contains the class `Model` which provides a crude interface for:

* (`load_mat`) Ingesting output of the `Loader` class of `loader_mat.py`.
* (`print_img`) Visualizing data/labels.
* (`fit_model`) If not done beforehand, either loads a pretrained model or builds, compiles, provides data generators and fits a model provided by the `build_model.py` script, see below.

## `build_model.py`

This script contains a crude Tensorflow implementation of UNet, see:
[UNet](https://arxiv.org/abs/1505.04597) \
with the hyperparameters set in `config.ini`.

## `layers.py`
Implements certain types of layers not currently present in Tensorflow, but which are part of the UNet neural network.