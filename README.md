# data_pipeline

## `util/loader_mat.py`

Contains the class `Loader` which has methods for loading `.mat` files containing output MATLAB `Struct`'s generated during simulation and from the SARUS scanner.

## `train.py`

contains the class `Model` which provides a crude interface for:

* (`load_mat`) Ingesting output of the `Loader` class of `loader_mat.py`.
* (`print_img`) Visualizing data/labels.
* (`illustrate_history`) Visualizing the metrics over training epochs.
* (`fit_model`) If not done beforehand, either loads a pretrained model or builds, compiles, provides data generators and fits a model provided by the `models/unet.py`/`models/unet3d.py` script, see below. 

__If `Movement = True` in `config.ini` the pipeline assumes that your data has a time axis, i.e. the tensors are of shape `(n, m, t)`, if `Movement = False` their shape is assumed to be `(n, m)`.__

## `data/generate_frame.py` 
Contains a class `FrameGenerator` which generates pairs of data, label tensors for training neural networks. *This module creates the tensors*.

## `util/generator.py`
Contains a class `DataGenerator` which yields pairs of data, label
tensors for training neural networks. *This module feeds the tensors to the `model.fit()` method*.

## `models/unet.py`

This script contains a Tensorflow implementation of UNet, see:
[UNet](https://arxiv.org/abs/1505.04597) \
with the hyperparameters set in `config.ini`.

## `models/unet3d.py`

This script contains a Tensorflow implementation of UNet, for 3D-data see:
[UNet](https://arxiv.org/abs/1505.04597) \
with the hyperparameters set in `config.ini`. Here it is worth noting that the `layers.layers.SubPixel3D` implementation forces the use of a
feature upscaling factor of 4 rather than 2 for each up-block, see
`layers.layers.SubPixel3D`, for more details on this.

## `layers/layers.py`
Implements certain types of layers not currently present in Tensorflow, but which are part of the UNet neural network.

## `util/run_pipeline.py`
This script assumes you have a the following setup:
* A config files, following the format of `config.ini`, **all** fields are required.
* A directory which contains the following structure:
```
storage ------- training --------- data  
              |                  |  
              |                  ---- labels  
              |  
              ---- validation --------- data  
                                |  
                                ------- labels
```
__In the future a `evaluation` folder with subfolders `data` and `labels` will be required to.__
* A directory which contains the following structure\*:
```
mat_folder ------- train --------- data ----- [.mat-files]
              |                   
              |  
              ---- valid --------- data ----- [.mat-files]
```
\* **Note the difference in folder structures, this is to ensure that the user does not interchange these folders.**


### Example usage:
```{bash}
python3 run_pipeline.py -conf 'config.ini' -s_dir 'storage' -m_dir 'dir/w/mat-files'
```
This scripts does the following operations:
* If the folder `storage/training/data/` is not empty, data from `dir/w/mat-files/train/data/` is loaded in using `Loader.load_mat_folder`.
* If the folder `storage/validation/data/` is not empty, data from `dir/w/mat-files/valid/data/` is loaded in using `Loader.load_mat_folder`.
* `Model.fit_model()` is executed, and broadcasts an execution log to a `.log` files contained in `output`, the naming format is: `model_{ddmmyyyy_hh}.log` with the date and hour of execution.
* `Model.illustrate_history(history)` is executed, where `history` is the `tensorflow.Model.History` object of the fitted model. The plots are saved to `output`, the naming format is: `accuracy_{ddmmyyyy_hh}.log` and `loss_{ddmmyyyy_hh}.log` with the date and hour of execution.
* `Model.print_img()` is executed and saves examples images of data to `output`, the naming format is: `examples_{ddmmyyyy_hh}.log` with the date and hour of execution.

## TODO

* Do docstrings for all functionalities.
* Automatic docstring broadcasting to wiki-page. 
* Broadcasting of examples should also be of labels, not only data.
* The `Model.illustrate_history(history)` should be of all used metrics, not only `accuracy` and `loss`.
