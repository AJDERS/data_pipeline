::: {.document}
::: {.documentwrapper}
::: {.bodywrapper}
::: {.body role="main"}
::: {#data-pipeline .section}
data_pipeline[¶](#data-pipeline "Permalink to this headline"){.headerlink}
==========================================================================

::: {#data-pipeline-util-loader-mat-py .section}
`data_pipeline.util.loader_mat.py`{.docutils .literal .notranslate}[¶](#data-pipeline-util-loader-mat-py "Permalink to this headline"){.headerlink}
---------------------------------------------------------------------------------------------------------------------------------------------------

Contains the class `Loader`{.docutils .literal .notranslate} which has
methods for loading `.mat`{.docutils .literal .notranslate} files
containing output MATLAB `Struct`{.docutils .literal .notranslate}'s
generated during simulation and from the SARUS scanner.
:::

::: {#data-pipeline-train-py .section}
`data_pipeline.train.py`{.docutils .literal .notranslate}[¶](#data-pipeline-train-py "Permalink to this headline"){.headerlink}
-------------------------------------------------------------------------------------------------------------------------------

Contains the class `Model`{.docutils .literal .notranslate} which
provides a crude interface for:

-   (`load_mat`{.docutils .literal .notranslate}) Ingesting output of
    the `Loader`{.docutils .literal .notranslate} class of
    `loader_mat.py`{.docutils .literal .notranslate}.

-   (`print_img`{.docutils .literal .notranslate}) Visualizing
    data/labels.

-   (`illustrate_history`{.docutils .literal .notranslate}) Visualizing
    the metrics over training epochs.

-   (`fit_model`{.docutils .literal .notranslate}) If not done
    beforehand, either loads a pretrained model or builds, compiles,
    provides data generators and fits a model provided by the
    `build_model.py`{.docutils .literal .notranslate} script, see below.

::: {#module-data_pipeline.train .section}
[]{#data-pipeline-train-model}

### `data_pipeline.train.Model`{.docutils .literal .notranslate}[¶](#module-data_pipeline.train "Permalink to this headline"){.headerlink}

This module contains a class called Model which implements an interface
for loading parsed `.mat`{.docutils .literal .notranslate} files,
building, compiling, fitting and inspect the the results of tensorflow
`tensorflow.keras.models`{.docutils .literal .notranslate}.

*class* `data_pipeline.train.`{.sig-prename .descclassname}`Model`{.sig-name .descname}[(]{.sig-paren}*[data_folder_path]{.n}*, *[config_path]{.n}*[)]{.sig-paren}[¶](#data_pipeline.train.Model "Permalink to this definition"){.headerlink}

:   **This class loads, builds, compiles and fits tensorflow models.**

    Furthermore it has methods for creating data generators for these
    models, inspecting metrics over epochs, inspecting runtime
    information and input data.

    The specifications of the input data is set in
    `config.ini`{.docutils .literal .notranslate}.

    `broadcast`{.sig-name .descname}[(]{.sig-paren}*[history]{.n}[:]{.p} [Type[\[]{.p}tensorflow.python.keras.callbacks.History[\]]{.p}]{.n}*[)]{.sig-paren} → None[¶](#data_pipeline.train.Model.broadcast "Permalink to this definition"){.headerlink}

    :   **This method broadcast training information to a log-file.**

        First a model summary is logged, and then the training metrics
        history is logged for each metric.

        Parameters

        :   **history** -- A `tensorflow.keras.History`{.docutils
            .literal .notranslate} object. Its History.history attribute
            is a record of training loss values and metrics values at
            successive epochs, as well as validation loss values and
            validation metrics values (if applicable).

    `build_model`{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren} → None[¶](#data_pipeline.train.Model.build_model "Permalink to this definition"){.headerlink}

    :   **This method builds a model if none is loaded.**

        The build model is stored in `self.model`{.docutils .literal
        .notranslate}, unless `self.loaded_model == True`{.docutils
        .literal .notranslate}, see note of `Model.load_model`{.docutils
        .literal .notranslate}.

    `compile_model`{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren} → None[¶](#data_pipeline.train.Model.compile_model "Permalink to this definition"){.headerlink}

    :   **This method compiles a model if none is loaded.**

        ::: {.admonition .note}
        Note

        If a model is compiled `self.model_compiled`{.docutils .literal
        .notranslate} is set to `True`{.docutils .literal .notranslate}.
        :::

    `fit_model`{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren} → Type[\[]{.p}tensorflow.python.keras.callbacks.History[\]]{.p}[¶](#data_pipeline.train.Model.fit_model "Permalink to this definition"){.headerlink}

    :   **This function completes a full network pipeline.**

        All of the below functionalities are completed if they are not
        already executed. The execution history is based on the booleans
        mentioned in `Model.load_model`{.docutils .literal
        .notranslate}, `Model.build_model`{.docutils .literal
        .notranslate}, and `train.Model.compile`{.docutils .literal
        .notranslate}, and the existence of generators mentioned in
        `train.Model.generator`{.docutils .literal .notranslate}.

        First this function builds a model, then compiles it, make a
        training and validation generator, fits the model and lastly
        broadcasts runtime information to
        `output/loggin_{dt_string}.log`{.docutils .literal
        .notranslate}, and metrics over epoch to
        `output/{metric}_{dt_string}.png`{.docutils .literal
        .notranslate}.

        ::: {.admonition .seealso}
        See also

        `Model.broadcasting`{.docutils .literal .notranslate} and
        `Model.illustrate_history`{.docutils .literal .notranslate}
        :::

        Returns

        :   A `tensorflow.keras.History`{.docutils .literal
            .notranslate} object. Its History.history attribute is a
            record of training loss values and metrics values at
            successive epochs, as well as validation loss values and
            validation metrics values (if applicable).

        Return type

        :   tensorflow.keras.History\`

    `generator`{.sig-name .descname}[(]{.sig-paren}*[type_data]{.n}[:]{.p} [str]{.n}*, *[X]{.n}[:]{.p} [numpy.ndarray]{.n}*, *[Y]{.n}[:]{.p} [numpy.ndarray]{.n}*[)]{.sig-paren} → Type[\[]{.p}tensorflow.python.keras.preprocessing.image.NumpyArrayIterator[\]]{.p}[¶](#data_pipeline.train.Model.generator "Permalink to this definition"){.headerlink}

    :   **Returns a \`\`ImageDataGenerator.flow\`\` object**

        Such an object is an iterator containing data/label matrices
        used in training. The type of matrix are based on
        `type_data`{.docutils .literal .notranslate}.

        Parameters

        :   -   **type_data** (`str`{.docutils .literal .notranslate}.)
                -- A string specifying which type of data the generator
                generates.

            -   **X** (`np.ndarray`{.docutils .literal .notranslate}.)
                -- An array containing the data matrices.

        Returns

        :   A `ImageGenerator.flow`{.docutils .literal .notranslate}
            iterator object.

        Return type

        :   `NumpyArrayIterator`{.docutils .literal .notranslate}.

        ::: {.admonition .warning}
        Warning

        `type_data`{.docutils .literal .notranslate} must be either
        `['train', 'valid', 'eval']`{.docutils .literal .notranslate}.
        :::

        ::: {.admonition .seealso}
        See also

        [ImageDataGenerator source
        code](https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/image_data_generator.py){.reference
        .external}
        :::

    `illustrate_history`{.sig-name .descname}[(]{.sig-paren}*[history]{.n}[:]{.p} [Type[\[]{.p}tensorflow.python.keras.callbacks.History[\]]{.p}]{.n}*[)]{.sig-paren} → None[¶](#data_pipeline.train.Model.illustrate_history "Permalink to this definition"){.headerlink}

    :   **Generates plots of metrics over epochs.**

        The metrics over epochs plots are saved to
        `output/{metric}_{dt_string}.png`{.docutils .literal
        .notranslate}

        Parameters

        :   **history** -- A `tensorflow.keras.History`{.docutils
            .literal .notranslate} object. Its History.history attribute
            is a record of training loss values and metrics values at
            successive epochs, as well as validation loss values and
            validation metrics values (if applicable).

    `load_model`{.sig-name .descname}[(]{.sig-paren}*[model_path]{.n}[:]{.p} [str]{.n}*[)]{.sig-paren} → None[¶](#data_pipeline.train.Model.load_model "Permalink to this definition"){.headerlink}

    :   **This function loads a model from a given path.**

        Parameters

        :   **model_path** (`str`{.docutils .literal .notranslate}.) --
            A directory containing a tensforflow models object.

        Returns

        :   A Keras model instance.

        Return type

        :   `tensorflow.keras.model.Model`{.docutils .literal
            .notranslate}.

        ::: {.admonition .note}
        Note

        If a model is loaded `self.model`{.docutils .literal
        .notranslate} is set to `True`{.docutils .literal .notranslate}.
        :::

    `print_img`{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren} → None[¶](#data_pipeline.train.Model.print_img "Permalink to this definition"){.headerlink}

    :   **Saves a plot of 16 examples of input data.**

        The destination folder is:
        `'output/examples_{dt_string}.png'`{.docutils .literal
        .notranslate} where `dt_string`{.docutils .literal .notranslate}
        is the current data and hour in the format:
        `{ddmmyyyy-hh}`{.docutils .literal .notranslate}.
:::
:::

::: {#data-pipeline-data-generate-frame-py .section}
`data_pipeline.data.generate_frame.py`{.docutils .literal .notranslate}[¶](#data-pipeline-data-generate-frame-py "Permalink to this headline"){.headerlink}
-----------------------------------------------------------------------------------------------------------------------------------------------------------

Contains a class `FrameGenerator`{.docutils .literal .notranslate} which
generates tensors for training neural networks. All parameters are set
in `config.ini`{.docutils .literal .notranslate}.

-   (`generate_single_frame`{.docutils .literal .notranslate}) Generates
    a single pair of data, label and saves it.

-   (`run`{.docutils .literal .notranslate}) Generates a number of such
    pairs.

::: {#module-data_pipeline.data.generate_frame .section}
[]{#data-pipeline-data-generate-frame-framegenerator}

### `data_pipeline.data.generate_frame.FrameGenerator`{.docutils .literal .notranslate}[¶](#module-data_pipeline.data.generate_frame "Permalink to this headline"){.headerlink}

This module contains a class called `FrameGenerator`{.docutils .literal
.notranslate} which generates tensors with point scatterers placed in
them. All specifications are set in `config.ini`{.docutils .literal
.notranslate}.

*class* `data_pipeline.data.generate_frame.`{.sig-prename .descclassname}`FrameGenerator`{.sig-name .descname}[(]{.sig-paren}*[config_path]{.n}*, *[data_folder_path]{.n}*[)]{.sig-paren}[¶](#data_pipeline.data.generate_frame.FrameGenerator "Permalink to this definition"){.headerlink}

:   **This class creates training data in the form of 3D/4D tensors.**

    The training data consists of data and labels, the labels are sparse
    tensors while the data are tensors where the sparse tensors which
    are convolved with a given point spread function.

    `generate_single_frame`{.sig-name .descname}[(]{.sig-paren}*[gaussian_map]{.n}*, *[index]{.n}*, *[mode]{.n}*[)]{.sig-paren}[¶](#data_pipeline.data.generate_frame.FrameGenerator.generate_single_frame "Permalink to this definition"){.headerlink}

    :   **This function generates a single data, label pair and saves
        them.**

        The frame is made and filled with scatterers, from which the
        label is generated and saved.

        Parameters

        :   -   **gaussian_map** (`np.ndarray`{.docutils .literal
                .notranslate}) -- The gaussian map used for convolution.

            -   **index** (`int`{.docutils .literal .notranslate}.) --
                The execution index, see
                `generate_frame.FrameGenerator.run`{.docutils .literal
                .notranslate}.

            -   **mode** (`str`{.docutils .literal .notranslate}.) -- A
                string specifying wether the frame is for training,
                validation, evaluation.

    `run`{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[¶](#data_pipeline.data.generate_frame.FrameGenerator.run "Permalink to this definition"){.headerlink}

    :   **Creates data, label pairs and saves them to
        \`\`data_folder_path\`\`.**

        All parameters are set in `config.ini`{.docutils .literal
        .notranslate}. First it tries to create the
        `data_folder_path`{.docutils .literal .notranslate} directory,
        and the required subdirectories:
        `(training/validation/evaluation)/(data/labels)`{.docutils
        .literal .notranslate}, after which the pairs are created using
        `generate_frame.FrameGenerator.generate_single_frame`{.docutils
        .literal .notranslate}.
:::
:::

::: {#data-pipeline-models-unet-py .section}
`data_pipeline.models.unet.py`{.docutils .literal .notranslate}[¶](#data-pipeline-models-unet-py "Permalink to this headline"){.headerlink}
-------------------------------------------------------------------------------------------------------------------------------------------

::: {.line-block}
::: {.line}
This script contains a crude Tensorflow implementation of UNet, see:
[UNet](https://arxiv.org/abs/1505.04597){.reference .external}
:::

::: {.line}
with the hyperparameters set in `config.ini`{.docutils .literal
.notranslate}.
:::
:::
:::

::: {#data-pipeline-layers-layers-py .section}
`data_pipeline.layers.layers.py`{.docutils .literal .notranslate}[¶](#data-pipeline-layers-layers-py "Permalink to this headline"){.headerlink}
-----------------------------------------------------------------------------------------------------------------------------------------------

Implements certain types of layers not currently present in Tensorflow,
but which are part of the UNet neural network.
:::

::: {#data-pipeline-util-run-pipeline-py .section}
`data_pipeline.util.run_pipeline.py`{.docutils .literal .notranslate}[¶](#data-pipeline-util-run-pipeline-py "Permalink to this headline"){.headerlink}
-------------------------------------------------------------------------------------------------------------------------------------------------------

This script assumes you have a the following setup: \* A config files,
following the format of `config.ini`{.docutils .literal .notranslate},
**all** fields are required. \* A directory which contains the following
structure:

::: {.highlight-default .notranslate}
::: {.highlight}
    storage ------- training --------- data
                  |                  |
                  |                  ---- labels
                  |
                  ---- validation --------- data
                                    |
                                    ------- labels
:::
:::

**In the future a \`\`evaluation\`\` folder with subfolders \`\`data\`\`
and \`\`labels\`\` will be required to.** \* A directory which contains
the following structure\*:

::: {.highlight-default .notranslate}
::: {.highlight}
    mat_folder ------- train --------- data ----- [.mat-files]
                  |
                  |
                  ---- valid --------- data ----- [.mat-files]
:::
:::

\* **Note the difference in folder structures, this is to ensure that
the user does not interchange these folders.**

::: {.highlight-{bash} .notranslate}
::: {.highlight}
    python3 run_pipeline.py -conf 'config.ini' -s_dir 'storage' -m_dir 'dir/w/mat-files'
:::
:::

This scripts does the following operations: \* If the folder
`storage/training/data/`{.docutils .literal .notranslate} is not empty,
data from `dir/w/mat-files/train/data/`{.docutils .literal .notranslate}
is loaded in using `Loader.load_mat_folder`{.docutils .literal
.notranslate}. \* If the folder `storage/validation/data/`{.docutils
.literal .notranslate} is not empty, data from
`dir/w/mat-files/valid/data/`{.docutils .literal .notranslate} is loaded
in using `Loader.load_mat_folder`{.docutils .literal .notranslate}. \*
`Model.fit_model()`{.docutils .literal .notranslate} is executed, and
broadcasts an execution log to a `.log`{.docutils .literal .notranslate}
files contained in `output`{.docutils .literal .notranslate}, the naming
format is: `model_{ddmmyyyy_hh}.log`{.docutils .literal .notranslate}
with the date and hour of execution. \*
`Model.illustrate_history(history)`{.docutils .literal .notranslate} is
executed, where `history`{.docutils .literal .notranslate} is the
`tensorflow.Model.History`{.docutils .literal .notranslate} object of
the fitted model. The plots are saved to `output`{.docutils .literal
.notranslate}, the naming format is:
`accuracy_{ddmmyyyy_hh}.log`{.docutils .literal .notranslate} and
`loss_{ddmmyyyy_hh}.log`{.docutils .literal .notranslate} with the date
and hour of execution. \* `Model.print_img()`{.docutils .literal
.notranslate} is executed and saves examples images of data to
`output`{.docutils .literal .notranslate}, the naming format is:
`examples_{ddmmyyyy_hh}.log`{.docutils .literal .notranslate} with the
date and hour of execution.
:::

::: {#data-pipeline-config-config-ini .section}
`data_pipeline/config/config.ini`{.docutils .literal .notranslate}[¶](#data-pipeline-config-config-ini "Permalink to this headline"){.headerlink}
-------------------------------------------------------------------------------------------------------------------------------------------------

To run a complete pipeline all fields are required.

::: {.highlight-default .notranslate}
+-----------------------------------+-----------------------------------+
| ::: {.linenodiv}                  | ::: {.highlight}                  |
|      1                            |             # This sect           |
|      2                            | ion is for data generation, i.e.  |
|      3                            | data_pipeline.data.generate_frame |
|      4                            |     [DATA]                        |
|      5                            |     # Numb                        |
|      6                            | er of training data, label pairs. |
|      7                            |     NumDataTrain = 1000           |
|      8                            |     # Number                      |
|      9                            |  of validation data, label pairs. |
|     10                            |     NumDataValid = 500            |
|     11                            |     # Size of tensors.            |
|     12                            |     TargetSize = 128              |
|     13                            |     #                             |
|     14                            |  Number of scatterers in tensors. |
|     15                            |     NumScatter = 5                |
|     16                            |     #                             |
|     17                            | Indication of tensor is 3D or 4D. |
|     18                            |     Movement = True               |
|     19                            |     # Speed of scatterers.        |
|     20                            |     MovementVelocity = 10         |
|     21                            |     # How many f                  |
|     22                            | rames there are in the time-axis. |
|     23                            |     MovementDuration = 10         |
|     24                            |     # The angle along             |
|     25                            |  which the scatterers are moving. |
|     26                            |     MovementAngle = 90            |
|     27                            |     # The                         |
|     28                            |  parameters for the gaussian map. |
|     29                            |     GaussianSigma = 0.1           |
|     30                            |     GaussianMu = 0.0              |
|     31                            |                                   |
|     32                            |                                   |
|     33                            |             # This se             |
|     34                            | ction is for training parameters. |
|     35                            |     [TRAINING]                    |
|     36                            |     # Number of training epochs.  |
|     37                            |     Epochs = 15                   |
|     38                            |                                   |
|     39                            |                                   |
|     40                            |                                   |
|     41                            |        # This section is for para |
|     42                            | meters relating to preprocessing  |
|     43                            |             # of training data.   |
|     44                            |     [PREPROCESS_TRAIN]            |
|     45                            |     # Ratio between size of data  |
|     46                            |  and labels: data_size/label_size |
|     47                            |     SizeRatio = 2                 |
|     48                            |     # Resca                       |
|     49                            | ling factor when generator loads. |
|     50                            |     Rescale = 1.0                 |
|     51                            |     # Size of tensors.            |
|     52                            |     TargetSize = 128              |
|     53                            |     # Size of batches.            |
|     54                            |     BatchSize = 32                |
|     55                            |                                   |
|     56                            |                                   |
|     57                            |                                   |
|     58                            |        # This section is for para |
|     59                            | meters relating to preprocessing  |
|     60                            |             # of validation data. |
|     61                            |     [PREPROCESS_VALID]            |
|     62                            |     # Ratio between size of data  |
|     63                            |  and labels: data_size/label_size |
|     64                            |     SizeRatio = 1                 |
|     65                            |     # Resca                       |
|     66                            | ling factor when generator loads. |
|     67                            |     Rescale = 1.0                 |
|     68                            |     # Size of tensors.            |
|     69                            |     TargetSize = 128              |
|     70                            |     # Size of batches.            |
|     71                            |     BatchSize = 16                |
|     72                            |                                   |
|     73                            |                                   |
|     74                            |                                   |
|     75                            |        # This section is for para |
|     76                            | meters relating to preprocessing  |
|     77                            |             # of evaluation data. |
|     78                            |     [PREPROCESS_EVAL]             |
|     79                            |     # Ratio between size of data  |
|     80                            |  and labels: data_size/label_size |
|     81                            |     SizeRatio = 1                 |
|     82                            |     # Resca                       |
|     83                            | ling factor when generator loads. |
|     84                            |     Rescale = 1.0                 |
| :::                               |     # Size of tensors.            |
|                                   |     TargetSize = 128              |
|                                   |     # Size of batches.            |
|                                   |                                   |
|                                   |                                   |
|                                   |                                   |
|                                   | # This section is for parameters  |
|                                   | relating the model. ATM these are |
|                                   |             # tai                 |
|                                   | lored to the UNet implementation. |
|                                   |     [MODEL]                       |
|                                   |     NumFilter = 64                |
|                                   |     Padding = same                |
|                                   |     OutputActivation = sigmoid    |
|                                   |     UpSampling = subpixel         |
|                                   |     DownSampling = maxpool        |
|                                   |     Activation = lrelu            |
|                                   |     Conv1DLater = True            |
|                                   |     Initializer = glorot_uniform  |
|                                   |     BatchNorm = True              |
|                                   |     Shortcut = True               |
|                                   |     DropOutRate = 0.3             |
|                                   |     DropOutWarning = False        |
|                                   |     SkipConnection = add          |
|                                   |     DropOutEncoderOnly = False    |
|                                   | :::                               |
+-----------------------------------+-----------------------------------+
:::
:::

::: {#todo .section}
TODO[¶](#todo "Permalink to this headline"){.headerlink}
--------------------------------------------------------

-   A more flexible input shape handling, allowing for both single 3d
    data frames, and 4d data.

-   Do docstrings for all functionalities.

-   Automatic docstring broadcasting to wiki-page.

-   Broadcasting of examples should also be of labels, not only data.

-   The `Model.illustrate_history(history)`{.docutils .literal
    .notranslate} should be of all used metrics, not only
    `accuracy`{.docutils .literal .notranslate} and `loss`{.docutils
    .literal .notranslate}.
:::
:::
:::
:::
:::

::: {.sphinxsidebar role="navigation" aria-label="main navigation"}
::: {.sphinxsidebarwrapper}
[data_pipeline](index.html) {#data_pipeline-1 .logo}
===========================

### Navigation

[Contents]{.caption-text}

-   [data_pipeline](#){.current .reference .internal}
    -   [`data_pipeline.util.loader_mat.py`{.docutils .literal
        .notranslate}](#data-pipeline-util-loader-mat-py){.reference
        .internal}
    -   [`data_pipeline.train.py`{.docutils .literal
        .notranslate}](#data-pipeline-train-py){.reference .internal}
    -   [`data_pipeline.data.generate_frame.py`{.docutils .literal
        .notranslate}](#data-pipeline-data-generate-frame-py){.reference
        .internal}
    -   [`data_pipeline.models.unet.py`{.docutils .literal
        .notranslate}](#data-pipeline-models-unet-py){.reference
        .internal}
    -   [`data_pipeline.layers.layers.py`{.docutils .literal
        .notranslate}](#data-pipeline-layers-layers-py){.reference
        .internal}
    -   [`data_pipeline.util.run_pipeline.py`{.docutils .literal
        .notranslate}](#data-pipeline-util-run-pipeline-py){.reference
        .internal}
    -   [`data_pipeline/config/config.ini`{.docutils .literal
        .notranslate}](#data-pipeline-config-config-ini){.reference
        .internal}
    -   [TODO](#todo){.reference .internal}

::: {.relations}
### Related Topics

-   [Documentation overview](index.html)
    -   Previous: [Documentation](index.html "previous chapter")
:::

::: {#searchbox style="display: none" role="search"}
### Quick search {#searchlabel}

::: {.searchformwrapper}
:::
:::
:::
:::

::: {.clearer}
:::
:::

::: {.footer}
©2020, Anders Jess Pedersen. \| Powered by [Sphinx
3.2.1](http://sphinx-doc.org/) & [Alabaster
0.7.12](https://github.com/bitprophet/alabaster) \| [Page
source](_sources/includeme.rst.txt)
:::
