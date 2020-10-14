data_pipeline
=============

``util/loader_mat.py``
----------------------

Contains the class ``Loader`` which has methods for loading ``.mat``
files containing output MATLAB ``Struct``\ ’s generated during
simulation and from the SARUS scanner.

``train.py``
------------

Contains the class ``Model`` which provides a crude interface for:

-  (``load_mat``) Ingesting output of the ``Loader`` class of
   ``loader_mat.py``.
-  (``print_img``) Visualizing data/labels.
-  (``illustrate_history``) Visualizing the metrics over training
   epochs.
-  (``fit_model``) If not done beforehand, either loads a pretrained
   model or builds, compiles, provides data generators and fits a model
   provided by the ``build_model.py`` script, see below.

``train.Model``
^^^^^^^^^^^^^^^
.. automodule:: data_pipeline.train
   :members:

``data/generate_frame.py``
--------------------------

Contains a class ``FrameGenerator`` which generates pairs of data, label
tensors for training neural networks.

``data.generate_frame.Model``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: data_pipeline.data.generate_frame
   :members:

``util/generator.py``
---------------------

Contains a class ``DataGenerator`` which yields pairs of data, label
tensors for training neural networks.

``data.generator.DataGenerator``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: data_pipeline.util.DataGenerator
   :members:

``models/unet.py``
------------------

| This script contains a crude Tensorflow implementation of UNet, see:
  `UNet <https://arxiv.org/abs/1505.04597>`__
| with the hyperparameters set in ``config.ini``.

``layers/layers.py``
--------------------

Implements certain types of layers not currently present in Tensorflow,
but which are part of the UNet neural network.

``util/run_pipeline.py``
------------------------

This script assumes you have a the following setup: \* A config files,
following the format of ``config.ini``, **all** fields are required. \*
A directory which contains the following structure:

::

   storage ------- training --------- data  
                 |                  |  
                 |                  ---- labels  
                 |  
                 ---- validation --------- data  
                                   |  
                                   ------- labels

**In the future a ``evaluation`` folder with subfolders ``data`` and
``labels`` will be required to.** \* A directory which contains the
following structure*:

::

   mat_folder ------- train --------- data ----- [.mat-files]
                 |                   
                 |  
                 ---- valid --------- data ----- [.mat-files]

\* **Note the difference in folder structures, this is to ensure that
the user does not interchange these folders.**

Example usage:
~~~~~~~~~~~~~~

.. code:: {bash}

   python3 run_pipeline.py -conf 'config.ini' -s_dir 'storage' -m_dir 'dir/w/mat-files'

This scripts does the following operations: \* If the folder
``storage/training/data/`` is not empty, data from
``dir/w/mat-files/train/data/`` is loaded in using
``Loader.load_mat_folder``. \* If the folder
``storage/validation/data/`` is not empty, data from
``dir/w/mat-files/valid/data/`` is loaded in using
``Loader.load_mat_folder``. \* ``Model.fit_model()`` is executed, and
broadcasts an execution log to a ``.log`` files contained in ``output``,
the naming format is: ``model_{ddmmyyyy_hh}.log`` with the date and hour
of execution. \* ``Model.illustrate_history(history)`` is executed,
where ``history`` is the ``tensorflow.Model.History`` object of the
fitted model. The plots are saved to ``output``, the naming format is:
``accuracy_{ddmmyyyy_hh}.log`` and ``loss_{ddmmyyyy_hh}.log`` with the
date and hour of execution. \* ``Model.print_img()`` is executed and
saves examples images of data to ``output``, the naming format is:
``examples_{ddmmyyyy_hh}.log`` with the date and hour of execution.

``config/config.ini``
---------------------
When running a full pipeline all fields are required in the config file.
Below an example config file can be seen.

.. literalinclude:: ../config/config.ini

TODO
----

-  A more flexible input shape handling, allowing for both single 3d
   data frames, and 4d data.
-  Do docstrings for all functionalities.
-  Automatic docstring broadcasting to wiki-page.
-  Broadcasting of examples should also be of labels, not only data.
-  The ``Model.illustrate_history(history)`` should be of all used
   metrics, not only ``accuracy`` and ``loss``.
