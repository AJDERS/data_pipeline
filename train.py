"""
This module contains a class called `Model` which implements an interface for 
loading parsed ``.mat`` files, building, compiling, fitting and inspect the
the results of tensorflow ``tensorflow.keras.models``. 
"""

import os
import keras
import logging
import configparser
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import unet as UNet
from typing import Type
from datetime import datetime
from callback import Callback
from loader_mat import Loader
from tensorflow.keras.callbacks import History
from tensorflow.keras.preprocessing.image import NumpyArrayIterator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

now = datetime.now()
dt_string = now.strftime("%d%m%Y_%H")
logging.basicConfig(
            filename=f'output/model_{dt_string}.log',
            level=logging.INFO
        )

class Model():
    """
    **This class loads, builds, compiles and fits tensorflow models.**

    Furthermore it has methods for creating data generators for these models,
    inspecting metrics over epochs, inspecting runtime information and input 
    data. 

    The specifications of the input data is set in ``config.ini``.
    """

    def __init__(self, data_folder_path, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.data_folder_path = data_folder_path
        self.train_path = os.path.join(self.data_folder_path, 'training')
        self.valid_path = os.path.join(self.data_folder_path, 'validation')
        self.eval_path = os.path.join(self.data_folder_path, 'evaluation')
        self.callback = Callback()
        self.loader = Loader()
        self.model = None
        self.loaded_model = False
        self.model_compiled = False
        self.with_validation_gen = True
        self.train_X = None
        self.train_Y = None
        self.valid_X = None
        self.valid_Y = None
        self.train_generator = None
        self.valid_generator = None

    def _get_data_dirs(self) -> tuple:
        """
        **Returns directories to training/validation data.**

        This is done based on the variables ``self.train_path`` and 
        ``self.valid_path``

        :returns: ``os.path.join(self.train_path, filename),
                    os.path.join(self.train_path, filename)`` for all filenames.
        :rtype: list
        """

        # Directory with our training data arrays
        train_data = os.path.join(self.train_path, 'data')
        train_data_names = os.listdir(train_data)
        train_data_dirs = [os.path.join(train_data,fname) for
            fname in train_data_names]

        # Directory with our testing data arrays
        valid_data = os.path.join(self.valid_path, 'data')
        valid_data_names = os.listdir(valid_data)
        valid_data_dirs = [os.path.join(valid_data,fname) for
            fname in valid_data_names]

        # # Directory with our evaluation data arrays
        # eval_data = os.path.join(self.eval_path, 'data')
        # eval_data_names = os.listdir(eval_data)

        return train_data_dirs, valid_data_dirs

    def generator(self,
        type_data: str,
        X: np.ndarray,
        Y: np.ndarray) -> Type[NumpyArrayIterator]:
        """
        **Returns a ``ImageDataGenerator.flow`` object**

        Such an object is an iterator containing data/label matrices used in
        training. The type of matrix are based on ``type_data``.

        :param type_data: A string specifying which type of data the generator
            generates.
        :type type_data: ``str``.
        :param X: An array containing the data matrices.
        :type X: ``np.ndarray``.
        :returns: A ``ImageGenerator.flow`` iterator object.
        :rtype: ``NumpyArrayIterator``.

        .. warning:: ``type_data`` must be either ``['train', 'valid', 'eval']``.
        .. seealso:: `ImageDataGenerator source code <https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/image_data_generator.py>`_
        """
        assert type_data in ['train', 'valid', 'eval']
        if type_data == 'train':
            preprocess = self.config['PREPROCESS_TRAIN']
            batch_size = preprocess.getint('BatchSize')
        elif type_data == 'valid':
            preprocess = self.config['PREPROCESS_VALID']
            batch_size = preprocess.getint('BatchSize')
        else:
            preprocess = self.config['PREPROCESS_EVAL']
            batch_size = preprocess.getint('BatchSize')
        print(f'Defining {type_data} data generator... \n')
        generator = ImageDataGenerator(
            rescale=preprocess.getfloat('Rescale')
        )
        flow_generator = generator.flow(X, Y, batch_size)
        return flow_generator

    def load_model(self, model_path: str) -> None:
        """**This function loads a model from a given path.**

        :param model_path: A directory containing a tensforflow models object.
        :type model_path: ``str``.
        :returns: A Keras model instance.
        :rtype: ``tensorflow.keras.model.Model``.

        .. note:: If a model is loaded ``self.model`` is set to ``True``.
        """
        try:
            reconstructed_model = keras.models.load_model(model_path)
            self.loaded_model = True
            reconstructed_model.summary()
            return reconstructed_model
        except FileNotFoundError:
            return FileNotFoundError

    def print_img(self) -> None:
        """
        **Saves a plot of 16 examples of input data.**

        The destination folder is:
        ``'output/examples_{dt_string}.png'``
        where ``dt_string`` is the current data and hour in the format:
        ``{ddmmyyyy-hh}``.
        """
        nrows = 4
        ncols = 4
        fig = plt.gcf()
        fig.set_size_inches(ncols * 4, nrows * 4)
        train_data_dirs, valid_data_dirs = self._get_data_dirs()
        pic_index = 8
        next_train_pix = train_data_dirs[pic_index-8:pic_index]
        next_valid_pix = valid_data_dirs[pic_index-8:pic_index]

        for i, img_path in enumerate(next_train_pix+next_valid_pix):
            # Set up subplot; subplot indices start at 1
            name = '/'.join(img_path.split('/')[-2:])
            sp = plt.subplot(nrows, ncols, i + 1)
            sp.axis('Off') # Don't show axes (or gridlines)
            sp.set_title(f'{name}')
            _ = self.loader.decompress_load(img_path)
            
        plt.savefig(f'output/examples_{dt_string}.png')

    def build_model(self) -> None:
        """
        **This method builds a model if none is loaded.**

        The build model is stored in ``self.model``, unless 
        ``self.loaded_model == True``, see note of ``Model.load_model``.
        """
        if not self.loaded_model:
            print('Building model... \n')
            self.model = UNet.create_model(self.config)
        else:
            print('Model is already loaded.')

    def compile_model(self) -> None:
        """
        **This method compiles a model if none is loaded.**

        .. note:: If a model is compiled ``self.model_compiled`` is set to ``True``.
        """
        if not self.loaded_model:
            print('Compiling model... \n')
            self.model.compile(
                loss='mse',
                optimizer=RMSprop(lr=0.001),
                metrics=['accuracy']
            )
            self.model_compiled = True
        else:
            print('Model is already loaded.')

    def fit_model(self) -> Type[History]:
        """
        **This function completes a full network pipeline.**

        All of the below functionalities are completed if they are not already
        executed. The execution history is based on the booleans mentioned in
        ``Model.load_model``, ``Model.build_model``, and
        ``~train.Model.compile``, and the existence of generators mentioned
        in ``~train.Model.generator``.

        First this function builds a model, then compiles it, make a training
        and validation generator, fits the model and lastly broadcasts runtime
        information to ``output/loggin_{dt_string}.log``, and metrics over epoch to
        ``output/{metric}_{dt_string}.png``.

        .. seealso:: ``Model.broadcasting`` and ``Model.illustrate_history``

        :returns: A ``tensorflow.keras.History`` object. Its `History.history` attribute is a record of training loss values and metrics values at
            successive epochs, as well as validation loss values and validation
            metrics values (if applicable).

        :rtype: `tensorflow.keras.History``
        """
        if not self.model:
            self.build_model()
        
        if not self.model_compiled:
            self.compile_model()

        if not self.train_generator:
            if any(v is None for v in [self.train_X, self.train_Y]):
                self.train_X = self.loader.load_array_folder(
                    source_path = os.path.join(self.train_path,'data'),
                    type_of_data = 'data'
                )
                
                self.train_Y = self.loader.load_array_folder(
                    source_path = os.path.join(self.train_path, 'labels'),
                    type_of_data = 'labels',
                    size_ratio = self.config['PREPROCESS_TRAIN'].getint('SizeRatio')
                )
                
                self.train_generator = self.generator(
                    'train',
                    self.train_X,
                    self.train_Y
                )

        if not self.valid_generator:
            if any(v is None for v in [self.valid_X, self.valid_Y]):
                self.valid_X = self.loader.load_array_folder(
                    os.path.join(self.valid_path, 'data'),
                    type_of_data = 'data',
                )

                self.valid_Y = self.loader.load_array_folder(
                    os.path.join(self.valid_path, 'labels'),
                    type_of_data = 'labels',
                    size_ratio = self.config['PREPROCESS_VALID'].getint('SizeRatio')

                )

                self.valid_generator = self.generator(
                    'valid',
                    self.valid_X,
                    self.valid_Y
                )

        if not self.loaded_model:
            # step_per_epoch * batch_size = # number of datapoints
            batch_size = self.config['PREPROCESS_TRAIN'].getint('BatchSize')
            steps_per_epoch = int(self.train_X.shape[0] / batch_size)
            logging.info(f'steps per epoch: {steps_per_epoch}')
            logging.info(f'batch size: {batch_size}')
            if not self.with_validation_gen:
                print('Fitting model without validation generator... \n')
                history = self.model.fit(
                    self.train_generator,
                    batch_size = batch_size,
                    steps_per_epoch=steps_per_epoch,  
                    epochs=self.config['TRAINING'].getint('Epochs'),
                    verbose=1,
                    callbacks=[self.callback]
                )
                self.broadcast(history)
                self.model.save(f'model_{dt_string}')
                return history
            else:
                print('Fitting model with validation generator... \n')
                history = self.model.fit(
                    self.train_generator,
                    steps_per_epoch=steps_per_epoch, 
                    epochs=self.config['TRAINING'].getint('Epochs'),
                    verbose=1,
                    validation_data=self.valid_generator,
                    validation_steps=steps_per_epoch, # Note only uses half of validation data in each epoch
                    callbacks=[self.callback]
                )
                self.broadcast(history)
                self.model.save(f'model_{dt_string}')
                return history
        else:
            print('Model is already loaded.')

    def illustrate_history(self,
        history: Type[History]) -> None:
        """
        **Generates plots of metrics over epochs.**

        The metrics over epochs plots are saved to 
        ``output/{metric}_{dt_string}.png``

        :param history: A ``tensorflow.keras.History`` object. 
            Its `History.history` attribute is a record of training loss values
            and metrics values at successive epochs, as well as validation loss
            values and validation metrics values (if applicable).
        """
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'output/accuracy_{dt_string}.png')

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'output/loss_{dt_string}.png')

    def broadcast(self,
        history: Type[History]) -> None:
        """
        **This method broadcast training information to a log-file.**

        First a model summary is logged, and then the training metrics
        history is logged for each metric.

        :param history: A ``tensorflow.keras.History`` object. 
            Its `History.history` attribute is a record of training loss values
            and metrics values at successive epochs, as well as validation loss
            values and validation metrics values (if applicable).
        """
        self.model.summary(print_fn=lambda x: logging.info(x + '\n'))
        logging.info(history.history.keys())
        for key in history.history.keys():
            logging.info(f'History for {key}: \n')
            logging.info(history.history[key])

"""
.. warning:: This script assumes you have a the following setup:
     - A config file, following the format of `config.ini`, **all** fields are required.
     - A directory which contains the following structure:
     `storage/training/data`
     `storage/training/labels`
     `storage/validation/data`
     `storage/validation/labels`
     - A directory which contains the following structure:
     `mat_folder/train/data/[.mat]`
     `mat_folder/valid/data/[.mat]`
"""