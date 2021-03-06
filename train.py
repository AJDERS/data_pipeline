"""
This module contains a class called `Model` which implements an interface for 
loading parsed ``.mat`` files, building, compiling, fitting and inspect the
the results of tensorflow ``tensorflow.keras.models``. 
"""

import os
import keras
import logging
import random
import configparser
import importlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import Type, Generator
from datetime import datetime
from util.callback import Callback
from util.loader_mat import Loader
from util.generator import DataGenerator
from tensorflow.keras.callbacks import History
from tensorflow.python.keras import backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from matplotlib.animation import FuncAnimation, PillowWriter


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
        self.config_path = config_path
        self.data_folder_path = data_folder_path
        self.train_path = os.path.join(self.data_folder_path, 'training')
        self.valid_path = os.path.join(self.data_folder_path, 'validation')
        self.eval_path = os.path.join(self.data_folder_path, 'evaluation')
        self.callback = Callback()
        self.loader = Loader()
        self.model = None
        self.loaded_model = False
        self.model_compiled = False
        self.predicted_data = False
        self.fitted = False
        self.train_X = None
        self.train_Y = None
        self.valid_X = None
        self.valid_Y = None
        self.eval_X = None
        self.eval_Y = None
        self.train_generator = None
        self.valid_generator = None
        self._reset_seeds()

    def _reset_seeds(self):
        seed = self.config['PIPELINE'].getint('Seed')
        backend.clear_session()
        tf.compat.v1.reset_default_graph()
        print("KERAS AND TENSORFLOW GRAPHS RESET")

        np.random.seed(seed)
        random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        print("RANDOM SEEDS RESET")

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

        # Directory with our testing data arrays
        eval_data = os.path.join(self.eval_path, 'data')
        eval_data_names = os.listdir(eval_data)
        eval_data_dirs = [os.path.join(eval_data,fname) for
            fname in eval_data_names]

        # Directory with our training label arrays
        train_label = os.path.join(self.train_path, 'labels')
        train_label_names = os.listdir(train_label)
        train_label_dirs = [os.path.join(train_label,fname) for
            fname in train_label_names]

        # Directory with our testing data arrays
        valid_label = os.path.join(self.valid_path, 'labels')
        valid_label_names = os.listdir(valid_label)
        valid_label_dirs = [os.path.join(valid_label,fname) for
            fname in valid_label_names]

        # Directory with our testing data arrays
        eval_label = os.path.join(self.eval_path, 'labels')
        eval_label_names = os.listdir(eval_label)
        eval_label_dirs = [os.path.join(eval_label,fname) for
            fname in eval_label_names]

        return [
            train_data_dirs,
            valid_data_dirs,
            eval_data_dirs,
            train_label_dirs,
            valid_label_dirs,
            eval_label_dirs
        ]

    def generator(self,
        mode: str,
        X: np.ndarray,
        Y: np.ndarray) -> Generator[np.ndarray, None, None]:
        """
        **Returns a DataGenerator.flow() generator object**

        Such an object is an iterator containing data/label matrices used in
        training. The type of matrix are based on ``mode``.

        :param mode: A string specifying which type of data the generator
            generates.
        :type mode: ``str``.
        :param X: An array containing the data matrices.
        :type X: ``np.ndarray``.
        :returns: A ``ImageGenerator.flow`` iterator object.
        :rtype: ``NumpyArrayIterator``.

        .. warning:: ``mode`` must be either 
            ``['training', 'validation', 'evaluation']``.
        .. seealso:: ``util.generator.DataGenerator``.
        """
        print(f'Defining {mode} data generator... \n')
        generator = DataGenerator(X, Y, mode, self.config)
        flow_generator = generator.flow()
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
            self.fitted = True
            reconstructed_model.summary()
            return reconstructed_model
        except FileNotFoundError:
            return FileNotFoundError

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        latest = tf.train.latest_checkpoint('output/checkpoints')
        self.build_model()
        self.fitted = True
        self.model.load_weights(latest)

    def print_img(self, mode: str) -> None:
        """
        **Saves a plot of 8 examples of input data.**

        The destination folder is:
        ``'output/examples_{dt_string}.png'``
        where ``dt_string`` is the current data and hour in the format:
        ``{ddmmyyyy-hh}``.
        """
        [
            train_data_dirs,
            valid_data_dirs,
            eval_data_dirs,
            train_label_dirs,
            valid_label_dirs,
            eval_label_dirs
        ] = self._get_data_dirs()
        if mode == 'training':
            dirs = train_data_dirs
            l_dirs = train_label_dirs
        elif mode == 'validation':
            dirs = valid_data_dirs
            l_dirs = valid_label_dirs
        elif mode == 'evaluation':
            dirs = eval_data_dirs
            l_dirs = eval_label_dirs

        nrows = 4
        ncols = 4
        fig = plt.gcf()
        fig.set_size_inches(ncols * 4, nrows * 4)
        fig.patch.set_facecolor('white')
        
        pic_index = random.randint(0, len(dirs))
        if pic_index < 8:
            pic_index = 0
        if pic_index > len(dirs) - 8:
            pic_index = len(dirs) - 8
        next_pix = dirs[pic_index-8:pic_index] + l_dirs[pic_index-8:pic_index]


        if self.config['DATA'].getboolean('Movement'):
            # Generates frames.
            def _update(next_pix, index):
                for i, img_path in enumerate(next_pix):
                    # Set up subplot; subplot indices start at 1
                    name = '/'.join(img_path.split('/')[-2:])
                    sp = plt.subplot(nrows, ncols, i + 1)
                    sp.set_title(f'{name}')
                    array = self.loader.decompress_load(img_path)

                    msg = 'Movement = True, but data does not have rank 3.'
                    assert len(array.shape) == 3, msg
                    plt.imshow(array[:,:,index])

            duration = self.config['DATA'].getint('MovementDuration')
            ani = FuncAnimation(
                fig,
                lambda i: _update(next_pix, i),
                list(range(duration)),
                init_func=_update(next_pix, 0)
            )  
            writer = PillowWriter(fps=duration)  
            ani.save(f"output/examples_{mode}_{dt_string}.gif", writer=writer) 
        else:
            for i, img_path in enumerate(next_pix):
                # Set up subplot; subplot indices start at 1
                name = '/'.join(img_path.split('/')[-2:])
                sp = plt.subplot(nrows, ncols, i + 1)
                sp.set_title(f'{name}')
                array = self.loader.decompress_load(img_path)
                msg = 'Movement = False, but data does not have rank 2.'
                assert len(array.shape) == 2, msg
                plt.imshow(array)
            plt.savefig(f'output/examples_{mode}_{dt_string}.png')

    def build_model(self) -> None:
        """
        **This method builds a model if none is loaded.**

        The build model is stored in ``self.model``, unless 
        ``self.loaded_model == True``, see note of ``Model.load_model``.
        """
        if not self.loaded_model:
            print('Building model... \n')
            model = importlib.import_module(
                self.config['MODEL'].get('ModelName'),
                package=None
            )
            self.model = model.create_model(self.config)
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
                loss='mean_squared_error',
                optimizer=Adam(lr=0.001),
                metrics=['mean_squared_error'],
                run_eagerly=True
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
        ``train.Model.compile``, and the existence of generators mentioned
        in ``train.Model.generator``.

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
        self._reset_seeds()

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
                    'training',
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
                    'validation',
                    self.valid_X,
                    self.valid_Y
                )
        if not self.loaded_model:
            # step_per_epoch * batch_size = # number of datapoints
            batch_size = self.config['PREPROCESS_TRAIN'].getint('BatchSize')
            steps_per_epoch = int(self.train_X.shape[0] / batch_size)
            logging.info(f'steps per epoch: {steps_per_epoch}')
            logging.info(f'batch size: {batch_size}')
            history = self._fit(batch_size, steps_per_epoch)
            return history
        else:
            print('Model is already loaded.')

    def _checkpoints(self):
        checkpoint_path = "./output/checkpoints/cp-{epoch:04d}.ckpt"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
           checkpoint_path,
           verbose=1,
           save_weights_only=True,
           save_best_only=True
        )
        return cp_callback

    def _fit(self, batch_size: int, steps_per_epoch: int) -> Type[History]:
        """
        **Fits a predefined model with given parameters.**

        Fits the predefined model with the given parameters, either with data
        loaded in memory, or from
        """
        checkpoint = self._checkpoints()
        if not self.config['TRAINING'].getboolean('WithValidationGenerator'):
            print('Fitting model without validation generator... \n')
            if self.config['TRAINING'].getboolean('InMemory'):
                print('Fitting model with data in memory... \n')
                history = self.model.fit(
                    self.train_X,
                    self.train_Y,
                    batch_size = batch_size,
                    steps_per_epoch=steps_per_epoch,  
                    epochs=self.config['TRAINING'].getint('Epochs'),
                    verbose=1,
                    callbacks=[self.callback, checkpoint]
                )
            else:
                print('Fitting model with data from generator... \n')
                history = self.model.fit(
                    self.train_generator,
                    batch_size = batch_size,
                    steps_per_epoch=steps_per_epoch,  
                    epochs=self.config['TRAINING'].getint('Epochs'),
                    verbose=1,
                    callbacks=[self.callback, checkpoint]
                )
            self.fitted = True
            self.broadcast(history)
            self.model.save(f'model_{dt_string}')
            backend.clear_session()
            return history
        else:
            print('Fitting model with validation generator... \n')
            if self.config['TRAINING'].getboolean('InMemory'):
                print('Fitting model with data in memory... \n')
                history = self.model.fit(
                    self.train_X,
                    self.train_Y,
                    steps_per_epoch=steps_per_epoch, 
                    epochs=self.config['TRAINING'].getint('Epochs'),
                    verbose=1,
                    validation_data=(self.valid_X, self.valid_Y),
                    validation_steps=steps_per_epoch, 
                    callbacks=[self.callback, checkpoint]
                )
            else:
                print('Fitting model with data from generator... \n')
                history = self.model.fit(
                    self.train_generator,
                    steps_per_epoch=steps_per_epoch, 
                    epochs=self.config['TRAINING'].getint('Epochs'),
                    verbose=1,
                    validation_data=self.valid_generator,
                    validation_steps=steps_per_epoch, 
                    callbacks=[self.callback, checkpoint]
                )
            self.fitted = True
            self.broadcast(history)
            self.model.save(f'model_{dt_string}')
            backend.clear_session()
            return history

    def predict(self, mode: str) -> list:
        assert self.fitted, 'Model is not fitted.'
        self.predicted_data = True
        if mode == 'training':
            if self.train_X is not None:
                data = self.train_X
            else:
                data = self.loader.load_array_folder(
                    source_path = os.path.join(self.train_path,'data'),
                    type_of_data = 'data'
                )
        elif mode == 'validation':
            if self.valid_X is not None:
                data = self.valid_X
            else:
                data = self.loader.load_array_folder(
                    source_path = os.path.join(self.valid_path,'data'),
                    type_of_data = 'data'
                )
        elif mode == 'evaluation':
            if self.eval_X is not None:
                data = self.eval_X
            else:
                data = self.loader.load_array_folder(
                    source_path = os.path.join(self.eval_path,'data'),
                    type_of_data = 'data'
                )
        batch_size = self.config['PREPROCESS_TRAIN'].getint('BatchSize')
        batch = data[0:batch_size]
        predicted_batch = self.model.predict(batch)
        return batch, predicted_batch

    def compare_predict(self, mode: str) -> None:
        # Generates frames.
        batch, predicted_batch = self.predict(mode)
        nrows = 2
        ncols = 4
        fig = plt.gcf()
        fig.set_size_inches(ncols * 4, nrows * 4)
        fig.patch.set_facecolor('white')
        predicted_stacks = predicted_batch[0:4]
        stacks = batch[0:4]

        if self.config['DATA'].getboolean('Movement'):   
            def _update(predicted_stacks, stacks, index):
                for i in range(len(predicted_stacks)):
                    msg = 'Movement = True, but data does not have rank 5.'
                    assert len(predicted_stacks.shape) == 5, msg

                    name = f'Stack {i}.'
                    sp = plt.subplot(nrows, ncols, i + 1)
                    sp.set_title(f'{name}')
                    plt.imshow(stacks[i,:,:,index])

                    p_name = f'Predicted stack {i}.'
                    sp = plt.subplot(nrows, ncols, i + 5)
                    sp.set_title(f'{p_name}')
                    plt.imshow(predicted_stacks[i,:,:,index])

            duration = self.config['DATA'].getint('MovementDuration')
            ani = FuncAnimation(
                fig,
                lambda i: _update(predicted_stacks, stacks, i),
                list(range(duration)),
                init_func=_update(predicted_stacks, stacks, 0)
            )  
            writer = PillowWriter(fps=duration)  
            ani.save(f"output/compare_{mode}_{dt_string}.gif", writer=writer) 
        else:
            for i in range(len(predicted_stacks)):
                msg = 'Movement = False, but data does not have rank 4.'
                assert len(predicted_stacks.shape) == 4, msg

                name = f'Stack {i}.'
                sp = plt.subplot(nrows, ncols, i + 1)
                sp.set_title(f'{name}')
                plt.imshow(stacks[i])

                p_name = f'Predicted stack {i}.'
                sp = plt.subplot(nrows, ncols, i + 5)
                sp.set_title(f'{p_name}')
                plt.imshow(predicted_stacks[i])

            plt.savefig(f'output/compare_{mode}_{dt_string}.png')

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
        # Get metrics which are not `val_{metric}`:
        metrics = [key for key in history.history.keys() if '_' not in key]
        for key in metrics:
            plt.plot(history.history[f'{key}'])
            plt.plot(history.history[f'val_{key}'])
            plt.title(f'model {key}')
            plt.ylabel(f'{key}')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(f'output/{key}_{dt_string}.png')

    def _layer_shapes(self, model):
        for l in model.layers:
            print(l.output_shape)


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
        with open(self.config_path, 'r') as file:
            logging.info(f'Using config-file: {self.config_path}.')
            logging.info(f'{self.config_path}:')
            for line in file:
                logging.info(line)
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