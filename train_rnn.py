import os
import random
import logging
import importlib
import configparser
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from typing import Type
from datetime import datetime
from models.feedback import Feedback
from util import clean_storage_folder as cleaner
from util.organiser import Organiser
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend
from tensorflow.keras.callbacks import History
from matplotlib.animation import FuncAnimation, PillowWriter


class Compiler():

    def __init__(self, data_folder_path, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.config_path = config_path
        self.data_folder_path = data_folder_path
        self.organiser = Organiser(data_folder_path, config_path)
        self._set_runtime_variables()
        self._allocate_memory()
        self._reset_seeds()
        self._make_logs()
        self._load_data()
        self._set_dtype()

    def _set_dtype(self):
        tf.keras.backend.set_floatx('float64')

    def _set_runtime_variables(self):
        self.train_X = None
        self.train_Y = None
        self.valid_X = None
        self.valid_Y = None
        self.eval_X = None
        self.eval_Y = None
        self.loaded_model = False
        self.fitted = False
        self.model_compiled = False


    # Utility functions for TF
    def _allocate_memory(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
          try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
          except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    def _reset_seeds(self):
        seed = self.config['PIPELINE'].getint('Seed')
        backend.clear_session()
        tf.compat.v1.reset_default_graph()
        print("KERAS AND TENSORFLOW GRAPHS RESET")

        np.random.seed(seed)
        random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        print("RANDOM SEEDS RESET") 
    
    def _clean_log_dir(self):
        cleaner.clean_logs('output/logs')

    def _make_logs(self):
        self._clean_log_dir()
        now = datetime.now()
        self.dt_string = now.strftime("%Y%m%d-%H%M%S")
        logging.basicConfig(
                    filename=f'output/logs/model_{self.dt_string}.log',
                    level=logging.INFO
        )

    def build_model(self, generator) -> None:
        """
        **This method builds a model if none is loaded.**

        The build model is stored in ``self.model``, unless 
        ``self.loaded_model == True``, see note of ``Model.load_model``.
        """

        model_name = self.config['RNN'].get('ModelName')
        tmp = model_name.split('.')
        module = '.'.join(tmp[:2])
        class_ = tmp[-1]
        if not self.loaded_model:
            print('Building model... \n')
            model = importlib.import_module(
                module,
                package=None
            )
            if self.config['RNN'].getboolean('CustomModel'):
                self.model = getattr(model, class_)
            else:
                class_ = getattr(model, class_)
                self.model = class_(
                    generator,
                    self.config_path
                )
        else:
            print('Model is already loaded.')

    def _load_data(self):
        pass

    def compile_and_fit(self):
        generator = self.organiser._make_batch('training')
        val_generator = self.organiser._make_batch('validation')
        self.build_model(
            generator
        )
        #self.model.summary()
        epochs = self.config['TRAINING'].getint('Epochs')
        learning_rate = self.config['TRAINING'].getfloat('LearningRate')

        cp_callback, early_stopping = self._checkpoints()
        self.model.compile(
            loss=self.custom_loss,
            optimizer=Adam(learning_rate=learning_rate),
            run_eagerly=True
        )
        print(self.model)
        history = self.model.fit_generator(
            generator,
            epochs=epochs,
            callbacks=[cp_callback, early_stopping],
            #batch_size=self.config['PREPROCESS_TRAIN'].getint('BatchSize'),
            verbose=1,
            #validation_data=val_generator,
        )
        self.fitted = True
        self.illustrate_history(history)
        self.model.save(f'model_{self.dt_string}')
        backend.clear_session()
        return history

    def custom_loss(y_actual, y_predicted):
        print('ok')


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
        plt.figure()
        for key in metrics:
            plt.plot(history.history[f'{key}'])
            plt.plot(history.history[f'val_{key}'])
            plt.title(f'model {key}')
            plt.ylabel(f'{key}')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(f'output/{key}_{self.dt_string}.png')


    def predict(self, mode: str) -> list:
        assert self.fitted, 'Model is not fitted.'
        if mode == 'training':
            data = self.train_X
            label = self.train_Y
        elif mode == 'validation':
            data = self.valid_X
            label = self.valid_Y
        elif mode == 'evaluation':
            data = self.eval_X
            label = self.eval_Y

        batch_size = self.config['PREPROCESS_TRAIN'].getint('BatchSize')
        batch = data[0:batch_size]
        label_batch = label[0:batch_size]
        predicted_batch = self.model.predict(batch)
        self.predicted_data = True
        return batch, predicted_batch, label_batch

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
            self.model = reconstructed_model
        except FileNotFoundError:
            return FileNotFoundError

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        latest = tf.train.latest_checkpoint(checkpoint_path)
        self.build_model()
        self.fitted = True
        self.model.load_weights(latest)

    def _checkpoints(self):
        checkpoint_path = "./output/checkpoints/cp-{epoch:04d}.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
           checkpoint_path,
           verbose=1,
           save_weights_only=True,
           save_best_only=True
        )
        patience = self.config['TRAINING'].getint('Patience')
        early_stopping = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=patience
        )

        return cp_callback, early_stopping

    def plot_predictions(self, amount, mode: str) -> None:
        assert mode in ['training', 'validation', 'evaluation']
        assert amount < self.config['PREPROCESS_TRAIN'].getint('BatchSize')

        batch, predicted_batch, label_batch = self.predict(mode)
        batch_frame_stacks = self._put_on_frame(batch[:amount])
        predicted_frame_stacks = self._put_on_frame(predicted_batch[:amount])
        label_frame_stacks = self._put_on_frame(label_batch[:amount])
        self._make_animation(
            amount,
            mode,
            batch_frame_stacks,
            predicted_frame_stacks,
            label_frame_stacks
        )

    def _make_animation(
        self,
        amount,
        mode,
        batch_frame_stacks,
        predicted_frame_stacks,
        label_frame_stacks
        ):

        nrows = amount
        ncols = 4
        fig = plt.gcf()
        fig.set_size_inches(ncols * 4, nrows * 4)
        fig.patch.set_facecolor('white')

        duration = len(batch_frame_stacks)
        def _update(batch, predict, label, index):
            length = len(batch)
            for k, type_ in enumerate([batch, predict, label]):
                for i in range(length):
                    sp = plt.subplot(nrows, ncols, k*length + (i + 1))
                    plt.imshow(type_[i][:,:,index])

        ani = FuncAnimation(
            fig,
            lambda i: _update(
                    batch_frame_stacks,
                    predicted_frame_stacks,
                    label_frame_stacks,
                    i
                ),
            list(range(duration)),
            init_func=_update(
                    batch_frame_stacks,
                    predicted_frame_stacks,
                    label_frame_stacks,
                    0
                )
        )  
        writer = PillowWriter(fps=duration)  
        ani.save(f"output/examples_{mode}_{self.dt_string}.gif", writer=writer) 


    def _put_on_frame(self, data):
        stacks = data.shape[0]
        frames = data.shape[1]

        frame_size = self.config['DATA'].getint('TargetSize')
        batch_frame_stacks = []
        for stack in range(stacks):
            frame_stack = np.zeros((frame_size, frame_size, frames))
            for frame in range(frames):
                x = int(data[stack, frame, 0])
                y = int(data[stack, frame, 1])
                frame_stack[x, y, frame] = 1.0
            batch_frame_stacks.append(frame_stack)
        return batch_frame_stacks
