import random
import importlib
import configparser
import numpy as np
import tensorflow as tf
from models.feedback import Feedback
from util.sliding_window import SlidingWindow
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend


class Compiler():

    def __init__(self, data_folder_path, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.config_path = config_path
        self.windows = SlidingWindow(data_folder_path, config_path)
        self.loaded_model = False
        self.build_model()
        self._allocate_memory()
        self._reset_seeds()

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


    def _make_windows(self):
        threshold = self.config['RNN'].getint('CandidateThreshold')
        self.windows._slinding_windows(self.windows.train_tracks, threshold)
        self.windows._slinding_windows_labels(self.windows.train_tracks)
        return self.windows._organize_windows()


    def build_model(self) -> None:
        """
        **This method builds a model if none is loaded.**

        The build model is stored in ``self.model``, unless 
        ``self.loaded_model == True``, see note of ``Model.load_model``.
        """

        model_name = self.config['MODEL'].get('ModelName')
        tmp = model_name.split('.')
        module = '.'.join(tmp[:1])
        class_ = tmp[-1]
        if not self.loaded_model:
            print('Building model... \n')
            model = importlib.import_module(
                module,
                package=None
            )
            self.model = getattr(model, class_)
        else:
            print('Model is already loaded.')

    def compile_and_fit(self):
        X, Y = self._make_windows()
        patience = self.config['TRAINING'].getint('Patience')
        epochs = self.config['TRAINING'].getint('Epochs')

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min'
        )
        self.model.compile(
            loss='mean_squared_error',
            optimizer=Adam(),
            metrics=['mean_squared_error'],
            run_eagerly=True
        )
        history = self.model.fit(
            X,
            epochs=epochs,
            validation_data=Y,
            callbacks=[early_stopping]
        )
        return history