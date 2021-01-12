import os
import random
import logging
import importlib
import configparser
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorflow import keras
from typing import Type
from datetime import datetime
from models.feedback import Feedback
from util import clean_storage_folder as cleaner
from util.loader_mat import Loader
from util.generator import DataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend
from tensorflow.keras.callbacks import History
from matplotlib.animation import FuncAnimation, PillowWriter
from data.generate_candidate_tracks import CandidateTrackGenerator



class Compiler():

    def __init__(self, data_folder_path, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.config_path = config_path
        self.data_folder_path = data_folder_path
        self.train_path = os.path.join(self.data_folder_path, 'training')
        self.valid_path = os.path.join(self.data_folder_path, 'validation')
        self.eval_path = os.path.join(self.data_folder_path, 'evaluation')
        self.loader = Loader()
        self.candidate_track_generator = CandidateTrackGenerator(config_path)
        self._set_runtime_variables()
        self._allocate_memory()
        self._reset_seeds()
        self._make_logs()
        self._load()
        self._make_generators()
        self._set_dtype()

    def _set_dtype(self):
        tf.keras.backend.set_floatx('float64')

    def _set_runtime_variables(self):
        self.batch_size = self.config['PREPROCESS_TRAIN'].getint('BatchSize')
        num_scatter_tmp = self.config['DATA'].get('NumScatterTrain')
        self.num_scat =  [int(y) for y in num_scatter_tmp.split(',')][0]
        self.max_cand = self.config['RNN'].getint('MaximalCandidate')
        self.time = self.config['DATA'].getint('MovementDuration')
        self.num_data_train = self.config['DATA'].getint('NumDataTrain')
        self.num_data_val = self.config['DATA'].getint('NumDataValid')
        self.warm_up_length = self.config['RNN'].getint('WarmUpLength')
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

    def build_model(self) -> None:
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
                self.model = class_(self.config_path)
        else:
            print('Model is already loaded.')

    def _load(self):
        self.train_tracks = self.loader.load_array_folder(
            source_path = os.path.join(self.train_path, 'tracks'),
            type_of_data = 'tracks',
        )
        self.eval_tracks = self.loader.load_array_folder(
            source_path = os.path.join(self.eval_path, 'tracks'),
            type_of_data = 'tracks',
        )
        self.val_tracks = self.loader.load_array_folder(
            source_path = os.path.join(self.valid_path, 'tracks'),
            type_of_data = 'tracks',
        )

    def _make_generators(self):
        modes = ['training', 'validation', 'evaluation']
        generators = []
        for mode in modes:
            if mode == 'training':
                X = self.train_tracks
                Y = self.train_tracks
            elif mode == 'validation':
                X = self.val_tracks
                Y = self.val_tracks
            else:
                X = self.eval_tracks
                Y = self.eval_tracks
            generators.append(DataGenerator(X, Y, mode, self.config).flow())
        self.train_gen, self.val_gen, self.eval_gen = generators

    def train_step(self, x, y):
        # Make candidate tracks.
        # x_batch: (batch_size, num_scatterer, coords, time, channel)
        window_data = self.candidate_track_generator.make_candidate_tracks(x)
        Y = tf.squeeze(tf.transpose(y, (0, 1, 3, 2, 4)))
        self.X = self._make_tensors(window_data)
        with tf.GradientTape() as tape:
            tape.watch(self.X)
            logits = self.model.call(self.X)
            loss = self.custom_loss(Y, logits)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss

    def test_step(self, x, y):
        # Make candidate tracks.
        # x_batch: (batch_size, num_scatterer, coords, time, channel)
        window_data = self.candidate_track_generator.make_candidate_tracks(x)
        Y = tf.squeeze(tf.transpose(y, (0, 1, 3, 2, 4)))
        self.X = self._make_tensors(window_data)
        logits = self.model.call(self.X, training=False)
        loss = self.custom_loss(Y, logits)
        return loss

    def eval_step(self, x, y):
        # Make candidate tracks.
        # x_batch: (batch_size, num_scatterer, coords, time, channel)
        window_data = self.candidate_track_generator.make_candidate_tracks(x)
        Y = tf.squeeze(tf.transpose(y, (0, 1, 3, 2, 4)))
        self.X = self._make_tensors(window_data)
        logits = self.model.call(self.X, training=False)
        loss = self.custom_loss(Y, logits)
        reshaped_logits = self._reshape(logits)
        return loss, reshaped_logits, Y

    def training_loop(self):
        self.build_model()
        epochs = self.config['TRAINING'].getint('Epochs')
        learning_rate = self.config['TRAINING'].getfloat('LearningRate')
        self.optimizer = Adam(learning_rate=learning_rate)
        history = {}
        history['loss'] = []
        history['val_loss'] = []
        for epoch in range(epochs):
            print('Start of epoch: ' + str(epoch))
            steps_per_epoch = int(self.num_data_train / self.batch_size)
            for step in tqdm(range(steps_per_epoch)):
                x_batch, y_batch = next(self.train_gen)
                loss = self.train_step(x_batch, y_batch)
            history['loss'].append(loss)
            print('Loss at end of epoch: ' + str(loss))

            steps_per_epoch_val= int(self.num_data_val / self.batch_size)
            for step in tqdm(range(steps_per_epoch_val)):
                x_batch, y_batch = next(self.val_gen)
                loss = self.test_step(x_batch, y_batch)
            history['val_loss'].append(loss)
            print('Validation loss at end of epoch: ' + str(loss))
        self.model.save(f'model_{self.dt_string}')
        return history


    def evaluate(self):
        x_batch, y_batch = next(self.eval_gen)
        return self.eval_step(x_batch, y_batch)



    def _make_tensors(self, window_data):
        X = np.full((self.batch_size * self.num_scat * self.max_cand, self.time, 2), -1.0)
        for b in range(self.batch_size):
            for s in range(self.num_scat):
                for c in range(len(window_data[b,s])):
                    X[b+s+c] = window_data[b,s][c]
        X = tf.convert_to_tensor(X)
        return X

    def _reshape(self, y):
        y = tf.split(y, self.max_cand, axis=0)
        y = tf.stack(y, axis=0)
        y = tf.split(y, self.num_scat, axis=1)
        y = tf.stack(y, axis=2)
        return y

    def custom_loss(self, y_actual, y_predicted):
        inputs = self.X
        y_actual = y_actual[:,:,:-self.warm_up_length]
        y_predicted = self._reshape(y_predicted)
        inputs = self._reshape(inputs)

        null_tensor = tf.convert_to_tensor(np.full((self.batch_size, self.num_scat, self.time, 2), -1.0))
        init = tf.convert_to_tensor(np.zeros((self.batch_size, self.num_scat, self.time-self.warm_up_length)))
        for i, y in enumerate(y_predicted):
            if not tf.reduce_all(tf.equal(inputs[i], null_tensor)):
                init += tf.losses.mean_squared_error(y, y_actual)
        loss = tf.reduce_sum(init)
        return loss

    def compile_and_fit(self):
        history = self.training_loop()
        self.illustrate_history(history)
        backend.clear_session()
        return history


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
