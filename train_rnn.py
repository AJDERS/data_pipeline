import os
import random
import logging
import importlib
import configparser
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from shutil import copyfile
from tqdm import tqdm
from tensorflow import keras
from typing import Type
from datetime import datetime
from models.feedback import Feedback
from util import clean_storage_folder as cleaner
from util.loader_mat import Loader
from util.generator import DataGenerator
from util.tanh_normalization import TanhNormalizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adadelta
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
        self.normalizer = TanhNormalizer()
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
        self.target_size = self.config['DATA'].getint('TargetSize')
        self.cap_values = self.config['RNN'].getboolean('CapValues')
        self.patience_thres = self.config['TRAINING'].getint('Patience')
        self.with_checkpoints = self.config['RNN'].getboolean('WithCheckpoints')
        self.patience_decay = self.config['TRAINING'].getboolean('PatienceLRDecay')
        self.learning_rate = self.config['TRAINING'].getfloat('LearningRate')
        self.lr_decay_time = self.config['TRAINING'].getint('LearningRateDecayTime')
        self.lr_decay_rate = self.config['TRAINING'].getfloat('LearningRateDecayRate')
        self.patience = 0
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
    
    def _make_logs(self):
        now = datetime.now()
        self.dt_string = now.strftime("%Y%m%d-%H%M%S")
        os.mkdir(f'output/rnn_checkpoints/model_{self.dt_string}')
        copyfile('config/config.ini', f'output/rnn_checkpoints/model_{self.dt_string}/config.ini')
        logging.basicConfig(
                    filename=f'output/rnn_checkpoints/model_{self.dt_string}/model_{self.dt_string}.log',
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
        self.class_name = tmp[-1]
        if not self.loaded_model:
            print('Building model... \n')
            model = importlib.import_module(
                module,
                package=None
            )
            if self.config['RNN'].getboolean('CustomModel'):
                self.model = getattr(model, self.class_name)
            else:
                self.class_ = getattr(model, self.class_name)
                self.model = self.class_(self.config_path)
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
        if self.cap_values:
            y = self._cap_ground_truth(y)
        Y = tf.squeeze(tf.transpose(y, (0, 1, 3, 2, 4)))
        y_actual = Y[:,:,self.warm_up_length:]
        y_normed = self.normalizer.tanh_normalization(y_actual)
        X = self._make_tensors(window_data)
        self.X_ = self._reshape(X)
        self.X = self.normalizer.tanh_normalization(X)
        with tf.GradientTape() as tape:
            tape.watch(self.X)
            logits = self.model.call(self.X)
            loss = self.custom_loss(y_normed, logits)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss

    def test_step(self, x, y, epoch):
        # Make candidate tracks.
        # x_batch: (batch_size, num_scatterer, coords, time, channel)
        window_data = self.candidate_track_generator.make_candidate_tracks(x)
        if self.cap_values:
            y = self._cap_ground_truth(y)
        Y = tf.squeeze(tf.transpose(y, (0, 1, 3, 2, 4)))
        y_actual = Y[:,:,self.warm_up_length:]
        y_normed = self.normalizer.tanh_normalization(y_actual)
        X = self._make_tensors(window_data)
        self.X_ = self._reshape(X)
        self.X = self.normalizer.tanh_normalization(X)
        logits = self.model.call(self.X, training=False)
        loss = self.custom_loss(y_normed, logits)
        return loss

    def eval_step(self, x, y):
        # Make candidate tracks.
        # x_batch: (batch_size, num_scatterer, coords, time, channel)
        window_data = self.candidate_track_generator.make_candidate_tracks(x)
        if self.cap_values:
            y = self._cap_ground_truth(y)
        Y = tf.squeeze(tf.transpose(y, (0, 1, 3, 2, 4)))
        X = self._make_tensors(window_data)
        self.X_ = self._reshape(X)
        self.X = self.normalizer.tanh_normalization(X)
        logits = self.model.call(self.X, training=False)
        
        # Find loss for each candidate.
        predicted, cost, prob, inputs = [self._reshape(x) for x in logits]
        y_actual = Y[:,:,self.warm_up_length:]
        y_normed = self.normalizer.tanh_normalization(y_actual)
        null_tensor = tf.convert_to_tensor(np.full((self.time, 2), -1.0))
        candidate_picks = []
        for b in range(self.batch_size):
            for s in range(self.num_scat):       
                scatter_loss = [] 
                for i, y in enumerate(predicted):
                    if not tf.reduce_all(tf.equal(self.X_[i,b,s], null_tensor)):
                        compound_loss = self._compound_loss(
                            y[b,s],
                            y_normed[b,s],
                            inputs[i,b,s,self.warm_up_length:],
                            cost[i,b,s,0],
                            prob[i,b,s]
                        )
                        scatter_loss.append((i, compound_loss))
                candidate_picks.append((sorted(scatter_loss, key=lambda x: x[1])[0][0],b,s))
        return predicted, cost, prob, self._reshape(X), y_actual, candidate_picks

    def training_loop(self):
        self.build_model()
        epochs = self.config['TRAINING'].getint('Epochs')
        self.optimizer = Adadelta(learning_rate=self.learning_rate)
        self.history = {}
        self.history['loss'] = []
        self.history['val_loss'] = []
        continue_training = True
        for epoch in range(epochs):
            print(f'Start of epoch: {epoch}/{epochs}')
            steps_per_epoch = int(self.num_data_train / self.batch_size)
            for step in tqdm(range(steps_per_epoch)):
                x_batch, y_batch = next(self.train_gen)
                loss = self.train_step(x_batch, y_batch)
            self.history['loss'].append(loss)
            print('Loss at end of epoch: ' + str(loss))

            steps_per_epoch_val= int(self.num_data_val / self.batch_size)
            for step in tqdm(range(steps_per_epoch_val)):
                x_batch, y_batch = next(self.val_gen)
                val_loss = self.test_step(x_batch, y_batch, epoch)
            print('Validation loss at end of epoch: ' + str(val_loss))
            continue_training = self._callbacks(val_loss, epoch)
            self.history['val_loss'].append(loss)
            self._log_loss(loss, val_loss)
            if not continue_training:
                break
        return self.history

    def _make_tensors(self, window_data):
        X = np.full((self.batch_size * self.num_scat * self.max_cand, self.time, 2), -1.0)
        step = 0
        for b in range(self.batch_size):
            for s in range(self.num_scat):
                for c in range(len(window_data[b,s])):
                    X[step + c] = window_data[b,s][c]
                step += self.max_cand
        X = tf.convert_to_tensor(X)
        return X

    def _reshape(self, y):
        y = tf.split(y, self.batch_size, axis=0)
        y = tf.stack(y, axis=1)
        y = tf.split(y, self.num_scat, axis=0)
        y = tf.stack(y, axis=2)
        return y

    def custom_loss(self, y_actual, y_predicted):
        if self.class_name == 'SimpleRNN':
            return self.custom_loss_simple(y_actual, y_predicted)
        if self.class_name == 'Feedback':
            return self.custom_loss_feedback(y_actual, y_predicted)

    def custom_loss_feedback(self, y_actual, y_predicted):
        predicted, cost, prob, inputs = y_predicted
        predicted = self._reshape(predicted)
        cost = self._reshape(cost)
        prob = self._reshape(prob)
        inputs = self._reshape(inputs)

        null_tensor = tf.convert_to_tensor(np.full((self.time, 2), -1.0))
        loss_sum = tf.constant(0.0, dtype='float64')
        num = 0.0
        for i, y in enumerate(predicted):
            for b in range(self.batch_size):
                for s in range(self.num_scat):        
                    if not tf.reduce_all(tf.equal(self.X_[i,b,s], null_tensor)):
                        num += 1.0
                        compound_loss = self._compound_loss(
                            y[b,s],
                            y_actual[b,s],
                            inputs[i,b,s,self.warm_up_length:],
                            cost[i,b,s,0],
                            prob[i,b,s]
                        ) 
                        loss_sum = tf.math.add(loss_sum, compound_loss)                    
        amount = tf.constant(1/num, dtype='float64')
        loss_sum = tf.reduce_sum(loss_sum)
        loss = tf.multiply(amount, loss_sum)
        return loss

    def _compound_loss(self, y, y_actual, inputs, cost, prob):
        def safe_norm(x, epsilon=1e-12, axis=None):
            return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis) + epsilon)
        loss_sum = tf.constant(0.0, dtype='float64')
        # MSE for prediction
        mse_prediction = tf.reduce_sum(tf.losses.mean_squared_error(y, y_actual))

        # Calculate distance from ground truth, to candidate
        true_cost = safe_norm(tf.math.subtract(y_actual, inputs))

        # SE for cost
        se_cost = tf.math.square(tf.math.subtract(true_cost, cost))

        # Was the candidate correct?
        if tf.reduce_all(tf.equal(inputs, y_actual)):
            # Yes
            true_prob = tf.constant([1.0, 0.0], dtype='float64')
        else:
            # No
            true_prob = tf.constant([0.0, 1.0], dtype='float64')
        # Negative log-likelihood for positive and negative association prob.
        nll = self._negative_log_likelihood(true_prob, prob)
        loss_sum = tf.math.add(mse_prediction,loss_sum)
        loss_sum = tf.math.add(se_cost,loss_sum)
        loss_sum = tf.math.subtract(loss_sum,nll)
        return loss_sum


    def _negative_log_likelihood(self, true_prob, pred_prob):
        log_pos_prob = tf.math.log(pred_prob[0])
        log_neg_prob = tf.math.log(pred_prob[1])
        nll_first = tf.math.multiply(true_prob[0], log_pos_prob)
        nll_second = tf.math.multiply(true_prob[1], log_neg_prob)

        if tf.math.is_inf(log_pos_prob):
            if tf.reduce_all(tf.equal(tf.constant([1.0, 0.0], dtype='float64'), true_prob)):
                nll_first = tf.constant(-np.inf, dtype='float64')
            else:
                nll_first = tf.constant(0.0, dtype='float64')

        elif tf.math.is_inf(log_neg_prob):
            if tf.reduce_all(tf.equal(tf.constant([0.0, 1.0], dtype='float64'), true_prob)):
                nll_second = tf.constant(-np.inf, dtype='float64')
            else:
                nll_second = tf.constant(0.0, dtype='float64')
        nll_compound = tf.math.add(nll_first, nll_second)
        nll = tf.math.multiply(nll_compound, tf.constant(self.time-self.warm_up_length, dtype='float64'))
        return nll

    def custom_loss_simple(self, y_actual, y_predicted):
        inputs = self.X
        y_actual = y_actual[:,:,:-self.warm_up_length]
        y_predicted = self._reshape(y_predicted)
        inputs = self._reshape(inputs)

        null_tensor = tf.convert_to_tensor(np.full((self.time, 2), -1.0))
        init = 0.0
        num = 0.0
        for i, y in enumerate(y_predicted):
            for b in range(self.batch_size):
                for s in range(self.num_scat):        
                    if not tf.reduce_all(tf.equal(inputs[i][b,s], null_tensor)):
                        init += tf.losses.mean_squared_error(y[b,s], y_actual[b,s])
                        num += 1.0
        num = tf.constant(1/num, dtype='float64')
        loss_sum = tf.reduce_sum(init)
        loss = tf.multiply(num, loss_sum)
        return loss

    def _cap_ground_truth(self, Y):
        multiplex_alts = np.full(Y.shape, -1.0)
        larger_than_zero = tf.greater_equal(Y, tf.constant([0.0]))
        smaller_than_thres = tf.less_equal(Y, tf.constant([self.target_size]))
        Y = tf.where(larger_than_zero, Y, multiplex_alts)
        Y = tf.where(smaller_than_thres, Y, multiplex_alts)
        return Y

    def _callbacks(self, loss, epoch):
        if not self.patience_decay:
            if (epoch+1) % self.lr_decay_time == 0:
                old_lr = self.learning_rate
                self.learning_rate *= self.lr_decay_rate
                self.optimizer.lr.assign(self.learning_rate)
                print(f'Updated learning rate from {old_lr} to {self.learning_rate}.')


        try:
            min(self.history['val_loss'])
        except ValueError:
            if self.with_checkpoints:
                self.model.save_weights(f'output/rnn_checkpoints/model_{self.dt_string}/model_{self.dt_string}_{epoch}.h5')
            self.patience = 0
            return True

        if loss < min(self.history['val_loss']):
            if self.with_checkpoints:
                self.model.save_weights(f'output/rnn_checkpoints/model_{self.dt_string}/model_{self.dt_string}_{epoch}.h5')
            self.patience = 0
            return True
        elif self.patience > self.patience_thres:
            if not self.patience_decay:
                return False
            else:
                old_lr = self.learning_rate
                self.learning_rate *= self.lr_decay_rate
                self.optimizer.lr.assign(self.learning_rate)
                print(f'Updated learning rate from {old_lr} to {self.learning_rate}.')
                self.patience = 0
                return True
        else:
            self.patience += 1
            return True

    def _log_loss(self, loss, val_loss):
        logging.info(f'Loss at end of epoch: {loss}')
        logging.info(f'Validation loss at end of epoch: {val_loss}')

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
        metrics = [key for key in history.keys() if '_' not in key]
        plt.figure()
        for key in metrics:
            plt.plot(history[f'{key}'])
            plt.plot(history[f'val_{key}'])
            plt.title(f'model {key}')
            plt.ylabel(f'{key}')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(f'output/rnn_checkpoints/model_{self.dt_string}/{key}_{self.dt_string}.png')

    def _pick_candidates(self, x, y, candidate_picks):
        picked_candidates = [x[candidate_picks[i]][self.warm_up_length:] for i in range(len(candidate_picks))]
        labels = [y[candidate_picks[i][1:]] for i in range(len(candidate_picks))]
        return picked_candidates, labels

    def _in_frame(self, coords):
        bools = []
        for c in coords:
            if c > 0 and c < self.target_size:
                bools.append(True)
            else:
                bools.append(False)
        return all(bools)

    def _plot_selection(self, x, y, percentage):
        nrows = 2
        ncols = 4
        fig = plt.gcf()
        fig.set_size_inches(ncols * 4, nrows * 4)
        fig.patch.set_facecolor('white')
        fig.suptitle(f'Percentage of correct candidate picks: {percentage}', fontsize=16)
        frames = np.zeros((2, self.batch_size, self.target_size, self.target_size))

        def _update(x, y, frame, i):
            for b in range(self.batch_size):
                for s in range(self.num_scat):
                    x_coords = x[b*10 + s][i]
                    y_coords = y[b*10 + s][i]
                    if self._in_frame(x_coords):
                        frames[0,b,int(x_coords[0]),int(x_coords[1])] = 1.0
                    if self._in_frame(y_coords):
                        frames[1,b,int(y_coords[0]),int(y_coords[1])] = 1.0
                
                sp = plt.subplot(nrows, ncols, (b + 1))
                plt.imshow(frames[0,b])
                sp = plt.subplot(nrows, ncols, 4 + (b + 1))
                plt.imshow(frames[1,b])

        ani = FuncAnimation(
            fig,
            lambda i: _update(x, y, frames, i),
            list(range(self.time-self.warm_up_length)),
            init_func=_update(x, y, frames, 0)
        )  
        writer = PillowWriter(fps=self.time)  
        ani.save(f"output/selection_{self.dt_string}.gif", writer=writer) 

    def evaluate(self):
        x_batch, y_batch = next(self.eval_gen)
        predicted, cost, prob, x, y_actual, candidate_picks = self.eval_step(x_batch, y_batch)
        x, y = self._pick_candidates(x, y_actual, candidate_picks)
        correct = 0
        for i in range(len(x)):
            if tf.reduce_all(tf.equal(x[i], y[i])):
                correct += 1
        percentage = correct/ len(x) * 100
        self._plot_selection(x, y, percentage)
        return x, y 
        
    def compile_and_fit(self):
        history = self.training_loop()
        #os.mkdir(f'output/rnn_checkpoints/model_{self.dt_string}')
        #copyfile('config/config.ini', f'output/rnn_checkpoints/model_{self.dt_string}/config.ini')
        self.model.save_weights(f'output/rnn_checkpoints/model_{self.dt_string}/model_{self.dt_string}.h5')
        backend.clear_session()
        self.illustrate_history(history)
        return history









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
            self.build_model()
            x, y = next(self.train_gen)
            window_data = self.candidate_track_generator.make_candidate_tracks(x)
            self.model(self._make_tensors(window_data))
            self.model.load_weights(model_path)
            self.loaded_model = True
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
