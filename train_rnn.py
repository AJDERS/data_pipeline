import os
import random
import logging
import importlib
import configparser
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from shutil import copyfile
from tqdm import tqdm
from tensorflow import keras
from typing import Type
from datetime import datetime
from models.feedback import Feedback
from util.safe_norm import safe_norm
from util.reshape import _reshape
from util import clean_storage_folder as cleaner
from util.loader_mat import Loader
from util.generator import DataGenerator
from util.tanh_normalization import TanhNormalizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.python.keras import backend
from tensorflow.keras.callbacks import History
from matplotlib.animation import FuncAnimation, PillowWriter
from data.generate_candidate_tracks import CandidateTrackGenerator



class Compiler():

    def __init__(self, data_folder_path, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        print(f'config_path: {config_path}')
        print(f'dir: {os.getcwd()}')
        self.config_path = config_path
        self.output_dir = ''#'/work3/ajepe/'
        self.config_name = '.'.join(self.config_path.split('/')[-1].split('.')[:-1])
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
        self.optimizer = self.config['TRAINING'].get('Optimizer')
        self.patience = 0
        self.train_X = None
        self.train_Y = None
        self.valid_X = None
        self.valid_Y = None
        self.eval_X = None
        self.eval_Y = None
        self.do_evaluation = False
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
        os.mkdir(f'{self.output_dir}output/rnn_checkpoints/{self.config_name}_model_{self.dt_string}')
        copyfile(self.config_path, f'{self.output_dir}output/rnn_checkpoints/{self.config_name}_model_{self.dt_string}/{self.config_name}.ini')
        logging.basicConfig(
                    filename=f'{self.output_dir}output/rnn_checkpoints/{self.config_name}_model_{self.dt_string}/model_{self.dt_string}.log',
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

    def train_step(self, x, y):
        # Make candidate tracks.
        # x_batch: (batch_size, num_scatterer, coords, time, channel)
        window_data = self.candidate_track_generator.make_candidate_tracks(x)
        if self.cap_values:
            y = self._cap_ground_truth(y)
        Y = tf.squeeze(tf.transpose(y, (0, 1, 3, 2, 4)))
        y_actual = Y[:,:,self.warm_up_length:]
        X = self._make_tensors(window_data)
        if self.class_name == 'TrackPicker':
            y_actual = self.get_best_track(window_data, Y)
        with tf.GradientTape() as tape:
            tape.watch(X)#tape.watch(self.X)
            logits = self.model.call(X, training=True)#self.model.call(self.X)
            loss = self.custom_loss(y_actual, logits)#self.custom_loss(y_normed, logits)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        if self.do_evaluation:
            predictions, candidate_cost, candidate_prob, inputs_ = logits
            chosen_candidates, predicted_candidates = self._pick_candidates(predictions, candidate_cost, candidate_prob, inputs_, y_actual)
            predicted = np.add(chosen_candidates, predicted_candidates)
            mean_squared_error = tf.reduce_mean(tf.losses.mean_squared_error(predicted, y_actual))
            logging.info(f'Mean square error of candidate selection at end of {epoch}: {mean_squared_error}')
            correct = 0
            for b in range(self.batch_size):
                for s in range(self.num_scat):
                    if np.array_equal(predicted[b,s], y_actual[b,s]):
                        correct += 1
            self.history_acc['val_acc'].append(correct/(self.batch_size*self.num_scat))
            self.do_evaluation = False
        return loss

    def test_step(self, x, y, epoch):
        # Make candidate tracks.
        # x_batch: (batch_size, num_scatterer, coords, time, channel)
        window_data = self.candidate_track_generator.make_candidate_tracks(x)
        if self.cap_values:
            y = self._cap_ground_truth(y)
        Y = tf.squeeze(tf.transpose(y, (0, 1, 3, 2, 4)))
        y_actual = Y[:,:,self.warm_up_length:]
        X = self._make_tensors(window_data)
        if self.class_name == 'TrackPicker':
            y_actual = self.get_best_track(window_data, Y)
        logits = self.model.call(X, training=False) #self.model.call(self.X, training=False)
        if self.do_evaluation:
            predictions, candidate_cost, candidate_prob, inputs_ = logits
            chosen_candidates, predicted_candidates = self._pick_candidates(predictions, candidate_cost, candidate_prob, inputs_, y_actual)
            predicted = np.add(chosen_candidates, predicted_candidates)
            mean_squared_error = tf.reduce_mean(tf.losses.mean_squared_error(predicted, y_actual))
            logging.info(f'Mean square error of candidate selection at end of {epoch}: {mean_squared_error}')
            correct = 0
            for b in range(self.batch_size):
                for s in range(self.num_scat):
                    if np.array_equal(predicted[b,s], y_actual[b,s]):
                        correct += 1
            self.history_acc['val_acc'].append(correct/(self.batch_size*self.num_scat))
            self.do_evaluation = False
        loss = self.custom_loss(y_actual, logits)#self.custom_loss(y_normed, logits)
        return loss

    def eval_step(self, x):
        # Make candidate tracks.
        # x_batch: (batch_size, num_scatterer, coords, time, channel)
        window_data = self.candidate_track_generator.make_candidate_tracks(x)
        X = self._make_tensors(window_data)
        predictions, candidate_cost, candidate_prob, inputs_ = self.model.call(X, training=False)
        return predictions, candidate_cost, candidate_prob, inputs_

    def training_loop(self):
        self.build_model()
        epochs = self.config['TRAINING'].getint('Epochs')
        if self.optimizer == 'Adadelta':
            self.optimizer = Adadelta(learning_rate=self.learning_rate)
        else:
            self.optimizer = Adam(learning_rate=self.learning_rate/100)
        self.history_loss = {}
        self.history_acc = {}
        self.history_loss['loss'] = []
        self.history_loss['val_loss'] = []
        self.history_acc['acc'] = []
        self.history_acc['val_acc'] = []
        continue_training = True

        # Only for testing
        #x_batch, y_batch = next(self.train_gen)
        #self.x_batch = x_batch
        #self.y_batch = y_batch
        #x_batch_v, y_batch_v = next(self.val_gen)

        for epoch in range(epochs):
            print(f'Start of epoch: {epoch}/{epochs}')

            # Training cycle
            steps_per_epoch = int(self.num_data_train / self.batch_size)
            for step in tqdm(range(steps_per_epoch)):
                if step == steps_per_epoch:
                    self.do_evaluation = True
                x_batch, y_batch = next(self.train_gen)
                loss = self.train_step(x_batch, y_batch)
            self.history_loss['loss'].append(loss.numpy())
            print('Loss at end of epoch: ' + str(loss))

            # Test cycle
            steps_per_epoch_val= int(self.num_data_val / self.batch_size)
            for step in tqdm(range(steps_per_epoch_val)):
                if step == steps_per_epoch_val:
                    self.do_evaluation = True
                x_batch, y_batch = next(self.val_gen)
                val_loss = self.test_step(x_batch, y_batch, epoch)
            print('Validation loss at end of epoch: ' + str(val_loss))
            continue_training = self._callbacks(val_loss, epoch)
            self.history_loss['val_loss'].append(val_loss.numpy())
            self._log_loss(loss, val_loss)
            if not continue_training:
                break
        return self.history_loss

    def _make_tensors(self, window_data):
        X = np.full((self.batch_size, self.time, 2 * self.num_scat * self.max_cand), -1.0)
        for b in range(self.batch_size):
            step = 0
            for s in range(self.num_scat):
                for c in range(len(window_data[b,s])):
                    X[b, :, (step + 2*c):(step + 2*(c+1))] = window_data[b,s][c]
                step += 2 * self.max_cand
        X = tf.convert_to_tensor(X)
        return X

    def get_best_track(self, x, y):
        best = np.zeros((self.batch_size, self.num_scat, self.max_cand))
        for b in range(self.batch_size):
            for s in range(self.num_scat):
                norms = []
                found = False
                stop = False
                for c in range(self.max_cand):
                    if not found:
                        norm = np.linalg.norm(tf.math.subtract(x[b,s][c], y[b,s]))
                        norms.append((c,norm))
                        if norm == 0.0:
                            best[b,s,c] = 1.0
                            found = True
                        else:
                            best[b,s,c] = 0.0
                    else:
                        pass
                if not found:
                    c = sorted(norms, key=lambda x: x[1])[0][0]
                    best[b,s,c] = 1.0
        return best

    def custom_loss(self, y_actual, y_predicted):
        if self.class_name == 'SimpleRNN':
            return self.custom_loss_simple(y_actual, y_predicted)
        if self.class_name == 'Feedback':
            return self.custom_loss_test(y_actual, y_predicted)
        if self.class_name == 'TrackPicker':
            return self.custom_loss_picker(y_actual, y_predicted)


    def custom_loss_picker(self, y_actual, y_predicted):
        
        ent = tf.losses.categorical_crossentropy(y_actual, y_predicted)
        sums = tf.reduce_sum(ent)
        return sums

    def custom_loss_test(self, y_actual, y_predicted):
        """
        For testing of predictions.
        """
        predicted, cost, prob, inputs = y_predicted
        # predicted: (batch, time, scatterer, candidate, coords)
        # inputs: (batch, time, scatterer, candidate, coords)
        # cost: (batch, scatterer, candidate, cost)
        # prob: (batch, time, scatterer, candidate, probs)
        null_tensor = tf.convert_to_tensor(np.full((self.time, 2), -1.0))
        loss_sum = tf.constant(0.0, dtype='float64')
        num = 0.0
        for b in range(self.batch_size):
            for s in range(self.num_scat):
                for c in range(self.max_cand):
                    if not tf.reduce_all(tf.equal(inputs[b,:,s,c], null_tensor)):
                        num += 1.0
                        prob_array = np.zeros((self.time-self.warm_up_length, 2))
                        true_prob = tf.equal(inputs[b,self.warm_up_length:,s,c], y_actual[b,s])
                        for t in range(self.time - self.warm_up_length):
                            if tf.reduce_all(true_prob[t]):
                                prob_array[t] = np.array([0.0, 1.0])
                            else:
                                prob_array[t] = np.array([1.0, 0.0])
                        prob_tensor = tf.convert_to_tensor(prob_array)
                        nll = tf.reduce_sum(tf.losses.binary_crossentropy(prob_tensor, prob[b,:,s,c]))
                        #logging.info(f'NLL: {nll}')
                        loss_sum = tf.math.add(loss_sum, nll)

                        true_cost = safe_norm(tf.math.subtract(y_actual[b,s], inputs[b,self.warm_up_length:,s,c]))
                        # SE for cost
                        se_cost = tf.math.square(tf.math.subtract(true_cost, cost[b,s,c,0]))
                        #logging.info(f'SE: {se_cost}')
                        loss_sum = tf.math.add(se_cost,loss_sum)     

                        mse_prediction = tf.reduce_sum(
                            tf.losses.mean_squared_error(predicted[b,:,s,c], y_actual[b,s])
                        )
                        #logging.info(f'MSE: {mse_prediction}')
                        loss_sum = tf.math.add(loss_sum, mse_prediction)
        #amount = tf.constant(1/num, dtype='float64')
        #loss = tf.multiply(amount, loss_sum)
        return loss_sum


    def _cap_ground_truth(self, Y):
        multiplex_alts = np.full(Y.shape, -1.0)
        larger_than_zero = tf.greater_equal(Y, tf.constant([0.0]))
        smaller_than_thres = tf.less_equal(Y, tf.constant([self.target_size]))
        Y = tf.where(larger_than_zero, Y, multiplex_alts)
        Y = tf.where(smaller_than_thres, Y, multiplex_alts)
        return Y

    def _callbacks(self, loss, epoch):
        if epoch+1 % 50 == 0:
            self.evaluate(epoch=epoch)
        if not self.patience_decay:
            if not self.lr_decay_time == 0:
                if (epoch+1) % self.lr_decay_time == 0:
                    old_lr = self.learning_rate
                    self.learning_rate *= self.lr_decay_rate
                    self.optimizer.lr.assign(self.learning_rate)
                    print(f'Updated learning rate from {old_lr} to {self.learning_rate}.')
        try:
            min(self.history_loss['val_loss'])
        except ValueError:
            if self.with_checkpoints:
                self.model.save_weights(f'{self.output_dir}output/rnn_checkpoints/{self.config_name}_model_{self.dt_string}/model_{self.dt_string}_{epoch}.h5')
            self.patience = 0
            return True

        if loss < min(self.history_loss['val_loss']):
            if self.with_checkpoints:
                self.model.save_weights(f'{self.output_dir}output/rnn_checkpoints/{self.config_name}_model_{self.dt_string}/model_{self.dt_string}_{epoch}.h5')
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
            plt.savefig(f'{self.output_dir}output/rnn_checkpoints/{self.config_name}_model_{self.dt_string}/{key}_{self.dt_string}.png')

    def _in_frame(self, coords):
        bools = []
        for c in coords:
            if c > 0 and c < self.target_size:
                bools.append(True)
            else:
                bools.append(False)
        return all(bools)

    def _plot_selection(self, chosen_candidates, predicted_candidates, y_actual, epoch=None):
        nrows = 2
        ncols = 4
        fig, ax = plt.subplots(nrows, ncols)
        fig.set_size_inches(ncols * 4, nrows * 4)
        fig.patch.set_facecolor('white')
        custom_xlim = (0, self.target_size)
        custom_ylim = (0, self.target_size)
        not_chosen = np.full((self.time-self.warm_up_length, 2), -1)
        self.X_plot_cand = {}
        self.Y_plot_cand = {}
        self.X_plot_pred = {}
        self.Y_plot_pred = {}
        for b in range(self.batch_size):
            self.X_plot_cand[b] = []
            self.Y_plot_cand[b] = []
            self.X_plot_pred[b] = []
            self.Y_plot_pred[b] = []
        def _update(chosen_candidates, predicted_candidates, y_actual, i):
            print(f'Making frame {i}/{self.time-self.warm_up_length}')
            for b in range(self.batch_size):
                for s in range(self.num_scat):
                    if not np.array_equal(chosen_candidates[b,s], not_chosen):
                        x_coords = chosen_candidates[b,s,i]
                        y_coords = y_actual[b,s,i]
                        if self._in_frame(x_coords):
                            self.X_plot_cand[b].append((int(x_coords[0]),int(x_coords[1])))
                        if self._in_frame(y_coords):
                            self.Y_plot_cand[b].append((int(y_coords[0]),int(y_coords[1])))
                    else:
                        x_coords = predicted_candidates[b,s,i]
                        y_coords = y_actual[b,s,i]
                        if self._in_frame(x_coords):
                            self.X_plot_pred[b].append((int(x_coords[0]),int(x_coords[1])))
                        if self._in_frame(y_coords):
                            self.Y_plot_pred[b].append((int(y_coords[0]),int(y_coords[1])))
                

                sp = plt.subplot(nrows, ncols, (b + 1))
                sp.set_xlim(custom_xlim[0], custom_xlim[1])
                sp.set_ylim(custom_ylim[0], custom_ylim[1])
                # Add X points
                for point in self.X_plot_cand[b]:
                    plt.scatter(point[0], point[1], color='green')
                for point in self.X_plot_pred[b]:
                    plt.scatter(point[0], point[1], color='red')  

                sp = plt.subplot(nrows, ncols, 4 + (b + 1))
                sp.set_xlim(custom_xlim[0], custom_xlim[1])
                sp.set_ylim(custom_ylim[0], custom_ylim[1])
                for point in self.Y_plot_cand[b]:
                    plt.scatter(point[0], point[1], color='green')
                for point in self.Y_plot_pred[b]:
                    plt.scatter(point[0], point[1], color='red')  

        ani = FuncAnimation(
            fig,
            lambda i: _update(chosen_candidates, predicted_candidates, y_actual, i),
            list(range(self.time-self.warm_up_length)),
            init_func=_update(chosen_candidates, predicted_candidates, y_actual, 0)
        )  
        writer = PillowWriter(fps=self.time)
        ani.save(f"{self.output_dir}output/rnn_checkpoints/{self.config_name}_model_{self.dt_string}/selection_{self.dt_string}_{epoch}.gif", writer=writer) 

    def _pick_candidates(self, predictions, candidate_cost, candidate_prob, inputs_, y_actual, epoch=None):
        chosen_candidates = np.full(
            (self.batch_size, self.num_scat, self.time-self.warm_up_length, 2), -1
        )
        predicted_candidates = np.zeros(
            (self.batch_size, self.num_scat, self.time-self.warm_up_length, 2)

        )
        null_tensor = tf.convert_to_tensor(np.full((self.time, 2), -1.0))
        all_true = np.array([[0.0, 1.0]]*(self.time-self.warm_up_length))
        print('Selecting candidates...')
        logging.info('E'*100)
        logging.info(f'Epoch: {epoch}')
        for b in tqdm(range(self.batch_size)):
            logging.info(f'Batch index: {b}')
            logging.info('B'*100 + '\n')
            for s in range(self.num_scat):
                logging.info(f'Scatter index: {s}')
                logging.info('S'*100 + '\n')
                probs = []
                costs = []
                for c in range(self.max_cand):
                    if not tf.reduce_all(tf.equal(inputs_[b,:,s,c], null_tensor)):
                        logging.info(f'Candidate index: {c}')
                        logging.info('C'*100 + '\n')
                        logging.info(f'Actual track: \n {y_actual[b,s]} \n')
                        logging.info(f'Candidate input: \n {inputs_[b,:,s,c]} \n')
                        logging.info(f'Prediction: \n {predictions[b,:,s,c]} \n')
                        logging.info(f'Cost prediction: \n {candidate_cost[b,s,c]} \n')
                        costs.append((c, candidate_cost[b,s,c,0]))
                        logging.info(f'Association probability: \n {candidate_prob[b,:,s,c]} \n')
                        probs.append((c, candidate_prob[b,:,s,c]))
                cost_choice = sorted(costs, key=lambda x: x[1])
                chosen = False
                for i in range((self.time-self.warm_up_length)*2, 5, -1):
                    if not chosen:
                        for choice in cost_choice:
                            if not chosen:
                                if np.sum(np.isclose(probs[choice[0]][1], all_true)) == i:
                                    chosen_candidates[b,s] = inputs_[b,self.warm_up_length:,s,choice[0]]
                                    chosen = True
                                else:
                                    pass
                            else:
                                pass
                    else:
                        break
                if not chosen:
                    predicted_candidates[b,s] = tf.dtypes.cast(predictions[b,:,s,cost_choice[0][0],:], 'int32')
        return chosen_candidates, predicted_candidates


    def evaluate(self, epoch=None):
        x, y = next(self.eval_gen)
        if self.cap_values:
            y = self._cap_ground_truth(y)
        Y = tf.squeeze(tf.transpose(y, (0, 1, 3, 2, 4)))
        y_actual = Y[:,:,self.warm_up_length:]
        predictions, candidate_cost, candidate_prob, inputs_ = self.eval_step(x)
        chosen_candidates, predicted_candidates = self._pick_candidates(predictions, candidate_cost, candidate_prob, inputs_, y_actual)
        self._plot_selection(chosen_candidates, predicted_candidates, y_actual, epoch)
        predicted = np.add(chosen_candidates, predicted_candidates)
        mean_squared_error = tf.reduce_mean(tf.losses.mean_squared_error(predicted, y_actual))
        logging.info(f'Mean square error of candidate selection: {mean_squared_error}')
        return mean_squared_error
        
    def compile_and_fit(self):
        history = self.training_loop()
        df = pd.DataFrame.from_dict(history)
        df.to_csv(f'{self.output_dir}output/rnn_checkpoints/{self.config_name}_model_{self.dt_string}/history_{self.dt_string}.csv')
        df = pd.DataFrame.from_dict(self.history_acc)
        df.to_csv(f'{self.output_dir}output/rnn_checkpoints/{self.config_name}_model_{self.dt_string}/history_{self.dt_string}.csv')
        self.model.save_weights(f'{self.output_dir}output/rnn_checkpoints/{self.config_name}_model_{self.dt_string}/model_{self.dt_string}.h5')
        backend.clear_session()
        self.illustrate_history(history)
        return history