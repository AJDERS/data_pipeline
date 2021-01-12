import configparser
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.layers import Dropout, Dense, LSTMCell, RNN, Masking, Flatten, Concatenate
from tensorflow.keras import Model



class RNNStepPredictor(Model):
    def __init__(self, config_path):
        super().__init__()
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.activation = self.config['RNN'].get('Activation')
        self.lstm_units = self.config['RNN'].getint('LSTMUnits')
        self.fc_units = self.config['RNN'].getint('FCUnits')
        self.warm_up_length = self.config['RNN'].getint('WarmUpLength')
        self.batch_size = self.config['PREPROCESS_TRAIN'].getint('BatchSize')
        num_scatter_tmp = self.config['DATA'].get('NumScatterTrain')
        self.num_scatterer =  [int(y) for y in num_scatter_tmp.split(',')][0]
        self.droprate = self.config['RNN'].getfloat('DropOutRate')
        self.time = self.config['DATA'].getint('MovementDuration')
        self.cost_weight = self.config['RNN'].getfloat('CostWeight')
        self._make_layers()

    def _make_layers(self):
        self.lstm_cell = LSTMCell(self.lstm_units, dtype=tf.float64)
        
        self.dropout_lstm_pred = Dropout(rate=self.droprate, dtype='float64')
        
        self.dense_lstm_pred = Dense(units=2, activation=self.activation)

        self.first_dense = Dense(self.fc_units, activation=self.activation)
        self.first_drop = Dropout(self.droprate,dtype='float64')

        self.second_dense = Dense(self.fc_units, activation=self.activation)
        self.second_drop = Dropout(self.droprate,dtype='float64')

        self.after_concat_dense = Dense(self.fc_units, activation=self.activation)
        self.after_concat_drop = Dropout(self.droprate,dtype='float64')

        self.prob_dense = Dense(2, activation='softmax')
        self.asso_dense = Dense(1, activation=None)

    def _activation(self):
        if self.activation == 'tanh':
            act = tf.nn.tanh
        elif self.activation  == 'relu':
            act = tf.nn.relu
        elif self.activation  == 'relu6':
            act = tf.nn.relu6
        elif self.activation  == 'crelu':
            act = tf.nn.crelu
        elif self.activation  == 'elu':
            act = tf.nn.elu
        elif self.activation  == 'softsign':
            act = tf.nn.softsign
        elif self.activation  == 'softplus':
            act = tf.nn.softplus
        elif self.activation  == 'sigmoid':
            act = tf.sigmoid
        else:
            act = None
        self.activation = act

    def lstm_prediction(self, data, state):
        x, state = self.lstm_cell(data, states=state)
        x = self.dropout_lstm_pred(x)
        prediction = self.dense_lstm_pred(x)
        return prediction, x, state

    def _warmup(self, start_features):
        """
        For each coordinate the LSTM predicts the next step.
        This method is for warm-up i.e. the first `window_size`-length
        set of frames are passed to the LSTM, and it predicts the steps from 1
        to `window_size`+1. All steps and the final state are returned. 
        """
        # start_features.shape => (1, time, features)
        # x.shape => (1, lstm_units)
        # # predictions.shape => (1, features)
        mask = Masking(mask_value=-1.0, dtype=np.float64)
        start_features = mask(start_features)
        lstm_outputs = []
        state = self.lstm_cell.get_initial_state(
            inputs=start_features[:,0],
            batch_size=self.batch_size,
            dtype=np.float64
        )
        for t in range(self.warm_up_length-1):
            x, state = self.lstm_cell(start_features[:,t], states=state)
            lstm_outputs.append(x)
        prediction, x, state = self.lstm_prediction(start_features[:,self.warm_up_length-1], state)
        lstm_outputs.append(x)
        lstm_outputs = tf.stack(lstm_outputs)
        lstm_outputs = tf.transpose(lstm_outputs, [1, 0, 2])
        return prediction, lstm_outputs, state

    def _fully_connected(self, tensor, order):
        assert order in ['first', 'second', 'after_concat']
        if order == 'first':
            tensor = self.first_dense(tensor)
            tensor = self.first_drop(tensor)
        elif order == 'second':
            tensor = self.second_dense(tensor)
            tensor = self.second_drop(tensor)
        else:
            tensor = self.after_concat_dense(tensor)
            tensor = self.after_concat_drop(tensor)
        return tensor

    def call(self, inputs):

        # Find maximal number of candidates
        num_candidate = 0
        for b in range(self.batch_size):
            for s in range(self.num_scatterer):
                length = len(inputs[b,s])
                if length > num_candidate:
                    num_candidate = length

        # Preallocate output tensors.
        # 0, lstm-predictions from each candidate.
        # 1, association probabilities for each candidate.
        # 2, association cost for each candidate.
        # 3, candidate.
        outputs = np.zeros((4, self.batch_size, self.num_scatterer, num_candidate, self.time - self.warm_up_length, 2))


        for b in range(self.batch_size):
            for s in range(self.num_scatterer):
                candidate_list = inputs[b,s]
                for c, candidate in enumerate(candidate_list):

                    # Warm up of LSTM. Returns:
                    # lstm_prediction, the `warm_up_length`+1 coord prediction.
                    # lstm_outputs, the lstm hidden states at each time step.
                    # state, the cell and hidden states of the last time step.
                    warm_up = tf.expand_dims(candidate[0:self.warm_up_length], axis=0)
                    lstm_prediction, warm_up_lstm_outputs, state = self._warmup(warm_up)

                    # Make LSTM prediction of entire track.
                    lstm_candidate_predictions = []
                    for _ in range(self.time - self.warm_up_length):
                        x = lstm_prediction
                        lstm_prediction, _, state = self.lstm_prediction(x, state)
                        lstm_candidate_predictions.append(lstm_prediction)

                    lstm_candidate_predictions = tf.stack(lstm_candidate_predictions)
                    lstm_candidate_predictions = tf.transpose(lstm_candidate_predictions, [1, 0, 2])

                    # Predict association cost and probability for LSTM predictions.
                    lstm_prob, lstm_cost = self._predict_association_prob_cost(
                        lstm_candidate_predictions,
                        warm_up_lstm_outputs
                    )
                    outputs[0,b,s,c] = lstm_candidate_predictions
                    outputs[1, b,s,c, 0, 0] = lstm_cost
                    outputs[2, b,s,c,0] = lstm_prob
                    outputs[3, b,s,c] = candidate[self.warm_up_length:]
        return outputs
                

    def _predict_association_prob_cost(self, predictions, lstm_outputs):
        flat_lstm_outputs = Flatten()(lstm_outputs)
        tensor = self._fully_connected(predictions, 'first')
        tensor = self._fully_connected(tensor, 'second')
        flat_candidate_tensor = Flatten()(tensor)
        concat_tensor = Concatenate()([flat_lstm_outputs, flat_candidate_tensor])
        evaluation_tensor = self._fully_connected(concat_tensor, 'after_concat')
        association_prop = self.prob_dense(evaluation_tensor)
        association_cost = self.asso_dense(evaluation_tensor)
        return association_prop, association_cost

