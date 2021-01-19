import configparser
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.layers import Dropout, Dense, LSTMCell, RNN, Masking, Flatten, Concatenate, Input
from tensorflow.keras import Model

class Feedback(Model):
    def __init__(self, config_path):
        super().__init__()
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.hidden_activation = self.config['RNN'].get('HiddenActivation')
        if self.hidden_activation == 'None':
            self.hidden_activation = None
        self.lstm_activation = self.config['RNN'].get('LSTMPredictionActivation')
        if self.lstm_activation == 'None':
            self.lstm_activation = None
        self.lstm_units = self.config['RNN'].getint('LSTMUnits')
        self.fc_units = self.config['RNN'].getint('FCUnits')
        self.batch_size = self.config['PREPROCESS_TRAIN'].getint('BatchSize')
        self.warm_up_length = self.config['RNN'].getint('WarmUpLength')
        num_scatter_tmp = self.config['DATA'].get('NumScatterTrain')
        self.num_scatterer =  [int(y) for y in num_scatter_tmp.split(',')][0]
        self.droprate = self.config['RNN'].getfloat('DropOutRate')
        self.time = self.config['DATA'].getint('MovementDuration')
        self.cost_weight = self.config['RNN'].getfloat('CostWeight')
        self._make_layers()

    def _make_layers(self):
        self.mask_layer = Masking(mask_value=-1.0)
        self.lstm_cell = LSTMCell(self.lstm_units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = RNN(self.lstm_cell, return_state=True)
        self.drop_warm = Dropout(self.droprate)
        self.dense_warm = Dense(2, activation=self.lstm_activation)
        self.drop_candidate = Dropout(self.droprate)
        self.dense_candidate = Dense(self.lstm_units, activation=self.hidden_activation)
        self.flatten = Flatten()
        self.concat = Concatenate(axis=1)
        self.drop_concat = Dropout(self.droprate)
        self.dense_concat = Dense(self.fc_units, activation=self.hidden_activation)
        self.dense_cost = Dense(1, activation=None, name='cost')
        self.dense_prob = Dense(2, activation='sigmoid')
        

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)
        x = self.drop_warm(x)  
        # predictions.shape => (batch, features)
        prediction = self.dense_warm(x)
        return prediction, state, x

    def call(self, inputs_, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        inputs = self.mask_layer(inputs_)
        # Initialize the lstm state
        prediction, state, warm_up_lstm_output = self.warmup(inputs[:,self.warm_up_length:])
        warm_up_lstm_output = self.flatten(warm_up_lstm_output)
        # Insert the first prediction and outputs
        predictions.append(prediction)  
        # Run the rest of the prediction steps
        for n in range(1, self.time-self.warm_up_length):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                                      training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense_warm(x)
            # Add the prediction and the lstm output to the output
            predictions.append(prediction) 
        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])

        candidate_tensor = self.drop_candidate(inputs[:,:self.warm_up_length-1])
        candidate_tensor = self.dense_candidate(candidate_tensor)
        candidate_tensor = self.flatten(candidate_tensor)
        candidate_tensor = self.concat([warm_up_lstm_output, candidate_tensor])
        candidate_tensor = self.drop_concat(candidate_tensor)
        candidate_tensor = self.dense_concat(candidate_tensor)
        candidate_prob = self.dense_prob(candidate_tensor)
        candidate_cost = self.dense_cost(candidate_tensor)

        return predictions, candidate_cost, candidate_prob, inputs_
