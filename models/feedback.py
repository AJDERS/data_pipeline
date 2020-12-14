import configparser
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LSTMCell, RNN, Input

class Feedback(Model):
    def __init__(self, config_path):
        super().__init__()
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.prior = self.config['RNN'].getint('Prior')
        self.future = self.config['RNN'].getint('Future')
        self.window_size = self.prior + self.future
        self.out_steps = self.config['RNN'].getint('OutputSteps')
        self.units = self.config['RNN'].getint('Units')
        self.lstm_cell = LSTMCell(self.units)

        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = RNN(self.lstm_cell, return_state=True)
        self.dense = Dense(self.window_size)

    def _get_row_splits(self, inputs):
        inner_dims = [0] + [x.shape[0]*16 for x in inputs]
        return inner_dims

    def _make_input(self, inputs, labels):
        """
        Seems complicated, but is to ensure, that we get output shape:
        [batch_size, window_size, 2, channel, (candidate)]
        where (candidate) is a ragged dimension.
        This is to better fit with LSTM/RNN input conventions.
        The `RaggedTensor` API is lackluster to say the least,
        so this is a hacky version to get the correct dimensions, while
        using the `RaggedTensor` functionalities when training.

        """
        x0 = []
        y0 = []
        for x in range(len(inputs)):
            for i in range(inputs[0].shape[2]):
                for j in range(inputs[0].shape[1]):
                    for k in range(inputs[0].shape[-1]):
                        x0.append(inputs[x][:,j,i,k])
                        y0.append(labels[x][:,j,i,k])
        
        x1 = tf.ragged.constant(x0)
        x2 = tf.RaggedTensor.from_uniform_row_length(x1, 1)
        x3 = tf.RaggedTensor.from_uniform_row_length(x2, 2)
        x4 = tf.RaggedTensor.from_uniform_row_length(x3, self.window_size)
        y1 = tf.ragged.constant(y0)
        y2 = tf.RaggedTensor.from_uniform_row_length(y1, 1)
        y3 = tf.RaggedTensor.from_uniform_row_length(y2, 2)
        y4 = tf.RaggedTensor.from_uniform_row_length(y3, self.window_size)
        return x4, y4

    def warmup(self, inputs):
      # inputs, (batch, time, features)
      # x, (batch, lstm_units)
      inputs.shape
      x, *state = self.lstm_rnn(inputs)

      # predictions, (batch, features)
      prediction = self.dense(x)
      return prediction, state 

    def call(self, inputs, training=None):
        predictions = []
        prediction, state = self.warmup(inputs)
        predictions.append(prediction)
        for n in tqdm(range(1, self.out_steps)):
          x = prediction
          x, state = self.lstm_cell(x, states=state,
                                    training=training)
          prediction = self.dense(x)
          predictions.append(prediction)

        # predictions, (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions,(batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions