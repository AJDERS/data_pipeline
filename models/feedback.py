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