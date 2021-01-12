import configparser
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.layers import Dropout, Dense, LSTMCell, RNN, Masking, Flatten, Concatenate
from tensorflow.keras import Model

class Feedback(Model):
    def __init__(self, config_path):
        super().__init__()
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.activation = self.config['RNN'].get('Activation')
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
        self.lstm_cell = LSTMCell(self.lstm_units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = RNN(self.lstm_cell, return_state=True)
        self.dense = Dense(2)
        self.mask_layer = Masking(mask_value=-1.0)

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)   
        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        inputs = self.mask_layer(inputs)
        self.mask = self.mask_layer.compute_mask(inputs)
        # Initialize the lstm state
        prediction, state = self.warmup(inputs[:,self.warm_up_length:]) 
        # Insert the first prediction
        predictions.append(prediction)  
        # Run the rest of the prediction steps
        for n in range(1, self.time-self.warm_up_length):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                                      training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output
            predictions.append(prediction)    
        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions
