import configparser
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from models.feedback import Feedback
from tensorflow.keras.layers import Dense, LSTM, Masking
from tensorflow.keras.models import Sequential

class SimpleRNN:
    def __init__(self, config_path, batch_size, window_length, features):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.units = self.config['RNN'].getint('Units')
        self.batch_size = batch_size
        self.window_length = window_length
        self.features = features


    def create_model(self):
        lstm_model = Sequential(
            [
                Masking(
                    mask_value=-1.0,
                    input_shape=(
                        self.window_length,
                        self.features
                    )
                ),
                LSTM(128, activation='sigmoid', return_sequences=True),
                Dense(units=2)
            ]
        )
        return lstm_model
