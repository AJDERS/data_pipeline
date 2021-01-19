import numpy as np
import tensorflow as tf

class TanhNormalizer:

    def __init__(self):
        pass

    def tanh_normalization(self, unnormalized_data):
        self.m = np.mean(unnormalized_data, axis=0)
        self.std = np.std(unnormalized_data, axis=0)
        data = 0.5 * (np.tanh(0.01 * ((unnormalized_data - self.m) / self.std)) + 1)
        return tf.constant(data)

