"""
This module contains the class ``DataGenerator`` which yields batches of
pairs of data and labels. The ``DataGenerator.flow`` method replaces the
``ImageDataGenerator.flow`` method from Tensorflow, and allows for tensors of
arbitrary rank.
"""

import numpy as np
from typing import Generator

class DataGenerator:
    """
    Instantiate this class and use the ``.flow()`` method in the same way
    one would use the ``ImageDataGenerator.flow()`` method from Tensorflow.

    :param X: An array containing your data.
    :type X: ``np.ndarray``.
    :param Y: An array containing your labels.
    :type Y: ``np.ndarray``.
    :param mode: Specifying if the generator is for training, validation or
        evaluation data.
    :type mode: ``str``.
    :param config: The parsed configuration dictionary.
    :type config: ``dict``.

    :Example:
    >>> import configparser
    >>> from util.generator import DataGenerator
    >>> from tensorflow.keras.models import Model
    >>> model = Model(...) # Define model here
    >>> X = np.array([1,2,3,4])
    >>> Y = np.array([2,4,6,8])
    >>> config = configparser.ConfigParser()
    >>> config.read('your/config/path')
    >>> generator = DataGenerator(X, Y, 'training', config)
    >>> training_gen = generator.flow()
    >>> model.fit(training_gen)

    """
    def __init__(self, X, Y, mode, config):
        self.config = config
        self.stacks = self.config['DATA'].getboolean('Stacks')
        self.tracks = self.config['DATA'].getboolean('Tracks')
        self.X = X
        self.Y = Y
        self.mode = mode
        self._checks()
        self._get_config()

    def _checks(self):
        msg = "Mode must be 'training', 'validation', 'evaluation'."
        assert self.mode in ['training', 'validation', 'evaluation'], msg
        assert type(self.X) == np.ndarray, 'X must be np.ndarray.'
        assert type(self.Y) == np.ndarray, 'Y must be np.ndarray.'
        if self.stacks:
            assert self.X.shape == self.Y.shape, 'X and Y must have same shape.'
        if self.tracks:
            assert self.X.shape[0] == self.Y.shape[0], 'X and Y must have same shape.'

    def _get_config(self):
        if self.mode == 'training':
            self.preprocess = self.config['PREPROCESS_TRAIN']
            self.batch_size = self.preprocess.getint('BatchSize')
            self.shuffle = self.preprocess.getboolean('Shuffle')
            self.noise = self.preprocess.getint('NoiseDB')
            self.rotation = self.preprocess.getint('RotationRange')
            self.out_of_plane = self.preprocess.get('OutOfPlaneScat')
            if self.out_of_plane == 'None':
                duration = self.config['DATA'].getint('MovementDuration')
                self.out_of_plane = duration * self.batch_size
            else:
                self.out_of_plane = int(self.out_of_plane)

        elif self.mode == 'validation':
            self.preprocess = self.config['PREPROCESS_VALID']
            self.batch_size = self.preprocess.getint('BatchSize')
            self.shuffle = self.preprocess.getboolean('Shuffle')
            self.noise = self.preprocess.getint('NoiseDB')
            self.rotation = self.preprocess.getint('RotationRange')
            self.out_of_plane = self.preprocess.get('OutOfPlaneScat')
            if self.out_of_plane == 'None':
                duration = self.config['DATA'].getint('MovementDuration')
                self.out_of_plane = duration * self.batch_size
            else:
                self.out_of_plane = int(self.out_of_plane)

        else:
            self.preprocess = self.config['PREPROCESS_EVAL']
            self.batch_size = self.preprocess.getint('BatchSize')
            self.shuffle = self.preprocess.getboolean('Shuffle')
            self.noise = self.preprocess.getint('NoiseDB')
            self.rotation = self.preprocess.getint('RotationRange')
            self.out_of_plane = self.preprocess.get('OutOfPlaneScat')
            if self.out_of_plane == 'None':
                duration = self.config['DATA'].getint('MovementDuration')
                self.out_of_plane = duration * self.batch_size
            else:
                self.out_of_plane = int(self.out_of_plane)

    def flow(self) -> Generator[np.ndarray, None, None]:
        """
        **Creates a Generator which yields batches of data and labels.**

        The data and labels are augmented in memory, based on the parameters
        of ``config.ini``, i.e. they are shuffled, noise is added, rotated,
        and scatterers are removed. Note that the generator has no 
        ``send``-type, and hence are meant to be closed to the user.

        :returns: A generator which yields batches of data and labels.
        :rtype: Generator[np.ndarray, None, None]
        """
        index = list(range(self.X.shape[0]))
        if self.shuffle:
            np.random.shuffle(index)

        idx = 0
        while True:
            # Preallocation
            batch_xs = np.zeros((self.batch_size, *self.X.shape[1:]))
            batch_ys = np.zeros((self.batch_size, *self.Y.shape[1:]))
            for batch_i in range(self.batch_size):

                # Break if no more data.
                if idx == self.X.shape[0]:
                    break
                
                # Get data and label
                current_data = self.X[index[idx]]
                current_label = self.Y[index[idx]]
                
                # Add noise
                if self.noise:
                    current_data = self._add_noise(self.noise, current_data)
                    current_label = self._add_noise(self.noise, current_label)
                
                # Rotate
                if self.rotation:
                    current_data = self._rotate(current_data)
                    current_label = self._rotate(current_label)

                # Remove scatterer
                if self.out_of_plane:
                    current_data = self._remove_scatterer_from_frame(
                        current_data
                    )
                    current_label = self._remove_scatterer_from_frame(
                        current_label
                    )                    
                
                idx += 1
                batch_xs[batch_i] = current_data
                batch_ys[batch_i] = current_label
            yield batch_xs, batch_ys

    def _add_noise(self, noise, tensor):
        return tensor

    def _remove_scatterer_from_frame(self, tensor):
        return tensor

    def _rotate(self, tensor):
        return tensor
    