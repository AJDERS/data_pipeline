import os
import configparser
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from .loader_mat import Loader

class SlidingWindow:

    def __init__(self, data_folder_path, config_path):
        self.loader = Loader()
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.config_path = config_path
        self.prior = self.config['RNN'].getint('Prior')
        self.future = self.config['RNN'].getint('Future')
        self.window_size = self.prior + self.future
        self.data_folder_path = data_folder_path
        self.train_path = os.path.join(self.data_folder_path, 'training')
        self.valid_path = os.path.join(self.data_folder_path, 'validation')
        self.eval_path = os.path.join(self.data_folder_path, 'evaluation')
        self._load()
        self.batch_size = self.train_tracks.shape[0]
        self.scatterer = self.train_tracks.shape[1]
        self.coords = self.train_tracks.shape[2]
        self.time = self.train_tracks.shape[3]

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

    def _slinding_windows_labels(self, data):
        num_sliding_windows = self.time-self.window_size+1
        self.sliding_windows_label = []
        for index in tqdm(range(self.batch_size)):
            for scatterer_index in range(self.scatterer):
                for time in range(0, num_sliding_windows):
                    self.index = index
                    self.scatterer_index = scatterer_index
                    window = data[
                            index,
                            scatterer_index,
                            :,
                            time:time+self.window_size
                        ]
                    self._append_data_label(time, window)
        self._remove_warmup_label()

    def _slinding_windows(self, data, threshold):
        num_sliding_windows = self.time-self.window_size
        self.sliding_windows = []
        for index in tqdm(range(self.batch_size)):
            for scatterer_index in range(self.scatterer):
                for time in range(0, num_sliding_windows):
                    self.index = index
                    self.scatterer_index = scatterer_index
                    if time == 0:
                        window = data[
                            index,
                            scatterer_index,
                            :,
                            time:time+self.window_size
                        ]
                        self._append_data(time, (0, window))
                        window = window[:, 1:]
                        new_windows = self._add_candidate(
                            index,
                            time,
                            window,
                            data,
                            threshold
                        )
                        self._append_data(time+1, (0, new_windows))
                    else:
                        #previous_windows = new_windows
                        previous_windows = []
                        for win in self.sliding_windows:
                            if (win[0]==index) and (win[1]==scatterer_index) and (win[2]==(time-self.prior)):
                                previous_windows.append(win[3])

                        new_windows = []
                        for previous_window in previous_windows:
                            previous_candidate_index = previous_window[1]
                            window = previous_window[2][:, 1:]
                            new_windows_for_candidate = self._add_candidate(
                                index,
                                time,
                                window,
                                data,
                                threshold
                            )
                            new_windows.append(
                                (previous_candidate_index, new_windows_for_candidate)
                            )
                        #new_windows = [item for sublist in new_windows for item in sublist]
                        for new_window in new_windows:
                            self._append_data(time+1, new_window)
        self._remove_warmup()

    def _append_data_label(self, time, window):
        self.sliding_windows_label.append(
            (
                self.index,
                self.scatterer_index,
                time-self.prior,
                window
            )
        )

    def _organize_windows(self):
        num_sliding_windows = self.time-self.window_size-self.prior
        self.prior_frames = []
        self.future_frames = []
        self.prior_frames_label = []
        self.future_frames_label = []
        xs = self.sliding_windows
        ys = self.sliding_windows_label
        X = []
        Y = []
        for index in tqdm(range(self.batch_size)):
            x = []
            y = []
            for scatterer_index in range(self.scatterer):
                for time in range(0, num_sliding_windows):
                    for tup in xs:
                        # Take all sliding windows with current index,
                        # scatterer index and time
                        if (
                            (tup[0]==index)
                            and (tup[1]==scatterer_index)
                            and (tup[2]==time)
                        ):
                            x.append(tup[3][2])
                            # Take corresponding label
                            # To avoid look-ups in the list of labels we use the below index
                            y_index = index*self.batch_size+scatterer_index*self.scatterer+time
                            y.append(ys[y_index][3])
            X.append(np.array(x))
            Y.append(np.array(y))
        X, Y = self._make_input(X, Y)
        return X, Y


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

    def _append_data(self, time, window):
        """
        For a given dataset (index), scatterer (scatterer_index) and time
        (time), a list of movement candidates are found, i.e. which scatterers
        are within a certain distance threshold of the current scatterer.
        For each candidate (indexed by candidate) this procedure is repeated.
        For the next time step the `candidate` index is set to 
        `previous_candidate`.  

        self.sliding_windows contains all of this information, i.e. is a list
        of tuples which contain the following data:
        (index, scatterer_index, time, (previous_candidate, candidate, window))

        index: The index of the dataset in the batch.
        scatterer_index: The index of the scatterer in the dataset.
        time: The time index of the specific scatterer in the dataset.
        previous_candidate: the index among the previous sliding windows 
            movement candidates.
        candidate: the index among the current sliding windows movement candidates.

        So to keep track of a stack of sliding windows, you pick a `index` and
        `scatterer` and at each time you know the sliding windows previous sliding
        window and the list of sliding windows it gives rise to.
        """
        if type(window[1]) == list:
            for i, win in enumerate(window[1]):
                self.sliding_windows.append(
                    (
                        self.index,
                        self.scatterer_index,
                        time-self.prior,
                        (window[0], i, win)
                    )
                )
        else:
            self.sliding_windows.append(
                (
                    self.index,
                    self.scatterer_index,
                    time-self.prior,
                    (window[0], 0, window[1])
                )
            )


    def _remove_warmup(self):
        self.sliding_windows = [x for x in self.sliding_windows if x[2] >= 0]

    def _remove_warmup_label(self):
        self.sliding_windows_label = [x for x in self.sliding_windows_label if x[2] >= 0]

    def _add_candidate(
        self,
        index,
        time,
        window,
        data,
        threshold
    ):
        current_scatterer = window[:, -1]
        # If the last step in the window is np.nan, go further back
        # and increase threshold to account for occlution.
        if np.isnan(np.sum(current_scatterer)):
            current_scatterer = window[:, -2]
            threshold *= 2

        candidates = data[index, :, :, time+self.window_size]
        distances = []
        new_windows = []
        suitable_candidate = False
        for c in range(self.scatterer):
            candidate = candidates[c]
            dist = np.linalg.norm(current_scatterer - candidate)
            distances.append((dist, c))
            if dist < threshold:
                suitable_candidate = True
                window_ = window.copy()
                candidate = np.reshape(candidate, (2,1,1))
                window_ = np.append(window_, candidate, axis=1)
                new_windows.append(window_)
        if not suitable_candidate:
            window_ = window.copy()
            empty = np.empty((2,1,1))
            empty[:] = np.nan
            window_ = np.append(window_, empty, axis=1)
            new_windows.append(window_)
        return new_windows        