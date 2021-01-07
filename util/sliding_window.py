import os
import gzip
import configparser
import numpy as np
import tensorflow as tf
from .loader_mat import Loader
from util import make_storage_folder as storage


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

    def _set_mode_parameters(self, tracks):
        self.batch_size = tracks.shape[0]
        self.scatterer = tracks.shape[1]
        self.coords = tracks.shape[2]
        self.time = tracks.shape[3]

    def _slinding_windows_labels(self, data):
        self._set_mode_parameters(data)
        num_sliding_windows = self.time-self.window_size+1
        self.sliding_windows_label = []
        for index in range(self.batch_size):
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
        self._set_mode_parameters(data)
        num_sliding_windows = self.time-self.window_size
        self.sliding_windows = []
        for index in range(self.batch_size):
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
        for index in range(self.batch_size):
            x = []
            y = []
            for scatterer_index in range(self.scatterer):
                for time in range(0, num_sliding_windows):
                    for tup in xs:
                        added = False
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
                            for tupl in ys:
                                if added:
                                    pass
                                else:
                                    if (
                                        (tup[0]==index)
                                        and (tup[1]==scatterer_index)
                                        and (tup[2]==time)
                                    ):
                                        y.append(tupl[3])
                                        added = True
                            
            X.append(np.array(x))
            Y.append(np.array(y))
        #X = self._pad_organise(X)
        #Y = self._pad_organise(Y)
        X = self._cap_values(X)
        Y = self._cap_values(Y)
        return X, Y

    def _determine_candidate_cap(self, array):
        l = [x.shape[0] for x in array]
        m = max(l)
        avg = sum(l) / len(l)
        if m > 10*avg:
            return int(((avg*0.25) // self.window_size) * self.window_size)
        else:
            return None

    def _pad_organise(self, array):
        candidate_cap = self._determine_candidate_cap(array)
        array = tf.keras.preprocessing.sequence.pad_sequences(
            array,
            padding='post',
            value=-1
        )
        array = self._cap_values(array)
        array = np.stack(array, axis=0)
        if candidate_cap:
            array = array[:,:candidate_cap]
        array = np.reshape(array, (-1, 2, self.window_size))
        array = np.transpose(array, (0, 2, 1))
        return array

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
                window_ = tf.identity(window)
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

    def _cap_values(self, arr):
        #array = tf.keras.preprocessing.sequence.pad_sequences(
        #    array,
        #   padding='post',
        #    value=-1
        #)
        for array in arr:
            try:
                array = array[3][2]
            except:
                array = array[3]
            cap = self.config['DATA'].getint('TargetSize')
            tmp = array[(array < 0) | (array > (cap-1))]
            tmp = -1
        return arr

    def run(self, tracks, labels=False):
        threshold = self.config['RNN'].getint('CandidateThreshold')
        if labels:
            self._slinding_windows_labels(tracks)
            sliding_windows_label = self.sliding_windows_label
            sliding_windows_label = self._cap_values(sliding_windows_label)
            return sliding_windows_label
        else:
            self._slinding_windows(tracks, threshold)
            sliding_windows = self.sliding_windows
            sliding_windows = self._cap_values(sliding_windows)
            return sliding_windows

