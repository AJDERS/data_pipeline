import os
import configparser
import numpy as np
from .loader_mat import Loader


class SlidingWindow:

    def __init__(self, data_folder_path, config_path):
        self.prior_frames = 3
        self.future_frames = 3
        self.window_size = self.prior_frames + self.future_frames
        self.loader = Loader()
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.config_path = config_path
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

    def  _add_candidates(self,
        data,
        index,
        time,
        current_window,
        threshold,
        candidate_index):
        # data, (batch, scatterer, coords, time, channel)
        # candidates, (scatterer, coords, channel)
        candidates = data[index, :, :, time + 1, :]
        new_windows = []
        # current_window, (candidates, coord, time, channel)
        current_scatterer = current_window[candidate_index, :, time]
        distances = []
        for c in range(self.scatterer):
            candidate = candidates[c]
            dist = np.linalg.norm(current_scatterer - candidate)
            # candidates[c], (coords, channel)
            distances.append((dist, c))
            if dist < threshold:
                window_ = current_window.copy()
                window_[candidate_index, :, time + 1, :] = candidate
                new_windows.append(window_)
        if not new_windows:
            # If no candidate within range pick nearest.
            candidate_w_min_dist = sorted(distances, key=lambda x: x[0])[0][1]
            window_ = current_window.copy()
            candidate = candidates[candidate_w_min_dist]
            window_[:, :, time + 1, :] = candidate
            new_windows.append(window_)
        new_windows = np.concatenate(new_windows,0)
        # new_windows, (candidate, coords, time, channel)
        return new_windows

    def _initial_window(
        self,
        data,
        index,
        scatterer_index,
        t,
        threshold):

        sliding_window = np.zeros(
            (1, self.coords, self.window_size, 1)
        )
        # Add dummy variables
        sliding_window[:, :, 0:self.prior_frames, :] = np.nan
        # Add real values
        sliding_window[:, :,self.prior_frames,:] = data[
            index,
            scatterer_index,
            :, 
            t, 
            :
        ]
        # Add candidates
        sliding_window = self._add_candidates(
            data,
            index, 
            self.prior_frames,
            sliding_window,
            threshold,
            0
        )
        for frame in range(1, self.future_frames-1):
            for candidate_index in range(sliding_window.shape[0]):
                sliding_window = self._add_candidates(
                    data,
                    index, 
                    self.prior_frames + frame,
                    sliding_window,
                    threshold,
                    candidate_index
                )
        return sliding_window

    def slinding_windows(self, data, threshold):
        num_sliding_windows = self.time-self.window_size+1
        sliding_windows = []
        # `(batch, scatterer, sliding_window_index, array(candidate, coord, time, channel))`
        # `(batch, scatterer, sliding_window_index)` are used to refer back to correct
        # track. `candidate` is used to enumerate the different candidates.
        for index in range(self.batch_size):
            for scatterer_index in range(self.scatterer):
                for t in range(0, num_sliding_windows):
                    if t == 0:
                        sliding_window = self._initial_window(
                            data,
                            index,
                            scatterer_index,
                            t,
                            threshold
                        )
                        sliding_windows.append(
                            (
                                index,
                                scatterer_index,
                                t,
                                sliding_window
                            )
                        )
                    else:
                        # previous_windows, (candidates, coords, time, channel)
                        previous_windows = sliding_windows[-1][-1]
                        # sliding_window, (candidates, coords, time, channel)
                        sliding_window = np.zeros(
                            (
                                previous_windows.shape[0],
                                self.coords,
                                self.window_size,
                                1
                            )
                        )
                        for candidate_index in range(previous_windows.shape[0]):
                            # Add last entries in from previous sliding window 
                            # as first entries in new sliding window
                            sliding_window[
                                candidate_index,
                                :,
                                0:(self.window_size - 1),
                                :
                            ] = previous_windows[candidate_index,:,1:,:]

                            # Add candidates.
                            sliding_window = self._add_candidates(
                                data,
                                index, 
                                self.prior_frames + t, # this takes the last time entry
                                sliding_window,
                                threshold,
                                0
                            )
                            sliding_windows.append(
                                (
                                    index,
                                    scatterer_index,
                                    t,
                                    sliding_window
                                )
                            )
                return sliding_windows