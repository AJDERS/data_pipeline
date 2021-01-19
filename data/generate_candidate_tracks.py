import os
import gzip
import configparser
import numpy as np
import tensorflow as tf
from util.loader_mat import Loader
from util import make_storage_folder as storage


class CandidateTrackGenerator:

    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.config_path = config_path
        self.threshold = self.config['RNN'].getint('CandidateThreshold')
        self.target_size = self.config['DATA'].getint('TargetSize')
        self.max_cand = self.config['RNN'].getint('MaximalCandidate')
        self.cap_values = self.config['RNN'].getboolean('CapValues')
        #self.batch_size = self.config['PREPROCESS_TRAIN'].getint('BatchSize')

    def _set_mode_parameters(self, tracks):
        self.batch_size = tracks.shape[0]
        self.scatterer = tracks.shape[1]
        self.coords = tracks.shape[2]
        self.time = tracks.shape[3]

    def make_candidate_tracks(self, data):
        # data: (batch_index, num_scat, coords, time, channel)
        self._set_mode_parameters(data)

        candidate_tracks = np.empty(
            (self.batch_size, self.scatterer),
            dtype=object
        )
        for index, dataset in enumerate(data):
            candidate_tracks = self._make_candidate_tracks(index, candidate_tracks, dataset)
        
        candidate_tracks = self._cap_normalize_values(candidate_tracks)
        return candidate_tracks

    def _cap_normalize_values(self, tracks):
        multiplex_alts = np.full((self.time,self.coords), -1.0)
        for b in range(self.batch_size):
            for s in range(self.scatterer):
                candidates = tracks[b,s]
                if self.cap_values:
                    for i, candidate in enumerate(candidates):
                        larger_than_zero = tf.greater_equal(candidate, tf.constant([0.0]))
                        smaller_than_thres = tf.less_equal(candidate, tf.constant([self.target_size]))
                        candidates[i] = tf.where(larger_than_zero, candidates[i], multiplex_alts)
                        candidates[i] = tf.where(smaller_than_thres, candidates[i], multiplex_alts)
                tracks[b,s] = candidates[:self.max_cand]
        return tracks

    def _make_candidate_tracks(self, index, candidate_tracks, dataset):
        # dataset: (num_scat, coords, time, channel)
        for scat_index in range(dataset.shape[0]):
            initial_coordinates = np.zeros((dataset.shape[2], 2))
            initial_coordinates[0,:] = dataset[scat_index,:,0,0]
            # initial_coordinates: (time, coords)

            current_coordinates = [initial_coordinates]
            # current_coordinates = list[(time, coords)]

            for time_index in range(0, dataset.shape[2]-1):
                future_coordinates = dataset[:,:,time_index + 1,0]
                # future_coordinates: (num_scat, coords)

                # Find suitable coordinates
                suitable_coordinate_container = []
                for current_coordinate in current_coordinates:
                    distances = []
                    suitable_coords = []
                    for future_coordinate_index in range(dataset.shape[0]):
                        boolean = self._within_threshold_dist(
                            current_coordinate,
                            time_index,
                            future_coordinates[future_coordinate_index]
                        )
                        if boolean:
                            suitable_coords.append(future_coordinates[future_coordinate_index])
                    if len(suitable_coords) == 0:
                        current_coordinate[time_index + 1, :] = np.array([-1, -1])
                    else:
                        suitable_coordinate_container.append(suitable_coords)

                # Update current coordinates list
                new_current_coordinates = []
                if len(suitable_coordinate_container) != 0:
                    for k, suitable_coords in enumerate(suitable_coordinate_container):
                        for suitable_coord in suitable_coords:
                            current_coordinates[k][time_index + 1,:] = suitable_coord
                            new_current_coordinates.append(np.copy(current_coordinates[k]))
                    current_coordinates = new_current_coordinates
            candidate_tracks[index, scat_index] = current_coordinates
        return candidate_tracks

    def _within_threshold_dist(self, xs, time, y):
        x = xs[time]
        step = 1
        threshold = self.threshold

        # If x is [-1,-1], i.e. out of frame, we go one step backwards,
        # and double the search threshold, to potentially catch occlusions.
        while np.array_equal(x,np.array([-1.,-1.])):
            try:
                x = xs[time-step]
            except IndexError:
                return False
            threshold *= 2
            step += 1
        dist = np.linalg.norm(x-y)
        return dist < threshold

