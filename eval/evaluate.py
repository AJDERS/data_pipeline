import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter

class Evaluator():

    def __init__(
        self,
        config,
        predicted_batch,
        batch,
        scatterer_positions,
        filter_size=5):
        self.batch = batch
        self.predicted_batch = predicted_batch 
        self.config = config
        self.scat_pos = scatterer_positions.astype(int)
        self.movement = self.config['DATA'].getboolean('Movement')
        self.tracks = self.config['DATA'].getboolean('Tracks')
        self.filter_size = filter_size

    def _cutoff_low_values(self):
        low_values = self.predicted_batch < 0.5
        self.cutoff_batch = self.predicted_batch.copy()
        self.cutoff_batch[low_values] = 0.0
        return low_values

    def _find_local_maxima(self, frame):
        return maximum_filter(frame, self.filter_size)

    def _find_all_maxima(self):
        if self.movement and not self.tracks:
            maxima = np.zeros(self.cutoff_batch.shape)
            for s, stack in enumerate(self.cutoff_batch):
                for t in range(self.config['DATA'].getint('MovementDuration')):
                    maxima[s,:,:,t] = self._find_local_maxima(stack[:,:,t])
        else:
            maxima = np.zeros(self.cutoff_batch.shape)
            for s, stack in enumerate(self.cutoff_batch):
                    maxima[s,:,:] = self._find_local_maxima(stack[:,:])
        return maxima


    def _compare_mask_w_scat_pos(self, mask):
        true_p = 0
        true_n = 0
        false_n = 0
        false_p = 0
        N = self.config['DATA'].getint('TargetSize')
        if self.movement and not self.tracks:
            for stack_index in range(self.scat_pos.shape[0]):
                for time in range(self.scat_pos.shape[3]):
                    coord_max = np.transpose(
                        np.nonzero(mask[stack_index,:,:,time])
                    )
                    coord_scat = np.transpose(
                        np.nonzero(self.scat_pos[stack_index,:,:,time])
                    )
                    for scat in coord_scat:
                        true_positive = False
                        for coord in coord_max:
                            if all(coord == scat):
                                true_p += 1
                                true_positive = True
                        if any(
                            [
                                scat[0]<0,
                                scat[1]<0,
                                scat[0]>=N,
                                scat[1]>=N
                            ]
                        ):
                            if not true_positive:
                                true_p += 1
                        else:
                            if not true_positive:
                                false_n += 1
                    
                    for max_ in coord_max:
                        if max_ not in coord_scat:
                            false_p += 1
        else:
            for stack_index in range(self.scat_pos.shape[0]):
                coord_max = np.transpose(
                    np.nonzero(mask[stack_index,:,:,0])
                )
                coord_scat = np.transpose(
                    np.nonzero(self.scat_pos[stack_index,:,:,0])
                )
                for scat in coord_scat:
                    if scat in coord_max:
                        true_p += 1
                    elif any(
                        [
                            scat[0]<0,
                            scat[1]<0,
                            scat[0]>=N,
                            scat[1]>=N
                        ]
                    ):
                        true_p += 1
                    else:
                        false_n += 1
                
                for max_ in coord_max:
                    if max_ not in coord_scat:
                        false_p += 1
        return true_p, true_n, false_p, false_n

    def _calculate_metrics(self, true_p, true_n, false_p, false_n):
        recall = true_p / (true_p + false_n)
        precision = true_p / (true_p + false_p) #### NO FALSE POSITIVES?!?!?!?!?!?
        return recall, precision

    def make_mask(self):
        low_values = self._cutoff_low_values()
        maxima = self._find_all_maxima()
        mask = (self.cutoff_batch == maxima).astype(float)
        mask[low_values] = 0.0
        return mask, maxima

    def evaluate(self):
        mask, maxima = self.make_mask()
        true_p, true_n, false_p, false_n = self._compare_mask_w_scat_pos(mask)
        recall, precision = self._calculate_metrics(true_p, true_n, false_p, false_n)
        return mask, maxima, recall, precision



        

