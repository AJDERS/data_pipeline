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
        filter_size=10):
        self.batch = batch
        self.predicted_batch = predicted_batch 
        self.config = config
        self.scat_pos = scatterer_positions.astype(int)
        self.movement = self.config['DATA'].getboolean('Movement')
        self.tracks = self.config['DATA'].getboolean('Tracks')
        self.filter_size = filter_size

    def _find_local_maxima(self, frame):
        return maximum_filter(frame, self.filter_size)

    def _find_all_maxima(self):
        if self.movement and not self.tracks:
            maxima = np.zeros(self.predicted_batch.shape)
            for s, stack in enumerate(self.predicted_batch):
                for t in range(self.config['DATA'].getint('MovementDuration')):
                    maxima[s,:,:,t] = self._find_local_maxima(stack[:,:,t])
        else:
            maxima = np.zeros(self.predicted_batch.shape)
            for s, stack in enumerate(self.predicted_batch):
                    maxima[s,:,:] = self._find_local_maxima(stack[:,:])
        return maxima

    def _check_if_scat_pos_is_max(self, mask):
        score = 0
        correct = []
        if self.movement and not self.tracks:
            for s, stack in enumerate(self.scat_pos):
                for t in range(stack.shape[3]):
                    xs = stack[:,0,t,0]
                    ys = stack[:,1,t,0]
                    coords = zip(xs,ys)
                    for (x,y) in coords:
                        if mask[s, x, y, t] == 1.0:
                            score += 1
                            correct.append((s,x,y,t))
        else:
            for s, stack in enumerate(self.scat_pos):
                xs = stack[:,0,0,0]
                ys = stack[:,1,0,0]
                coords = zip(xs,ys)
                for (x,y) in coords:
                    if mask[s, x, y] == 1.0:
                        score += 1
                        correct.append((s,x,y))
        return correct, score

    def _add_correct_to_frame(self, correct):
        show_correct = self.predicted_batch.copy()
        if self.movement and not self.tracks:
            for (s,x,y,t) in correct:
                show_correct[s,x,y,t] += 2.0
        else:
            for (s,x,y) in correct:
                show_correct[s,x,y] += 2.0
        return show_correct

    def _calculate_metrics(self, correct, score, num_local_max, num_scat):
        recall = score / num_scat
        precision = score / num_local_max
        return recall, precision

    def num_scatter(self):
        scat_dim = self.scat_pos.shape[1]
        pos_dim = self.scat_pos.shape[2]
        num_scat = np.count_nonzero(self.scat_pos) / (scat_dim * pos_dim)
        return num_scat


    def evaluate(self):
        maxima = self._find_all_maxima()
        mask = (self.predicted_batch == maxima).astype(float)
        num_local_max = np.count_nonzero(mask)
        num_scat = self.num_scatter()
        correct, score = self._check_if_scat_pos_is_max(mask)
        recall, precision = self._calculate_metrics(correct, score, num_local_max, num_scat)
        show_correct = self._add_correct_to_frame(correct)
        return mask, show_correct, recall, precision



        

