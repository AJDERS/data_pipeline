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
        self.filter_size = filter_size

    def _find_local_maxima(self, frame):
        return maximum_filter(frame, self.filter_size)

    def evaluate(self):
        if self.movement:
            maxima = np.zeros(self.predicted_batch.shape)
            for s, stack in enumerate(self.predicted_batch):
                for t in range(self.config['DATA'].getint('MovementDuration')):
                    maxima[s,:,:,t] = self._find_local_maxima(stack[:,:,t])

            msk = (self.predicted_batch == maxima).astype(float)
            score = 0
            correct = []
            for s, stack in enumerate(self.scat_pos):
                for t in range(stack.shape[3]):
                    xs = stack[:,0,t,0]
                    ys = stack[:,1,t,0]
                    coords = zip(xs,ys)
                    for (x,y) in coords:
                        if msk[s, x, y, t] == 1.0:
                            score += 1
                            correct.append((s,x,y,t))
            num_scat = self.config['DATA'].getint('NumScatter')
            maximal_score = num_scat * self.scat_pos.shape[0]
            accuracy = score / maximal_score
            show_correct = self.predicted_batch.copy()
            for (s,x,y,t) in correct:
                show_correct[s,x,y,t] += 2.0 
            return msk, show_correct, accuracy

        else:
            maxima = np.zeros(self.predicted_batch.shape)
            for s, stack in enumerate(self.predicted_batch):
                    maxima[s,:,:] = self._find_local_maxima(stack[:,:])

            msk = (self.predicted_batch == maxima).astype(float)
            score = 0
            correct = []
            for s, stack in enumerate(self.scat_pos):
                xs = stack[:,0,0,0]
                ys = stack[:,1,0,0]
                coords = zip(xs,ys)
                for (x,y) in coords:
                    if msk[s, x, y] == 1.0:
                        score += 1
                    correct.append((s,x,y))
            num_scat = self.config['DATA'].getint('NumScatter')
            maximal_score = num_scat * self.scat_pos.shape[0]
            accuracy = score / maximal_score
            show_correct = self.predicted_batch.copy()
            for (s,x,y) in correct:
                show_correct[s,x,y] += 2.0
            return msk, show_correct, accuracy


        

