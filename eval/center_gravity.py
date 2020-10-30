import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pdb

class CenterGravity():

    def __init__(self):
        self.max_values = []
        pass

    #def __init__(self, config, predicted_batch, batch):
    #    self.batch = batch
    #    self.predicted_batch = predicted_batch 
    #    self.config = config
    #    self.movement = self.config['DATA'].getboolean('Movement')

    def find_max_quad(self, frame):
        max_coords = np.where(frame == np.amax(frame))
        coord_list = list(zip(max_coords[0], max_coords[1]))
        return [c for c in coord_list if frame[c] > 0.8]

    def split(self, array, nrows, ncols):
        """Split a matrix into sub-matrices."""

        r, h = array.shape
        return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

    def _find_gaussians(self, frame, coord=None, cur_point=None, up_level_point=None, quadrant_index=None):
        # Add middle to point to find.
        if cur_point:
            step = frame.shape[0] / 2
            up_level_point = cur_point.copy()
            if quadrant_index == 0:
                cur_point[0] -= step
                cur_point[1] -= step 
                print(cur_point)
            if quadrant_index == 1:
                cur_point[0] += step
                cur_point[1] -= step
                print(cur_point)
            if quadrant_index == 2:
                cur_point[0] -= step
                cur_point[1] += step
                print(cur_point)
            if quadrant_index == 3:
                cur_point[0] += step
                cur_point[1] += step
                print(cur_point)
        else:
            # If start of recursion define middle of frame.
            up_level_point = None
            self.up_level_point = up_level_point
            cur_point = [(len(frame[0])-1)/2, (len(frame[1])-1)/2]
            print(cur_point)
        split_step = (frame.shape[0] // 2)
        try:
            split_step = int(split_step)
        except:
            pass
        # Stop recursion and move to next quadrant.
        if split_step < 1:
            print(f'bottom: {cur_point}')
            self.max_values.append(cur_point)
            self.up_level_point = up_level_point
            raise StopIteration
        now = datetime.now()
        dt_string = now.strftime("%S_%f")
        plt.imshow(frame)
        plt.savefig(f'output/test_img/{dt_string}.png')
        quadrants = self.split(frame, split_step, split_step)
        for i, quadrant in enumerate(quadrants): 
            up_level_point = self.up_level_point
            coord_list = self.find_max_quad(quadrant)
            print(f'quadrant: {i}')
            if coord_list:
                cur_point = self._rec_helper(quadrant, coord_list, cur_point, up_level_point, i)
        
    def _rec_helper(self, quadrant, coord_list, cur_point, up_level_point, quadrant_index):
        from_bottom = False
        for coord in coord_list:
            try:
                self._find_gaussians(quadrant, coord, cur_point, up_level_point, quadrant_index)
            except StopIteration:
                up_level_point = self.up_level_point
                print('reached bottom')
                from_bottom = True
                continue
        if from_bottom:
            return up_level_point
        else:
            return cur_point