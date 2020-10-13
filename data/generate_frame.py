"""
This module contains a class called ``FrameGenerator`` which generates tensors
with point scatterers placed in them. All specifications are set in ``config.ini``.
"""

import random as r
import numpy as np
import configparser

class FrameGenerator:
    """
    **This class creates training data in the form of 3D/4D tensors.**

    The training data consists of data and labels, the labels are sparse
    tensors while the data are tensors where the sparse tensors which are 
    convolved with a given point spread function.
    """
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.target_size = self.config['DATA'].getint('TargetSize')
        self.num_scatter = self.config['DATA'].getint('NumScatter')
        self.movement = self.config['DATA'].getboolean('Movement')
        self.velocity = self.config['DATA'].getint('MovementVelocity')
        self.duration = self.config['DATA'].getint('MovementDuration')
        self.angle = self.config['DATA'].getint('MovementAngle') * (np.pi / 180.0)
        self.scat_pos = []

    def _make_frame(self) -> np.ndarray:
        """
        **This generates the tensors in which the scatters are placed.**

        If ``Movement = True`` in ``config.ini``
        the tensors are of dimension ``(TargetSize, TargetSize, Duration)``
        if ``Movement = False`` they are of dimension
        ``(TargetSize, TargetSize)``.

        :returns: A np.ndarray filled with zeros.
        :rtype: ``np.ndarray``.
        """
        if self.movement:
            shape = (
                self.target_size,
                self.target_size,
                self.duration
            )
        else:
            shape = (self.target_size, self.target_size)
        frame = np.zeros(shape)
        return frame

    def _next_pos(self, previous_x: int, previous_y: int) -> tuple:
        """
        **Simple helper function calculating next position in time.**

        It is based on the parameters ``MovementVelocity`` and 
        ``MovementAngle`` set in ``config.ini``.

        :param previous_x: The previous x-coordinate of the scatterer.
        :param previous_y: The previous y-coordinate of the scatterer.
        :returns: x,y - which are the new coordinates of the scatterer.
        :rtype: ``tuple``
        """
        x = int(previous_x + np.cos(self.angle) * self.velocity)
        y = int(previous_y + np.sin(self.angle) * self.velocity)
        return x, y


    def _place_scatterers(self, frame: np.ndarray) -> np.ndarray:
        """
        **Places scatterers in the tensors.**

        If ``Movement == True`` it also defines the paths of scatterers based
        on the parameters ``MovementVelocity`` and ``MovementDuration``
        ``MovementAngle`` set in ``config.ini``.

        :param frame: The frame in which the scatterers are placed.
        :returns frame: The updated frame with scatterers.
        :rtype: ``np.ndarray``.

        .. warning:: If a scatterer leaves the frame its coordinates will be
            set to ``np.nan`` from that timestep and beyond. 
        """
        if self.movement:
            for _ in range(self.num_scatter):
                scat_pos_timeseries = []
                for t in range(self.duration):
                    
                    # If first time step, place randomly, else make movement.
                    if t == 0:
                        x = r.randint(0, self.target_size)
                        y = r.randint(0, self.target_size)
                    else:
                        previous_x = scat_pos_timeseries[t-1][0]
                        previous_y = scat_pos_timeseries[t-1][1]
                        
                        # Check if scatter moved out of frame in previous
                        # time step.
                        if type(previous_x) == int and type(previous_y) == int:
                            x, y = self._next_pos(previous_x, previous_y)
                        else:
                            x, y = np.nan, np.nan

                    # Check if scatter moved out of frame.
                    if x < 0 or x > self.target_size:
                        x = np.nan
                    if y < 0 or y > self.target_size:
                        y = np.nan

                    if all([type(x) == int, type(y) == int]):
                        frame[x][y][t] = 1.0

                    scat_pos_timeseries.append((x, y, t))
                self.scat_pos.append(scat_pos_timeseries)
            return frame
        else:
            for _ in range(self.num_scatter):
                x = r.randint(0, self.target_size)
                y = r.randint(0, self.target_size)
                self.scat_pos.append((x, y))
                frame[x][y] = 1.0
            return frame
            
        