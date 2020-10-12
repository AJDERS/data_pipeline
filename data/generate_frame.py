import numpy as np
import configparser

class Frame:

    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.target_size = self.config['DATA'].getint('TargetSize')
        self.num_scatter = self.config['DATA'].getint('NumScatter')
        self.movement = self.config['DATA'].getboolean('Movement')
        self.movement_duration = self.config['DATA'].getint('MovementDuration')
        self.movement_angle = self.config['DATA'].getint('MovementAngle')

    def _make_frame(self) -> np.ndarray:
        if self.movement:
            shape = (
                self.target_size,
                self.target_size,
                3,
                self.movement_duration
            )
        else:
            shape = (self.target_size, self.target_size, 3)
        frame = np.zeros(shape)
        return frame