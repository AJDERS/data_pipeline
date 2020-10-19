"""
This module contains a class called ``FrameGenerator`` which generates tensors
with point scatterers placed in them. All specifications are set in ``config.ini``.
"""

import configparser
import multiprocessing
import random as r
import numpy as np
from tqdm import tqdm
from util import loader_mat
from scipy.signal import convolve2d
from joblib import Parallel, delayed
from util import make_storage_folder as storage

class FrameGenerator:
    """
    **This class creates training data in the form of 3D/4D tensors.**

    The training data consists of data and labels, the labels are sparse
    tensors while the data are tensors where the sparse tensors which are 
    convolved with a given point spread function.
    """


    def __init__(self, config_path, data_folder_path):
        self.data_folder_path = data_folder_path
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.num_data_train = self.config['DATA'].getint('NumDataTrain')
        self.num_data_valid = self.config['DATA'].getint('NumDataValid')
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
        :type previous_x: ``int``
        :param previous_y: The previous y-coordinate of the scatterer.
        :type previous_y: ``int``
        :returns: x,y - which are the new coordinates of the scatterer.
        :rtype: ``tuple``
        """
        x = int(previous_x + np.cos(self.angle) * self.velocity)
        y = int(previous_y + np.sin(self.angle) * self.velocity)
        return x, y


    def _place_scatterers(
        self,
        frame: np.ndarray,
        gaussian_map_data: np.ndarray,
        gaussian_map_label: np.ndarray
    ) -> np.ndarray:
        """
        **Places scatterers in the tensors.**

        If ``Movement == True`` it also defines the paths of scatterers based
        on the parameters ``MovementVelocity`` and ``MovementDuration``
        ``MovementAngle`` set in ``config.ini``.

        :param frame: The frame in which the scatterers are placed.
        :type frame: ``np.ndarray``
        :param gaussian_map_data: The gaussian map used for convolution of data.
        :type gaussian_map_data: ``np.ndarray``
        :param gaussian_map_label: The gaussian map used for convolution of labels.
        :type gaussian_map_label: ``np.ndarray``
        :returns: The updated frame with scatterers.
        :rtype: ``np.ndarray``.

        .. warning:: If a scatterer leaves the frame its coordinates will be
            set to ``np.nan`` from that timestep and beyond. 
        """
        finished_frame = frame
        finished_label = frame
        if self.movement:
            for k in range(self.num_scatter):
                scat_pos_timeseries = []
                for t in range(self.duration):
                    temp_frame = np.zeros(frame[:,:,t].shape)
                    temp_label = np.zeros(frame[:,:,t].shape)
                    # If first time step, place randomly, else make movement.
                    if t == 0:
                        x = r.randint(0, self.target_size-1)
                        y = r.randint(0, self.target_size-1)
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
                    if x < 0 or x >= self.target_size:
                        x = np.nan
                    if y < 0 or y >= self.target_size:
                        y = np.nan

                    # Place scatterer
                    if all([type(x) == int, type(y) == int]):
                        temp_frame[x,y] = 1.0
                        temp_label[x,y] = 1.0

                    # Convolve with gaussian map
                    temp_frame = convolve2d(
                        temp_frame,
                        gaussian_map_data,
                        'same'
                    )
                    temp_label = convolve2d(
                        temp_label,
                        gaussian_map_label,
                        'same'
                    )

                    # Normalize and transfor max to finished frame
                    finished_frame[:,:,t] = self._normalize(
                        np.maximum(
                            finished_frame[:,:,t],
                            temp_frame
                        )
                    )
                    finished_label[:,:,t] = self._normalize(
                        np.maximum(
                            finished_label[:,:,t],
                            temp_label
                        )
                    )
                    scat_pos_timeseries.append((x, y, t))
                self.scat_pos.append(scat_pos_timeseries)
            return finished_frame, finished_label
        else:
            for _ in range(self.num_scatter):
                temp_frame = frame
                temp_label = frame
                x = r.randint(0, self.target_size)
                y = r.randint(0, self.target_size)
                self.scat_pos.append((x, y))

                # Place scatterer
                temp_frame[x][y] = 1.0
                temp_label[x][y] = 1.0

                # Convolve with gaussian map
                temp_frame = convolve2d(
                    temp_frame,
                    gaussian_map_data,
                    'same'
                )
                temp_label = convolve2d(
                    temp_label,
                    gaussian_map_label,
                    'same'
                )

                # Normalize and transfor max to finished frame
                finished_frame = self._normalize(
                    np.maximum(
                        finished_frame,
                        temp_frame
                    )
                )
                finished_label = self._normalize(
                    np.maximum(
                        finished_label,
                        temp_label
                    )
                )
            return finished_frame, finished_label


    def _gaussian_map(self, sigma: float, mu: float) -> np.ndarray:
        """
        **Generates a gaussian map.**

        This function creates a 2D gaussian map stored in a matrix.

        :param sigma: The standard deviation of the gaussian map.
        :type sigma: ``float``
        :param mu: The mean of the gaussian map.
        :type mu: ``float``
        :returns: The gaussian map.
        :rtype: ``np.ndarray``. 
        """
        x_coord, y_coord = np.meshgrid(
            np.linspace(-1, 1, self.target_size-1),
            np.linspace(-1, 1, self.target_size-1)
        )
        distance = np.sqrt((x_coord * x_coord) + (y_coord * y_coord))
        gaussian_map = np.exp(-( (distance-mu)**2 / ( 2.0 * sigma**2 ) ) )
        return gaussian_map


    def _normalize(self, frame: np.ndarray) -> np.ndarray:
        """
        **Normalizes the frame**
        
        :param frame: Frame which is to be normalized.
        :type frame: ``np.ndarray``.
        :returns: Normalized frame.
        :rtype: ``np.ndarray`.
        """
        frame -= min(map(np.min, frame))
        frame /= max(map(np.max, frame))
        return frame

 
    def generate_single_frame(
        self,
        gaussian_map_data: np.ndarray,
        gaussian_map_label: np.ndarray,
        index: int,
        mode: str
        ) -> None:
        """
        **This function generates a single data, label pair and saves them.**

        The frame is made and filled with scatterers, from which the label is
        generated and saved.

        :param gaussian_map_data: The gaussian map used for convolution of data.
        :type gaussian_map_data: ``np.ndarray``
        :param gaussian_map_label: The gaussian map used for convolution of labels.
        :type gaussian_map_label: ``np.ndarray``
        :param index: The execution index, see 
            ``generate_frame.FrameGenerator.run``.
        :type index: ``int``.
        :param mode: A string specifying wether the frame is for training,
            validation, evaluation.
        :type mode: ``str``.
        """
        frame = self._make_frame()
        output, label = self._place_scatterers(
            frame,
            gaussian_map_data,
            gaussian_map_label
        )
        loader = loader_mat.Loader()
        loader.compress_and_save(
            array=output,
            data_type='data',
            name=str(index).zfill(5),
            type_of_data=mode,
            container_dir=self.container_dir,
        )
        loader.compress_and_save(
            array=label,
            data_type='labels',
            name=str(index).zfill(5),
            type_of_data=mode,
            container_dir=self.container_dir,
        )


    def run(self):
        """
        **Creates data, label pairs and saves them to ``data_folder_path``.**

        All parameters are set in ``config.ini``.
        First it tries to create the ``data_folder_path`` directory, and the
        required subdirectories: 
        ``(training/validation/evaluation)/(data/labels)``, after which the 
        pairs are created using 
        ``generate_frame.FrameGenerator.generate_single_frame``.
        """
        self.container_dir = storage.make_storage_folder(
            self.data_folder_path
        )
        gaussian_map_data = self._gaussian_map(
            self.config['DATA'].getfloat('GaussianSigma'),
            self.config['DATA'].getfloat('GaussianMu')
        )
        gaussian_map_label = self._gaussian_map(
            self.config['DATA'].getfloat('GaussianSigma')/2.0,
            self.config['DATA'].getfloat('GaussianMu')
        )
        num_cores = multiprocessing.cpu_count()
        print('Making training data.')
        Parallel(n_jobs=num_cores)(
            delayed(self.generate_single_frame)
                (
                    gaussian_map_data,
                    gaussian_map_label,
                    i,
                    'training'
                ) for i in tqdm(range(self.num_data_train))
        )

        print('Making validation data.')
        Parallel(n_jobs=num_cores)(
            delayed(self.generate_single_frame)
                (
                    gaussian_map_data,
                    gaussian_map_label,
                    i,
                    'validation'
                ) for i in tqdm(range(self.num_data_valid))
        )
