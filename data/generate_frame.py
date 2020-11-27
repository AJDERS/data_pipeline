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
        self.num_data_eval = self.config['DATA'].getint('NumDataEval')
        self.target_size = self.config['DATA'].getint('TargetSize')
        num_scatter_train_tmp = self.config['DATA'].get('NumScatterTrain')
        self.num_scatter_train =  [int(y) for y in num_scatter_train_tmp.split(',')]
        num_scatter_eval_tmp = self.config['DATA'].get('NumScatterEval')
        self.num_scatter_eval =  [int(y) for y in num_scatter_eval_tmp.split(',')]
        self.movement = self.config['DATA'].getboolean('Movement')
        velocity_tmp = self.config['DATA'].get('MovementVelocityRange')
        self.velocity = [int(y) for y in velocity_tmp.split(',')]
        angle_tmp = self.config['DATA'].get('MovementAngleRange')
        self.angle = [int(y) * (np.pi / 180.0) for y in angle_tmp.split(',')]
        self.duration = self.config['DATA'].getint('MovementDuration')
        self.removal_prob = self.config['DATA'].getfloat('RemovalProbability')
        self.stacks = self.config['DATA'].getboolean('Stacks')
        self.tracks = self.config['DATA'].getboolean('Tracks')
        self.tracks_with_gaussian = self.config['DATA'].getboolean('')
        self.noise = self.config['DATA'].getfloat('Noise')
        msg = 'One and only one output format must be set.'
        assert self.stacks != self.tracks, msg
        r.seed(self.config['PIPELINE'].getint('Seed'))


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


    def _next_pos(
        self,
        previous_x: int,
        previous_y: int,
        angle: float,
        velocity: int) -> tuple:
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
        x = int(previous_x + np.cos(angle) * velocity)
        y = int(previous_y + np.sin(angle) * velocity)
        return x, y

    def _in_frame(self, position: tuple) -> bool:
        return all(
            [
                (position[0] >= 0 and position[0] < self.target_size),
                (position[1] >= 0 and position[1] < self.target_size)
            ]
        )

    def _place_scatterers(
        self,
        frame: np.ndarray,
        gaussian_map_data: np.ndarray,
        gaussian_map_label: np.ndarray,
        mode: str
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
        finished_frame = frame.copy()
        scat_pos = frame.copy()
        finished_label = frame.copy()
        if self.movement:
            # Define matrix to contain scatterer positions.
            if not mode=='evaluation':
                tmp_pos = np.empty((self.num_scatter_train[1], 2, self.duration))
                tmp_pos[:] = np.nan
                num_scatter = r.randint(self.num_scatter_train[0], self.num_scatter_train[1])
            else:
                tmp_pos = np.empty((self.num_scatter_eval[1], 2, self.duration))
                tmp_pos[:] = np.nan
                num_scatter = r.randint(self.num_scatter_eval[0], self.num_scatter_eval[1])

            for k in range(num_scatter):                
                velocity = r.randint(self.velocity[0], self.velocity[1])
                angle = r.uniform(self.angle[0], self.angle[1])
                for t in range(self.duration):
                    temp_frame = np.zeros(frame[:,:,t].shape)
                    temp_label = np.zeros(frame[:,:,t].shape)

                    # If first time step, place randomly, else make movement.
                    if t == 0:
                        x = r.randint(0, self.target_size-1)
                        y = r.randint(0, self.target_size-1)
                    else:
                        previous_x = tmp_pos[k,0,t-1]
                        previous_y = tmp_pos[k,1,t-1]
                        x, y = self._next_pos(
                            previous_x,
                            previous_y,
                            angle,
                            velocity
                        )

                    # Place scatterer if in frame
                    if self._in_frame((x,y)):
                        scat_pos[x,y,t] = 1.0
                        temp_label[x,y] = 1.0
                        if not (r.random() <= self.removal_prob):
                            temp_frame[x,y] = 1.0



                    # Convolve with gaussian map
                    temp_frame = convolve2d(
                        temp_frame,
                        gaussian_map_data,
                        'same'
                    )
                    # Normalize and transfor max to finished frame
                    finished_frame[:,:,t] = self._normalize(
                        np.maximum(
                            finished_frame[:,:,t],
                            temp_frame
                        )
                    )

                    temp_label = convolve2d(
                        temp_label,
                        gaussian_map_label,
                        'same'
                    )
                    finished_label[:,:,t] = self._normalize(
                        np.maximum(
                            finished_label[:,:,t],
                            temp_label
                        )
                    )
                    tmp_pos[k,0,t] = x
                    tmp_pos[k,1,t] = y
            
            for t in range(self.duration):
                # Add noise to frame
                noise = np.random.normal(0, self.noise, finished_frame.shape[:-1])
                finished_frame[:,:,t] = finished_frame[:,:,t] + noise

            if all([self.stacks, self.tracks]):
                return finished_frame, finished_label, scat_pos, tmp_pos
            elif self.stacks:
                return finished_frame, finished_label, scat_pos, tmp_pos
            else:
                return finished_frame, finished_label, scat_pos, tmp_pos
        else:
            if not mode=='evaluation':
                tmp_pos = np.zeros((self.num_scatter_train[1], 2, 1))
                num_scatter = r.randint(self.num_scatter_train[0], self.num_scatter_train[1])
            else:
                tmp_pos = np.zeros((self.num_scatter_eval[1], 2, 1))
                num_scatter = r.randint(self.num_scatter_eval[0], self.num_scatter_eval[1])
            
            for k in range(num_scatter):
                temp_frame = np.zeros(frame.shape)
                temp_label = np.zeros(frame.shape)
                # If first time step, place randomly, else make movement.
                
                x = r.randint(0, self.target_size-1)
                y = r.randint(0, self.target_size-1)
                
                # Place scatterer
                temp_frame[x,y] = 1.0
                temp_label[x,y] = 1.0
                scat_pos[x,y] = 1.0
                tmp_pos[k,0,0] = x
                tmp_pos[k,1,0] = y

                # Convolve with gaussian map
                temp_frame = convolve2d(
                    temp_frame,
                    gaussian_map_data,
                    'same'
                )
                # Normalize and transfor max to finished frame
                finished_frame = self._normalize(
                    np.maximum(
                        finished_frame,
                        temp_frame
                    )
                )
                # Add noise to frame
                noise = np.random.normal(0, self.noise, finished_frame.shape)
                finished_frame = finished_frame + noise

                temp_label = convolve2d(
                    temp_label,
                    gaussian_map_label,
                    'same'
                )
                finished_label = self._normalize(
                    np.maximum(
                        finished_label,
                        temp_label
                    )
                )
            return finished_frame, finished_label, scat_pos, tmp_pos


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
        assert (frame >= 0.0).all()
        maximum = max(map(np.max, frame))
        minimum = min(map(np.min, frame))
        if minimum < maximum:
            frame -= minimum
            frame /= maximum
        return frame

    def _make_tracks(self, label: np.ndarray) -> np.ndarray:
        """
        **Creates frames with tracks from position data.**
        
        :param label: A list of lists of tuples of positions.
        :type label: ``list``.
        :returns: Frame with tracks.
        :rtype: ``np.ndarray`.
        """
        assert self.movement, 'Movement is set to False in the config.'
        assert self.tracks, 'Tracks is set to False in the config.'
        frame = self._make_frame()
        for t in range(self.duration):
            if not t == 0:
                frames = [label[:,:,time] for time in range(1,t+1)]
                stacked_frames = np.maximum.reduce(frames)
            else:
                stacked_frames = label[:,:,0]
            frame[:,:,t] = stacked_frames
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
        output, label, scat_pos, tmp_pos = self._place_scatterers(
            frame,
            gaussian_map_data,
            gaussian_map_label,
            mode
        )
        loader = loader_mat.Loader()
        loader.compress_and_save(
            array=output,
            data_type='data',
            name='stack_'+str(index).zfill(5),
            type_of_data=mode,
            container_dir=self.container_dir,
        )
        loader.compress_and_save(
            array=scat_pos,
            data_type='scatterer_positions',
            name='pos_'+str(index).zfill(5),
            type_of_data=mode,
            container_dir=self.container_dir,
        )
        loader.compress_and_save(
            array=tmp_pos,
            data_type='tracks',
            name=str(index).zfill(5),
            type_of_data=mode,
            container_dir=self.container_dir,
        )
        if self.stacks:
            loader.compress_and_save(
                array=label,
                data_type='labels',
                name=str(index).zfill(5),
                type_of_data=mode,
                container_dir=self.container_dir,
            )
        if self.tracks:
            tracks = self._make_tracks(label)
            loader.compress_and_save(
                array=tracks,
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

        print('Making evaluation data.')
        Parallel(n_jobs=num_cores)(
            delayed(self.generate_single_frame)
                (
                    gaussian_map_data,
                    gaussian_map_label,
                    i,
                    'evaluation'
                ) for i in tqdm(range(self.num_data_eval))
        )
