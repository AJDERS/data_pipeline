import configparser
import numpy as np
from util.loader_mat import Loader
from util.sliding_window import SlidingWindow



class Organiser():

    def __init__(self, data_folder_path, config_path):
        self.loader = Loader()
        self.windows = SlidingWindow(data_folder_path, config_path)
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.config_path = config_path
        self.data_folder_path = data_folder_path
        self.batch_size = self.config['PREPROCESS_TRAIN'].getint('BatchSize')
        self.window_length = self.windows.window_size
        self.create_data()

    def create_data(self):
        data = self.windows.run()
        self.train_X = data[0]
        self.train_Y = data[1]
        self.valid_X = data[2]
        self.valid_Y = data[3]
        self.eval_X = data[4]
        self.eval_Y = data[5]

    def load_data(self):
        data = []
        for mode in ['training', 'validation', 'evaluation']:
            path_x = f'{self.data_folder_path}/tracks/{mode}/data/X.npy.gz'
            path_y = f'{self.data_folder_path}/tracks/{mode}/labels/Y.npy.gz'
            data.append(
                self.loader.decompress_load(
                    path_x
                )
            )
            data.append(
                self.loader.decompress_load(
                    path_y
                )
            )
        self.train_X = data[0]
        self.train_Y = data[1]
        self.valid_X = data[2]
        self.valid_Y = data[3]
        self.eval_X = data[4]
        self.eval_Y = data[5]

    def get_priors(self, mode):
        if mode == 'training':
            num_data = self.config['DATA'].getint('NumDataTrain')
            X = self.train_X
            Y = self.train_Y

        elif mode == 'validation':
            num_data = self.config['DATA'].getint('NumDataValid')
            X = self.valid_X
            Y = self.valid_Y

        else:
            num_data = self.config['DATA'].getint('NumDataEval')
            X = self.eval_X
            Y = self.eval_Y        
        for i in range(num_data):
            frame_specific_X = [x for x in X if x[0]==i]
            frame_specific_Y = [y for y in Y if y[0]==i]
            num_scat = len(set([x[1] for x in frame_specific_X]))
            for scat in range(num_scat):
                scat_specific_X = [x for x in frame_specific_X if x[1]==scat]
                scat_specific_Y = [y for y in frame_specific_Y if y[1]==scat]
                yield scat_specific_X, scat_specific_Y

    def _make_batch(self, mode):
        batch = []
        generator = self.get_priors(mode)
        for i in range(self.batch_size):
            X, Y = next(generator)
            batch.append(X)
        yield batch