import os
import gzip
import scipy.io
import numpy as np
from PIL import Image
from tqdm import tqdm
from preprocess import bin_ndarray

class Loader():

    def __init__(self):
        pass

    def load_mat(self, path: str) -> dict:
        mat = scipy.io.loadmat(path)
        assert 'low_res' in mat.keys()
        return mat

    def get_data(self, mat: dict) -> np.ndarray:
        data = mat['low_res']['das'][0][0]
        assert data.shape == (64, 64, 11)
        return np.sum(np.real(data), axis=2)

    def get_label(self, mat: dict) -> np.ndarray:
        label_gauss = mat['conf_map']['gauss'][0][0]
        label_count = mat['conf_map']['count'][0][0]
        label_amp = mat['conf_map']['amp'][0][0]
        assert label_gauss.shape == (128, 128)
        return [label_gauss, label_count, label_amp]

    def compress_and_save(
        self,
        array: np.ndarray,
        data_type: str,
        name: str,
        type_of_data: str
        ) -> None:

        parquet = gzip.GzipFile(
            f'storage/{type_of_data}/{data_type}/{name}.npy.gz',
            'w'
        )
        np.save(file=parquet, arr=array)
        parquet.close()

    def decompress_load(self, path):
        gzipped_file_data = gzip.GzipFile(path, 'r')
        array = np.load(gzipped_file_data)
        return array

    def load_array_folder(
        self,
        source_path: str,
        type_of_data: str,
        size_ratio: float = 1.0
        ) -> np.ndarray:

        assert type_of_data in ['data', 'labels']
        folder = os.path.join(source_path)
        file_names = os.listdir(folder)

        loaded_arrays = []
        print(f'Loading {type_of_data} from storage... \n')
        for file_name in tqdm(file_names):
            fi = os.path.join(source_path, file_name)
            array = self.decompress_load(fi)
            # Downscale label array, to fit data shape
            if not size_ratio == 1.0 and type_of_data == 'labels':
                array = bin_ndarray(
                    array,
                    list(map(lambda x: int(x / size_ratio), array.shape))
                )
            loaded_arrays.append(array)
        out_array = np.array(loaded_arrays)
        out_array = np.expand_dims(out_array, -1)
        return out_array

    def load_mat_folder(self, source_path: str, type_of_data: str, name: str) -> None:
        assert type_of_data in ['training', 'evaluation', 'validation']
        file_path_list = os.listdir(source_path)
        for i, file_name in enumerate(tqdm(file_path_list)):
            path = os.path.join(source_path, file_name)
            if path.endswith('.mat'):
                mat = self.load_mat(path)
                X = self.get_data(mat)
                Y_gauss, Y_count, Y_amp = self.get_label(mat)

                self.compress_and_save(
                    X,
                    'data',
                    f'{name}_X_{str(i).zfill(5)}',
                    type_of_data
                )
                self.compress_and_save(
                    Y_gauss,
                    'labels',
                    f'{name}_gauss_{str(i).zfill(5)}',
                    type_of_data
                )
#                self.compress_and_save(
#                    Y_count,
#                    'labels',
#                    f'{name}_count_{str(i).zfill(5)}',
#                    type_of_data
#                    )
#                self.compress_and_save(
#                    Y_amp,
#                    'labels',
#                    f'{name}_amp_{str(i).zfill(5)}',
#                    type_of_data
#                )

