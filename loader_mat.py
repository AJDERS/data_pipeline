import os
import gzip
import scipy.io
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_mat(path: str) -> dict:
    mat = scipy.io.loadmat(path)
    assert 'low_res' in mat.keys()
    return mat

def get_data(mat: dict) -> np.ndarray:
    data = mat['low_res']['das'][0][0]
    assert data.shape == (64, 64, 11)
    return np.sum(data, axis=2)

def get_label(mat: dict) -> np.ndarray:
    label_gauss = mat['conf_map']['gauss'][0][0]
    label_count = mat['conf_map']['count'][0][0]
    label_amp = mat['conf_map']['amp'][0][0]
    assert label_gauss.shape == (128, 128)
    return [label_gauss, label_count, label_amp]

def compress_and_save(
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

def load_folder(source_path: str, type_of_data: str, name: str) -> None:
    assert type_of_data in ['training', 'evaluation', 'validation']
    file_path_list = os.listdir(source_path)
    X = np.zeros((len(file_path_list), 64, 64))
    Y_gauss = np.zeros((len(file_path_list), 128, 128))
    Y_count = np.zeros((len(file_path_list), 128, 128))
    Y_amp = np.zeros((len(file_path_list), 128, 128))
    for i, file_name in enumerate(tqdm(file_path_list)):
        path = os.path.join(source_path, file_name)
        if path.endswith('.mat'):
            mat = load_mat(path)
            X[i] = get_data(mat)
            Y_gauss[i], Y_count[i], Y_amp[i] = get_label(mat)
    compress_and_save(X, 'data', f'{name}_X', type_of_data)
    compress_and_save(Y_gauss, 'labels', f'{name}_gauss', type_of_data)
    compress_and_save(Y_count, 'labels', f'{name}_count', type_of_data)
    compress_and_save(Y_amp, 'labels', f'{name}_amp', type_of_data)
    