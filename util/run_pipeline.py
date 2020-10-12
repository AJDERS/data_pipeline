import os
from train import Model
from loader_mat import Loader
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='configuration script')
    parser.add_argument('--storage_dir', '-s_dir', type=str, nargs=1,
    help='destination path for the parsed .mat file.')
    parser.add_argument('--mat_dir', '-m_dir', type=str, nargs=1,
    help='directory that contains the folder which contain the data and the labels as .mat files.')
    parser.add_argument('--config_file', '-conf', type=str, nargs=1,
    help='path to the .ini config file, e.g. ./config.yaml')

    return vars(parser.parse_args())

def main():
    opt = parse_args()
    print(opt)
    config = opt['config_file'][0]
    mat_dir = opt['mat_dir'][0]
    storage_dir = opt['storage_dir'][0]
    
    M = Model(storage_dir, config)
    # Load training data if not already done
    if not os.listdir(os.path.join(storage_dir, 'training', 'data')):
        M.loader.load_mat_folder(
            os.path.join(mat_dir, 'train', 'data'),
            'training',
            'data'
        )
    else:
        path = os.path.join(storage_dir, 'training', 'data')
        print(f'{path} already contains data, skipping load.')

    # Load validation data if not already done
    if not os.listdir(os.path.join(storage_dir, 'validation', 'data')):
        M.loader.load_mat_folder(
            os.path.join(mat_dir, 'train', 'data'),
            'validation',
            'data'
        )
    else:
        path = os.path.join(storage_dir, 'validation', 'data')
        print(f'{path} already contains data, skipping load.')

    history = M.fit_model()
    M.illustrate_history(history)
    M.print_img()

if __name__ == "__main__":
    main()