import os
from util.clean_storage_folder import clean_storage
from data.generate_frame import FrameGenerator as FG
from config.config_generator import ConfigGenerator

def main():
    # Clean storage
    clean_storage('storage')
    fg = FG('config/default_config.ini', 'storage')
    fg.run()
    
    # Make configs
    cfg = ConfigGenerator('config/default_config.ini', 'config/hyperparameter_list.txt')
    cfg.make_configs()

    # Make output folders
    os.mkdir('output')
    os.mkdir('output/rnn_checkpoints')

if __name__ == "__main__":
    main()