import configparser
from itertools import product

class ConfigGenerator():

    def __init__(self, default_config_path, hyperparameter_list_path):
        self.config = configparser.ConfigParser()
        self.config.optionxform=str
        self.config.read(default_config_path)
        self.para = hyperparameter_list_path

    def load_parameters(self):
        file = open(self.para, 'r')
        option_list = []
        for line in file:        
            split_line = line.split(':')
            options = split_line[-1].split(',')
            for i, option in enumerate(options):
                if '\n' in option:
                    options[i] = option[:-1]
            option_list.append(options)
        file.close()
        return option_list
            

    def make_configs(self):
        option_list = self.load_parameters()
        products = product(*option_list)
        #print(f'Making {len(list(products))} different configs...')
        for prod in products:
            optimizer, lr, lrtime, lrrate, epoch, warm, units, act = prod
            self.config['TRAINING']['Optimizer'] = optimizer
            self.config['TRAINING']['LearningRate'] = lr
            self.config['TRAINING']['LearningRateDecayTime'] = lrtime
            self.config['TRAINING']['LearningRateDecayRate'] = lrrate
            self.config['TRAINING']['Epochs'] = epoch
            self.config['RNN']['WarmUpLength'] = warm 
            self.config['RNN']['LSTMUnits'] = units
            self.config['RNN']['HiddenActivation'] = act
            config_name = f'config/active_configs/{optimizer}_{lr}_{lrtime}_{lrrate}_{epoch}_{warm}_{units}_{act}_config.ini'
            with open(config_name, 'w') as configfile:
                self.config.write(configfile)
        



