import os
from util import prepare_for_runs
from train_rnn import Compiler

def main():
    prepare_for_runs.main()
    configs = [f for f in os.listdir('config/active_configs') if os.path.isfile(os.path.join('config/active_configs', f))]
    for config in configs:
        if config == '__init__.py':
            continue
        print(config)
        C = Compiler('storage', os.path.join('config/active_configs', config))
        C.compile_and_fit()
        C.evaluate()

if __name__ == "__main__":
    main()