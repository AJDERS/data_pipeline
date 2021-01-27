import os
import sys
import argparse
from train_rnn import Compiler

def main(args):
    parser = argparse.ArgumentParser(description="Input config path.")
    parser.add_argument("-c", '--config')
    args = parser.parse_args(args)
    config = args.config
    C = Compiler('storage', config)
    C.compile_and_fit()
    C.evaluate()

if __name__ == "__main__":
    main(sys.argv[1:])