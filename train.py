import argparse

from utils.config import *
from agents import *

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--config',
        default=None,
        help='The path of configuration file in yaml format')
    args = arg_parser.parse_args()
    #args.config = 'configs/train_stgan_mat.yaml'
    config = process_config(args.config)

    agent = globals()['{}'.format(config.network)](config)
    agent.run()


if __name__ == '__main__':
    main()
