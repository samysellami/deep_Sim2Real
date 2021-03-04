import numpy as np
import configparser
import math
import os

from deep_calibration.utils.kinematics import Kinematics
from deep_calibration import script_dir

def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args

def build_args(parser):
    parser.add_argument(
        "--config","--configs",
        help="config file name", type=str,
        metavar="CONFIG_NAME", dest="config", required=True
    )
    parser.add_argument("--algo", type=str, default=None, help="algorithm")

    return parser


def main(args, unknown_args):

    # path to the configuration file 
    path = os.path.join(script_dir,'configs', args.config)
    
    config = configparser.ConfigParser()
    config.read(path)
    print(config['ADAPT']['maxmsteps'])
	
if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
