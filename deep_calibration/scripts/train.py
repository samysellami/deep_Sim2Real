import numpy as np
import configparser
import math
import os
import gym

from deep_calibration.utils.kinematics import Kinematics
from deep_calibration import script_dir


from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args

def build_args(parser):
    parser.add_argument(
        "--config","--configs",
        help = "config file name", type = str,
        metavar = "CONFIG_NAME", dest = "config", required = True
    )
    parser.add_argument("--algo", type = str, default=None, help = "algorithm")
    return parser


def main(args, unknown_args):

    # path to the configuration file 
    path = os.path.join(script_dir,'configs', args.config)
    
    config = configparser.ConfigParser()
    config.read(path)

    total_timesteps = config.getint('ADAPT','total_timesteps')
    env_name = config['ADAPT']['environment']


    env = gym.make(env_name)
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    # env = DummyVecEnv([lambda: env])

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=total_timesteps)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

    env.close()

	
if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
