import numpy as np
import configparser
import math
import os
import gym

from deep_calibration.utils.kinematics import Kinematics
from deep_calibration import script_dir


from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args

def build_args(parser):
    """
        Build the parser arguments
        :param parser: (parser) the parser to which the arguments are added
        :return: (parser) The modified parser
    """
    parser.add_argument(
        "--config","--configs",
        help = "config file name", type = str,
        metavar = "CONFIG_NAME", dest = "config", required = True
    )
    parser.add_argument(
        "--algo", type = str, default=None, 
        dest = "algo", help = "algorithm", required = True
    )
    return parser


def main(args, unknown_args):


    # path to the configuration file 
    path = os.path.join(script_dir,'configs', args.config)
    
    # parsing config file    
    config_file = configparser.ConfigParser()
    config_file.read(path)
    total_timesteps = config_file.getint('ADAPT','total_timesteps')
    env_name = config_file['ADAPT']['environment']
    algo = args.algo

    # instanciating the environment
    env = Monitor(gym.make(env_name))
    env = DummyVecEnv([lambda: env])

    if algo == 'PPO':
        # creating the model and training
        model = PPO(MlpPolicy, env, verbose=1)
    else:
        raise NotImplementedError('the algo specified is has not been recognized !!')

    # Create save dir
    save_dir = os.path.join(script_dir,'saved_models', args.algo)
    os.makedirs(save_dir, exist_ok=True)
    

    model.learn(total_timesteps=total_timesteps)
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    # The model will be saved under PPO.zip
    model.save(save_dir + '/' + args.algo)

    # sample an observation from the environment
    # obs = model.env.observation_space.sample()
    obs = model.env.reset()

    # Check prediction before saving
    print("pre saved", model.predict(obs, deterministic=True))

    del model # delete trained model to demonstrate loading
    loaded_model = PPO.load(save_dir + '/' + args.algo)
    # Check that the prediction is the same after loading (for the same observation)
    print("loaded", loaded_model.predict(obs, deterministic=True))

	
if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
