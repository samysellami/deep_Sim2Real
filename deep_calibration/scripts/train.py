import numpy as np
import configparser
import math
import os
import gym

from deep_calibration.utils.kinematics import Kinematics
from deep_calibration import script_dir
from deep_calibration.scripts.callbacks import SaveOnBestTrainingRewardCallback
from deep_calibration.scripts.callbacks import PlottingCallback
from deep_calibration.scripts.callbacks import ProgressBarManager, ProgressBarCallback
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env



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
    
    # parsing config file and the args parser    
    # path to the configuration file 
    path = os.path.join(script_dir,'configs', args.config)
    config_file = configparser.ConfigParser()
    config_file.read(path)
    total_timesteps = config_file.getint('ADAPT','total_timesteps')
    env_name = config_file['ADAPT']['environment']
    algo = args.algo

    # Create saving directory
    log_dir = os.path.join(script_dir,'saved_models', args.algo)
    os.makedirs(log_dir, exist_ok = True)

    # Create and wrap the environment
    env = make_vec_env(env_name, n_envs = 1, monitor_dir = log_dir)
    # equivalent to:
    # env = Monitor(gym.make(env_name), log_dir)
    # env = DummyVecEnv([lambda: env])    

    if algo == 'PPO':
        # creating the model and training
        model = PPO(MlpPolicy, env, verbose = 1)
    else:
        raise NotImplementedError('the algo specified is has not been recognized !!')


    # Create Callbacks and train the model
    auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq = 20, log_dir = log_dir, verbose = 0)
    plotting_callback = PlottingCallback(log_dir = log_dir)
    
    # model.learn(total_timesteps = total_timesteps, callback = plotting_callback)
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    
    with ProgressBarManager(total_timesteps) as progress_callback: # this the garanties that the tqdm progress bar closes correctly
        model.learn(total_timesteps = total_timesteps, callback = [auto_save_callback, progress_callback, plotting_callback])

    # The model will be saved under PPO.zip
    model.save(log_dir + '/' + args.algo)

    # sample an observation from the environment
    obs = model.env.reset()
    best_model = PPO.load(log_dir + '/best_model')
    
    # Check that the prediction is the same after loading (for the same observation)
    action = best_model.predict(obs, deterministic = True)
    print("best model prediction: ", action)
    # print('best end effector position:', model.env.get_position(action))
	
if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
