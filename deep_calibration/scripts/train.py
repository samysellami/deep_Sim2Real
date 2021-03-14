import numpy as np
import configparser
import math
import os
import gym

from deep_calibration import script_dir
from deep_calibration.utils.kinematics import Kinematics
from deep_calibration.scripts.callbacks import SaveOnBestTrainingRewardCallback
from deep_calibration.scripts.callbacks import PlottingCallback
from deep_calibration.scripts.callbacks import ProgressBarManager, ProgressBarCallback
from deep_calibration.scripts.wrappers import NormalizeActionWrapper, TimeLimitWrapper


from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
# from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.sac.policies import MlpPolicy
# from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization


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
    
    # parsing the config file and the args parser 
    config_file = configparser.ConfigParser()
    config_file.read(path)
    total_timesteps = config_file.getint('ADAPT','total_timesteps')
    env_name = config_file['ADAPT']['environment']

    # define the algorithm 
    if args.algo == 'PPO':    
        algo = PPO
    elif args.algo == 'SAC':
        algo = SAC
    elif args.algo == 'TD3':
        algo = TD3
    else:
        raise NotImplementedError('the algorithm specified is has not been recognized !!')

    # Create the saving directory
    log_dir = os.path.join(script_dir,'saved_models', args.algo)
    os.makedirs(log_dir, exist_ok = True)

    # Create and wrap the environment
    # env = make_vec_env(env_name, n_envs = 1, monitor_dir = log_dir)
    eval_env = Monitor(gym.make(env_name), log_dir)  
    env = Monitor(gym.make(env_name), log_dir)
    env = DummyVecEnv([lambda: env])
    # env = NormalizeActionWrapper(env)

    # create the model
    model = algo(MlpPolicy, env, verbose = 1)

    # Create Callbacks and train the model
    auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq = 1, log_dir = log_dir, verbose = 1)
    plotting_callback = PlottingCallback(log_dir = log_dir)
    eval_callback = EvalCallback(eval_env, best_model_save_path = log_dir,
                                log_path = log_dir, eval_freq = 10, n_eval_episodes = 1,
                                deterministic = True, render = False, verbose = 0)

    with ProgressBarManager(total_timesteps) as progress_callback: # this the garanties that the tqdm progress bar closes correctly
        model.learn(total_timesteps = total_timesteps, callback = [eval_callback, progress_callback])
    del model

    # get the best predition from the best model
    best_model = algo.load(log_dir + '/best_model')  

    # sample an observation from the environment and compute the action
    obs = eval_env.reset()
    action = best_model.predict(obs, deterministic = True)[0]
    print("best calibration parameters: ", action)
    print('best distance to goal: ', eval_env.distance_to_goal(action))

    # evaluate the best policy
    mean_reward, std_reward = evaluate_policy(
        best_model,
        eval_env,
        n_eval_episodes = 1,
        render = False,
        deterministic = True,
    )
    print(f"mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
	
if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
