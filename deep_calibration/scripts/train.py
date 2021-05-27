import numpy as np
import configparser
import math
import os
import gym
import json 

from deep_calibration import script_dir
from deep_calibration.utils.kinematics import Kinematics
from deep_calibration.utils.callbacks import SaveOnBestTrainingRewardCallback
from deep_calibration.utils.callbacks import EvalCallback
from deep_calibration.utils.callbacks import PlottingCallback
from deep_calibration.utils.callbacks import ProgressBarManager, ProgressBarCallback
from deep_calibration.utils.wrappers import NormalizeActionWrapper, TimeLimitWrapper
from deep_calibration.utils.evaluation import evaluate_policy
from deep_calibration.utils.exp_manager import ExperimentManager

from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization



ALGOS = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}

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
        "--config","--configs", help = "config file name", type = str, metavar = "CONFIG_NAME", dest = "config", required = True
    )
    parser.add_argument("--algo", type = str, default=None, dest = "algo", help = "algorithm", required = True)
    parser.add_argument(
    "-optimize", "--optimize-hyperparameters", action="store_true", default=False, dest = "optimize",  help="Run hyperparameters search"
    )

    return parser


def main(args, unknown_args):
    
    # path to the configuration file 
    path = os.path.join(script_dir,'configs', args.config)
    
    # check if the algorithm is implemented     
    if  args.algo not in ALGOS:   
        raise NotImplementedError('the algorithm specified has not been recognized !!')

    # parsing the config file and the args parser 
    config_file = configparser.ConfigParser()
    config_file.read(path)
    n_timesteps = config_file.getint('ADAPT','total_timesteps')
    env_id = config_file['ADAPT']['environment']
    batch_size = config_file.getint(args.algo, 'batch_size')        
    net_arch = json.loads(config_file.get(args.algo, 'net_arch')) 
    seed = config_file.getint(args.algo, 'seed')        
    n_eval_episodes = 5
    n_eval_test = 5
    eval_freq = 10
    n_trials = 100

    # Create the saving directory
    log_path = os.path.join(script_dir,'saved_models')
    log_folder = os.path.join(script_dir,'saved_models', args.algo)
    os.makedirs(log_folder, exist_ok = True)
    with open(f"{script_dir}/saved_models/{args.algo}/best_reward.npy", 'wb') as f:
        np.save(f, np.array([-np.inf]))

    # Create and wrap the environment
    # env = make_vec_env(env_name, n_envs = 1, monitor_dir = log_folder)
    # eval_env = Monitor(gym.make(f"deep_calibration:{env_id}"), log_folder)  
    # eval_env = NormalizeActionWrapper(eval_env)
    # env = Monitor(gym.make(f"deep_calibration:{env_id}"), log_folder)
    # env = NormalizeActionWrapper(env)
    # env = DummyVecEnv([lambda: env])
    np.set_printoptions(precision=5, suppress=True)


    # create the model
    # model = algo(MlpPolicy, env, verbose = 1)

    # if args.optimize == True:

    exp_manager = ExperimentManager(
        args,
        algo = args.algo,
        env_id = env_id,
        log_folder = log_path,
        n_timesteps = n_timesteps,
        eval_freq = eval_freq,
        n_eval_episodes = n_eval_episodes,
        n_trials = n_trials,
        optimize_hyperparameters =  args.optimize,
    )

    # Prepare experiment and launch hyperparameter optimization if needed
    model = exp_manager.setup_experiment()

    # Normal training
    if model is not None:
        exp_manager.learn(model)
        # exp_manager.save_trained_model(model)
    else:
        exp_manager.hyperparameters_optimization()

    return










    model = ALGOS[args.algo](
        'MlpPolicy', env, 
        batch_size = batch_size, 
        policy_kwargs = dict(net_arch = net_arch), 
        verbose = 1, seed = seed, 
    )

    # Create Callbacks and train the model
    auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq = 1, log_dir = log_folder, verbose = 1)
    plotting_callback = PlottingCallback(log_dir = log_folder)
    eval_callback = EvalCallback(eval_env, best_model_save_path = log_folder,
                                log_path = log_folder, eval_freq = eval_freq, n_eval_episodes = n_eval_episodes,
                                deterministic = True, render = False, verbose = 0)

    with ProgressBarManager(n_timesteps) as progress_callback: # this the garanties that the tqdm progress bar closes correctly
        model.learn(total_timesteps = n_timesteps, callback = [eval_callback, progress_callback])
    del model

    # get the best predition from the best model
    best_model = ALGOS[args.algo].load(log_folder + '/best_model')  

    # sample an observation from the environment and compute the action
    dists = []
    for i in range(n_eval_episodes):
        obs = eval_env.reset()
        action = best_model.predict(obs, deterministic = True)[0]
        action = eval_env.rescale_action(action)

        dist = eval_env.distance_to_goal(action)
        print(f'best distance to goal for config {i} is  {dist}')
        dists.append(dist)

    # print("best calibration parameters: ", action)   
    print('best mean distance: ', np.mean(dists))

    # testing for random configurations
    eval_env.rand = 1
    dists = []
    for i in range(n_eval_test):
        obs = eval_env.reset()
        action = best_model.predict(obs, deterministic = True)[0]
        action = eval_env.rescale_action(action)

        dist = eval_env.distance_to_goal(action)
        print(f'best distance to goal for a random config {i} is  {dist}')
        dists.append(dist)

    print('best random mean distance: ', np.mean(dists))
	
if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
