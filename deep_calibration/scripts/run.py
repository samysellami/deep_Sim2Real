import argparse
import importlib
import os
import sys
import configparser
import json
import gym

import numpy as np
import torch as th
import yaml
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

from deep_calibration.utils.wrappers import NormalizeActionWrapper, TimeLimitWrapper
import deep_calibration.utils.import_envs  # noqa: F401 pylint: disable=unused-import
from deep_calibration.utils.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from deep_calibration.utils.exp_manager import ExperimentManager
from deep_calibration.utils.utils import StoreDict
from deep_calibration import script_dir
from deep_calibration.calibration.utils.utils import *


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
    parser.add_argument("--config", "--configs", help="config file name",
                        type=str, metavar="CONFIG_NAME", dest="config", required=True)
    parser.add_argument("--algo", type=str, default=None,
                        dest="algo", help="algorithm", required=True)
    parser.add_argument("-optimize", "--optimize-hyperparameters", action="store_true",
                        default=False, dest="optimize", help="Run hyperparameters search")
    parser.add_argument(
        "--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0,
        type=int)
    parser.add_argument("--load-best", action="store_true", default=False,
                        help="Load best model instead of last model if available")
    parser.add_argument("--prms",  type=list, default=None,
                        dest="prms", help="parameters to tune", required=True)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)

    return parser


def std(data, mean):
    n = len(data)
    deviations = [(x - mean) ** 2 for x in data]
    std = sum(deviations) / n
    std = np.sqrt(std)
    return std


def main(args, unknown_args):  # noqa: C901

    tune = False

    # check if the algorithm is implemented
    if args.algo not in ALGOS:
        raise NotImplementedError('the algorithm specified has not been recognized !!')

    # path to the configuration file
    config_path = os.path.join(script_dir, 'configs', args.config)

    # parsing the config file and the args parser
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            try:
                loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr

            except yaml.YAMLError as exc:
                print(exc)

    n_timesteps = loaded_args['ADAPT']['total_timesteps']
    env_id = loaded_args['ADAPT']['environment']
    env_kwargs = {'prms': args.prms}

    n_eval_episodes = 10
    n_eval_test = 5
    eval_freq = 10
    n_trials = 20

    # Create the saving directory
    log_folder = os.path.join(script_dir, 'saved_models')

    algo = args.algo
    folder = log_folder

    # if args.exp_id == 0:
    #     args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
    #     print(f"Loading latest experiment, id={args.exp_id}")

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    if args.load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    env_kwargs = {'prms': args.prms}

    env = Monitor(gym.make(f"deep_calibration:{env_id}", **env_kwargs), log_path)
    eval_env = NormalizeActionWrapper(env)

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    best_model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)

    obs = env.reset()
    np.set_printoptions(precision=7, suppress=True)

    identified_prms = save_read_data(
        file_name='p_ij',
        io='r',
        data=None
    )
    calib_prms = identified_prms['calib_prms']

    try:
        # sample an observation from the environment and compute the action
        print('\n ############# resutls of the parameters tuning ##################### \n')

        dists = []
        actions = []
        for i in range(n_eval_episodes):
            obs = eval_env.reset()
            action = best_model.predict(obs, deterministic=True)[0]
            action = eval_env.rescale_action(action)
            actions.append(action)
            dist = eval_env.distance_to_goal(action) * 1000
            dists.append(dist)
            # print(f'distance to goal for config {i} = {dist:.4f}')
            # print(f'parameters for config {i} is {action}')

        print(f' average distance to goal : {np.mean(dists):.7f}')

        if tune:
            actions = actions + calib_prms

        ind_min = np.argmin(dists)
        ind_max = np.argmax(dists)
        best_action = actions[ind_min]
        worst_action = actions[ind_max]

        print(f' worst distance to goal : {dists[ind_max]:.7f}')
        print(f' best distance to goal : {dists[ind_min]:.7f}')
        # print(f'best action =  {best_action}')
        # print(f'worst action =  {worst_action}')
        # std_actions = std(actions, best_action)
        # print(f'std actions =  {std_actions}')

        read_data = save_read_data(
            file_name='best_action',
            io='r',
            data=None
        )
        prms_action = read_data['prms_action']

        save_read_data(
            file_name='best_action',
            io='w',
            data={
                'best_action': best_action,
                'prms_action': prms_action,
            }
        )

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
