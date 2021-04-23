from gym.envs.registration import register
import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))  # main directory path
# rl_zoo_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'rl-baselines3-zoo'))
# sys.path.append(rl_zoo_dir)

register(
    id='calib_env-v0',
    entry_point='deep_calibration.envs:CalibrationEnv',
)