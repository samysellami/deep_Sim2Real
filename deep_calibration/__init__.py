from gym.envs.registration import register
import os

script_dir = os.path.dirname(os.path.realpath(__file__))  # main directory path

register(
    id='calib_env-v0',
    entry_point='deep_calibration.envs:CalibrationEnv',
)