from gym.envs.registration import register

register(
    id='calib_env-v0',
    entry_point='deep_calibration.envs:CalibrationEnv',
)