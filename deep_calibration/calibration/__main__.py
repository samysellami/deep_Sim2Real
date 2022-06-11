import numpy as np
from regex import F
from sqlalchemy import true

from deep_calibration.calibration.utils.utils import *
from deep_calibration.__main__ import main as deep_calib
from deep_calibration.calibration.calibration import Calibration


class DeepCalibration:
    """
    Calibration of the UR10 arm base on the paper "Geometric and elastostatic calibration of robotic manipulator
            using partial pose measurements" and using deep learing algorithms
    """

    def __init__(self):
        self._calib = Calibration()
        self._form_goal_pos = False  # if True compute the goal position from the kinematics without tools
        self.save_data()

    def save_data(self):
        save_read_data(
            file_name='p_ij',
            io='w',
            data={
                'p_ij': self._calib._p_ij,
                'p_base': self._calib._p_base,
                'R_base': self._calib._R_base,
                'p_tool': self._calib._p_tool,
                'prms': self._calib._prms,
                'configs': self._calib._configs,
                'calib_prms': self._calib._delta,
                'goal_position': self._calib._goal_pos,
            }
        )

    def step1(self):
        # step 1 identification of p_base, phi_base and u_tool
        print('\n ############# identification of the base and tool ##################### \n')

        p_base, R_base, p_tool = self._calib.identity_base_tool()
        print(' p_base: \n', p_base * 1000)
        print('\n R_base: \n', R_base)
        print('\n p_tool: \n', [p * 1000 for p in p_tool])

        self._calib._p_base = p_base
        self._calib._R_base = R_base
        self._calib._p_tool = p_tool
        print(
            f'\n distance to goal after the base and tool identification: {self._calib.dist_to_goal(self._form_goal_pos) * 1000:.7f}')

    def step2(self):
        # step 2 identification of the calibration parameters
        print('\n ############# identification of the calibration parameters ##################### \n')

        for i in range(5):
            calib_prms = self._calib.identify_calib_prms()
            self._calib._delta += calib_prms
            print(
                f' distance to goal after calibration {i} : {self._calib.dist_to_goal(self._form_goal_pos) * 1000:.7f}')

        self._calib.update_kinematics(
            prms={'delta': self._calib._delta}
        )
        print('\n delta parameters:', self._calib._delta)

    def step3(self):
        # step 3 identification of the elastostatic parameters
        print('\n ############# identification of the elastostic parameters ##################### \n')

        for i in range(5):
            ksi_prms = self._calib.identify_elastostatic_prms()
            self._calib._ksi += np.diag(ksi_prms)

            print(
                f' distance to goal after calibration {i} : {self._calib.dist_to_goal(self._form_goal_pos) * 1000:.7f}')
        self._calib.update_kinematics(
            prms={'ksi': self._calib._ksi}
        )
        print('\n ksi parameters:', np.diag(self._calib._ksi) * 1000000)

    def deep_calibration(self, prms_to_tune):
        # tuning all the parameters
        deep_calib(['train', '--config', 'config.yml', '--algo', 'sac', '--prms', prms_to_tune])
        deep_calib(['run', '--config', 'config.yml', '--algo', 'sac', '--load-best', '--prms', prms_to_tune])
        self._calib.update_prms()
        # self._calib.base_tool_after_tuning()

def main():
    np.set_printoptions(precision=7, suppress=True)
    tune_all = True  # if True tune the calibration parameters using deep learning
    calibrate = True  # if True use the partial pose measurement geometric calibration
    calibrate_elastostatic = True # if True use the partial pose meaasurement elastostatic calibration
    epsilon_ = 10
    epsilon = 10

    deep_calib = DeepCalibration()
    print(f'\n initial distance to goal: {deep_calib._calib.dist_to_goal(deep_calib._form_goal_pos) * 1000:.7f}')

    while epsilon > 0.0001 and calibrate:
        # for i in range(5):
        # step 1 identification of p_base, phi_base and u_tool
        deep_calib.step1()
        deep_calib.save_data()

        # step 2 identification of the calibration parameters
        deep_calib.step2()
        deep_calib.save_data()

        dist = deep_calib._calib.dist_to_goal(deep_calib._form_goal_pos) * 1000
        epsilon = np.abs(dist - epsilon_)
        epsilon_ = dist

    epsilon_ = 10
    epsilon = 10
    while epsilon > 0.00000001 and calibrate_elastostatic:
        # for i in range(5):
        # step 3 identification of the elastostatic parameters
        deep_calib.step3()
        deep_calib.save_data()

        dist = deep_calib._calib.dist_to_goal(deep_calib._form_goal_pos) * 1000
        epsilon = np.abs(dist - epsilon_)
        epsilon_ = dist

    if tune_all:
        # tuning all the parameters
        print('\n ############# tuning the calibration parameters ##################### \n')
        prms_to_tune = ['epsilon']
        deep_calib.deep_calibration(prms_to_tune)
        # deep_calib.step1()

    print('\n final parameters:', deep_calib._calib._prms[prms_to_tune[0]])
    print(f'\n final distance to goal: {deep_calib._calib.dist_to_goal(False) * 1000:.7f}')


if __name__ == "__main__":
    main()
