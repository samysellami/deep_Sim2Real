import numpy as np

from deep_calibration.calibration.utils import *
from deep_calibration.__main__ import main as deep_calib
from deep_calibration.calibration.calibration import Calibration


def main():

    np.set_printoptions(precision=4, suppress=True)
    tune = True

    calib = Calibration()
    calib._delta = np.zeros(5)

    # step 1 identification of p_base, phi_base and u_tool
    print('\n ############# identification of the base and tool ##################### \n')
    p_base, R_base, p_tool = calib.identity_base_tool()
    print('\n p_base: \n', p_base * 1000)
    print('\n R_base: \n', R_base)
    print('\n p_tool: \n', [p * 1000 for p in p_tool])

    calib._p_base = p_base
    calib._R_base = R_base
    calib._p_tool = p_tool
    print(f'\n distance to goal after the base and tool identification: {calib.dist_to_goal() * 1000:.4f}', )

    save_read_data(
        file_name='p_ij',
        io='w',
        data={
            'p_ij': calib._p_ij,
            'p_base': calib._p_base,
            'R_base': calib._R_base,
            'p_tool': calib._p_tool,
            'calib_prms': calib._delta,
            'goal_position': calib._goal_pos,
        }
    )

    # tuning the base and tool parameters
    if tune:
        print('\n ############# tuning the base and tool ##################### \n')
        prms_to_tune = ['base_p', 'tool']
        deep_calib(['train', '--config', 'config.yml', '--algo', 'sac', '--prms', prms_to_tune])
        deep_calib(['run', '--config', 'config.yml', '--algo', 'sac', '--load-best', '--prms', prms_to_tune])
        calib.update_prms()
        calib.base_tool_after_tuning()
        print(f'\n distance to goal after tuning the base and tool: {calib.dist_to_goal() * 1000:.4f}')

    # step 2 identification of the calibration parameters
    print('\n ############# identification of the calibraition parameters ##################### \n')
    for i in range(3):
        calib_prms = calib.identify_calib_prms()
        calib._delta += calib_prms

        # print('delta_calib_prms:', calib_prms)
        print(f'distance to goal after calibration {i} : {calib.dist_to_goal() * 1000:.4f}')
    calib.update_kinematics()
    print('calib_prms:', calib._delta)

    save_read_data(
        file_name='p_ij',
        io='w',
        data={
            'p_ij': calib._p_ij,
            'p_base': calib._p_base,
            'R_base': calib._R_base,
            'p_tool': calib._p_tool,
            'calib_prms': calib._delta,
            'goal_position': calib._goal_pos,
        }
    )

    # tuning the calibration parameters
    if tune:
        print('\n ############# tuning the calibration parameters ##################### \n')
        prms_to_tune = ['delta']
        deep_calib(['train', '--config', 'config.yml', '--algo', 'sac', '--prms', prms_to_tune])
        deep_calib(['run', '--config', 'config.yml', '--algo', 'sac', '--load-best', '--prms', prms_to_tune])
        calib.update_prms()

        print('calib_prms:', calib._delta)
        print(f'\n distance to goal after tuning the calibration parameters: {calib.dist_to_goal() * 1000:.4f}')


if __name__ == "__main__":
    main()
