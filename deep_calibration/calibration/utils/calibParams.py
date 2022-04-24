import numpy as np
import math


class CalibParams:
    """
    Calibration parameters to be identified 
    """

    def __init__(
        self,
    ):

        # DH  = [a, alpha, d, theta] --- DH parameters of the UR10 arm
        self._DH = np.array(
            [
                [0, -math.pi / 2, 0.128, 0],
                [0.6127, 0, 0, 0],
                [0.5716, 0, 0, 0],
                [0, -math.pi / 2, 0.1639, 0],
                [0, math.pi / 2, 0.1157, 0],
                [0, 0, 0.0922, 0],
            ]
        )

        self._DH_used = {
            "base": self._DH[0, 2],  # 128
            "joint1": self._DH[3, 2] + self._DH[5, 2],  # 163.9 + 92.2
            "joint2": self._DH[1, 0],  # 621.7
            "joint3": {"z": self._DH[2, 0], "x": -(self._DH[3, 2] + self._DH[5, 2])},  # z: 571.6, x: 163.9 + 92.2
            "joint4": self._DH[3, 2],  # 163.9
            "joint5": self._DH[4, 2],  # 115.7
            "joint6": self._DH[5, 2],  # 92.2
        }

        # jacobian for the elastostatic model
        self._prms_J = {
            "joint1": "delta_z",
            "joint2": "delta_x",
            "joint3": "delta_x",
            "joint4": "delta_x",
            "joint5": "delta_z",
            "joint6": "delta_x"
        }

    def calib_prms(self,
                   base_p=np.zeros(3),
                   base_phi=np.zeros(3),
                   delta=np.zeros(5),
                   p_x=np.zeros(2),
                   p_y=np.zeros(3),
                   p_z=np.zeros(3),
                   phi_y=np.zeros(5),
                   phi_z=np.zeros(2),
                   tool=np.zeros((3, 3)),
                   ksi=np.zeros((5, 5)),
                   ):
        return {
            "base": {
                "p_x": 0, "p_y": 0, "p_z": 0,
                "phi_x": 0, "phi_y": 0, "phi_z": 0,
                'p_base': base_p, 'phi_base': base_phi
            },
            "joint1": {"delta_z": 0, "p_x": p_x[0], "p_y": p_y[0], "phi_y": phi_y[0]},
            "joint2": {"delta_x": delta[0], "p_z": p_z[0], "phi_y": phi_y[1], "phi_z": phi_z[0]},
            "joint3": {"delta_x": delta[1], "p_z": p_z[1], "phi_y": phi_y[2], "phi_z": phi_z[1]},
            "joint4": {"delta_x": delta[2], "p_y": p_y[1], "p_z": p_z[2], "phi_y": phi_y[3]},
            "joint5": {"delta_z": delta[3], "p_x": p_x[1], "p_y": p_y[2], "phi_y": phi_y[4]},
            "joint6": {"delta_x": 0},
            'tool': [
                tool[0, :], tool[1, :], tool[2, :]
            ],
            "ksi": ksi,
        }
