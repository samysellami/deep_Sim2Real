import numpy as np
import math
from scipy.linalg import block_diag


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

        # jacobian of the elastostatic model(joints only)
        self._prmsJ_theta = {
            "joint1": ["delta_z"],
            "joint2": ["delta_x"],
            "joint3": ["delta_x"],
            "joint4": ["delta_x"],
            "joint5": ["delta_z"],
            "joint6": ["delta_x"]
        }

        # jacobian of the elastostatic model (links only)
        self._prmsJ_links = {
            "base": ["p_x", "p_y", "p_z", "phi_x", "phi_y", "phi_z"],
            "joint1": ["p_x", "p_y", "p_z", "phi_x", "phi_y", "phi_z"],
            "joint2": ["p_x", "p_y", "p_z", "phi_x", "phi_y", "phi_z"],
            "joint3": ["p_x", "p_y", "p_z", "phi_x", "phi_y", "phi_z"],
            "joint4": ["p_x", "p_y", "p_z", "phi_x", "phi_y", "phi_z"],
            "joint5": ["p_x", "p_y", "p_z", "phi_x", "phi_y", "phi_z"],
            "joint6": ["p_x", "p_y", "p_z", "phi_x", "phi_y", "phi_z"],
        }

        E = 7e10  # Young's modulus
        G = 25.5e9  # shear modulus

        # parameters for cylinder links
        d = [0.12] * 7
        L = [0.128, 0.1639, 0.6127, 0.5716, 0.1639, 0.1157, 0.0922]

        self._K = block_diag(
            self.R().dot(self.K(d, L, E, G, 0).dot(self.R().transpose())),
            self.K(d, L, E, G, 1),
            self.R().dot(self.K(d, L, E, G, 2).dot(self.R().transpose())),
            self.R().dot(self.K(d, L, E, G, 3).dot(self.R().transpose())),
            self.K(d, L, E, G, 4),
            self.R().dot(self.K(d, L, E, G, 5).dot(self.R().transpose())),
            self.K(d, L, E, G, 6),
        )

    def S(self, d):
        return (np.pi * (d ** 2))/4

    def I(self, d):
        return (np.pi * (d ** 4))/64

    # compliance matrices for the links
    def K(self, d, L, E, G, i):
        return np.array(
            [
                [(E * self.S(d[i]))/L[i], 0, 0, 0, 0, 0],
                [0, (12 * E * self.I(d[i]))/(L[i] ** 3), 0, 0, 0, -(6 * E * self.I(d[i]))/(L[i] ** 2)],
                [0, 0, (12 * E * self.I(d[i]))/(L[i] ** 3), 0, (6 * E * self.I(d[i]))/(L[i] ** 2), 0],
                [0, 0, 0, (G * 2 * self.I(d[i]))/L[i], 0, 0],
                [0, 0,  (6 * E * self.I(d[i]))/(L[i] ** 2), 0, (4 * E * self.I(d[i]))/L[i], 0],
                [0, -(6 * E * self.I(d[i]))/(L[i] ** 2), 0, 0, 0, (4 * E * self.I(d[i]))/L[i]]
            ]
        )

    def calib_prms(
        self,
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
            "base": {"p_x": 0, "p_y": 0, "p_z": 0, "phi_x": 0, "phi_y": 0, "phi_z": 0,'p_base': base_p, 'phi_base': base_phi},
            "joint1": {"delta_z": 0, "p_x": p_x[0], "p_y": p_y[0], "p_z": 0, "phi_x": 0,  "phi_y": phi_y[0], "phi_z": 0},
            "joint2": {"delta_x": delta[0], "p_x": 0, "p_y": 0, "p_z": p_z[0], "phi_x": 0, "phi_y": phi_y[1], "phi_z": phi_z[0]},
            "joint3": {"delta_x": delta[1], "p_x": 0, "p_y": 0, "p_z": p_z[1], "phi_x": 0, "phi_y": phi_y[2], "phi_z": phi_z[1]},
            "joint4": {"delta_x": delta[2], "p_x": 0, "p_y": p_y[1], "p_z": p_z[2], "phi_x": 0, "phi_y": phi_y[3], "phi_z": 0},
            "joint5": {"delta_z": delta[3], "p_x": p_x[1], "p_y": p_y[2], "p_z": 0, "phi_x": 0, "phi_y": phi_y[4], "phi_z": 0},
            "joint6": {"delta_x": 0, "p_x": 0, "p_y": 0, "p_z": 0, "phi_x": 0, "phi_y": 0, "phi_z": 0},
            'tool': [ tool[0, :], tool[1, :], tool[2, :]],
            "ksi": ksi,
        }
    
    def Ry(self, theta):
        return np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )

    def R(self):
        return  block_diag(
            self.Ry(np.pi/2),
            self.Ry(np.pi/2)
        )
