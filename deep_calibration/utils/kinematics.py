import numpy as np
from numpy.linalg import multi_dot
import math
import quaternion


class Kinematics:
    """
    Kinematics of the UR10 arm
            :param delta: (np.ndarray) the delta calibration parameters of the UR10 arm
            :param p_x, p_y, p_z: (np.ndarray) the linear calibration parameters of the UR10 arm
            :param phi_x, phi_y, phi_z: (np.ndarray) the angular calibration parameters of the UR10 arm
            :param quater: (np.ndarray) the quaternion parameters for the base orientation of the UR10 arm
    """

    def __init__(
        self,
        quater=None,
        base_p=np.zeros(3),
        base_phi=np.zeros(3),
        delta=np.zeros(5),
        p_x=np.zeros(2),
        p_y=np.zeros(3),
        p_z=np.zeros(3),
        phi_y=np.zeros(5),
        phi_z=np.zeros(2),
        tool=np.zeros((3, 3)),
        p_base=None,
        R_base=None,
        p_tool=None,
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
        # calibration parameters
        self._calib_prms = {
            "base": {
                "p_x": 0, "p_y": 0, "p_z": 0,
                "phi_x": 0, "phi_y": 0, "phi_z": 0,
                'p_base': base_p, 'phi_base': base_phi
            },
            "joint1": {"p_x": p_x[0], "p_y": p_y[0], "phi_y": phi_y[0]},
            "joint2": {"delta_x": delta[0], "p_z": p_z[0], "phi_y": phi_y[1], "phi_z": phi_z[0]},
            "joint3": {"delta_x": delta[1], "p_z": p_z[1], "phi_y": phi_y[2], "phi_z": phi_z[1]},
            "joint4": {"delta_x": delta[2], "p_y": p_y[1], "p_z": p_z[2], "phi_y": phi_y[3]},
            "joint5": {"delta_z": delta[3], "p_x": p_x[1], "p_y": p_y[2], "phi_y": phi_y[4]},
            "joint6": {"delta_x": delta[4]},
            'tool': [
                tool[0, :], tool[1, :], tool[2, :]
            ]
        }
        self._quater = quater
        self._p_base = p_base
        self._R_base = R_base
        if p_tool is None:
            self._p_tool = [
                np.array([277.23, -46.53, -93.87]) * 0.001,
                np.array([276.49, -48.25, 94.05]) * 0.001,
                np.array([278.44, 103.73, -2.17]) * 0.001,
            ]
        else:
            self._p_tool = p_tool

    def R_baseq(self, q):
        Rb = np.zeros((4, 4))
        Rb[:3, :3] = quaternion.as_rotation_matrix(q)
        Rb[3, 3] = 1
        return Rb

    def Rx(self, theta):
        return np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(theta), -np.sin(theta), 0],
                [0, np.sin(theta), np.cos(theta), 0],
                [0, 0, 0, 1]
            ]
        )

    def Ry(self, theta):
        return np.array(
            [
                [np.cos(theta), 0, np.sin(theta), 0],
                [0, 1, 0, 0],
                [-np.sin(theta), 0, np.cos(theta), 0],
                [0, 0, 0, 1]
            ]
        )

    def Rz(self, theta):
        return np.array(
            [
                [np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta), np.cos(theta), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        )

    def Tx(self, x):
        return np.array(
            [
                [1, 0, 0, x],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        )

    def Ty(self, y):
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, y],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        )

    def Tz(self, z):
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, z],
                [0, 0, 0, 1]
            ]
        )

    def skew(self, phi):
        """
            Computes the skew symetric matrix of a certain set of angles
                :param qhi: (np.ndarray) the angles
                :return: (np.ndarray) the skew symetric matrix
        """
        return np.array([[0, -phi[2], phi[1]], [phi[2], 0, -phi[0]], [-phi[1], phi[0], 0]])

    def forward_kinematics(self, k=None, j=None, q=np.array([0, 0, 0, 0, 0, 0])):
        """
        Computes the forward kinematics of the UR10 arm robot
                :param q: (np.ndarray) the joint angles
                :return: (np.ndarray) the cartesian position and orientation of the end effector
        """

        if self._p_base is not None and self._R_base is not None:
            H_base = [np.identity(4)]
            H_base[0][:3, :3] = self._R_base + self.skew(self._calib_prms["base"]["phi_base"])
            H_base[0][:3, 3] = self._p_base + self._calib_prms["base"]["p_base"]
            if k is not None:
                k = k + 2
        else:
            if self._quater is not None:
                self._quater = np.quaternion(
                    self._quater[0],
                    self._quater[1],
                    self._quater[2],
                    self._quater[3])
                H_base = [
                    self.Tx(self._calib_prms["base"]["p_x"]),
                    self.Ty(self._calib_prms["base"]["p_y"]),
                    self.Tz(self._DH_used["base"] + self._calib_prms["base"]["p_z"]),
                    self.R_baseq(self._quater),
                ]
            else:
                H_base = [
                    self.Tx(self._calib_prms["base"]["p_x"]),
                    self.Ty(self._calib_prms["base"]["p_y"]),
                    self.Tz(self._DH_used["base"] + self._calib_prms["base"]["p_z"]),
                    self.Rx(self._calib_prms["base"]["phi_x"]),
                    self.Ry(self._calib_prms["base"]["phi_y"]),
                    self.Rz(self._calib_prms["base"]["phi_z"]),
                ]
            if k is not None:
                k = k + 7

        H_01 = [
            self.Rz(q[0]),
            self.Tx(self._DH_used["joint1"] + self._calib_prms["joint1"]["p_x"]),
            self.Ty(self._calib_prms["joint1"]["p_y"]),
            self.Ry(self._calib_prms["joint1"]["phi_y"]),
        ]

        H_12 = [
            self.Rx(q[1] + self._calib_prms["joint2"]["delta_x"]),
            self.Tz(self._DH_used["joint2"] + self._calib_prms["joint2"]["p_z"]),
            self.Ry(self._calib_prms["joint2"]["phi_y"]),
            self.Rz(self._calib_prms["joint2"]["phi_z"]),
        ]

        H_23 = [
            self.Rx(q[2] + self._calib_prms["joint3"]["delta_x"]),
            self.Tx(self._DH_used["joint3"]["x"]),  # added
            self.Tz(self._DH_used["joint3"]["z"] + self._calib_prms["joint3"]["p_z"]),
            self.Ry(self._calib_prms["joint3"]["phi_y"]),
            self.Rz(self._calib_prms["joint3"]["phi_z"]),
        ]

        H_34 = [
            self.Rx(q[3] + self._calib_prms["joint4"]["delta_x"]),
            self.Tx(self._DH_used["joint4"]),  # added
            self.Ty(self._calib_prms["joint4"]["p_y"]),
            self.Tz(self._calib_prms["joint4"]["p_z"]),
            self.Ry(self._calib_prms["joint4"]["phi_y"]),
        ]

        H_45 = [
            self.Rz(q[4] + self._calib_prms["joint5"]["delta_z"]),
            self.Tx(self._calib_prms["joint5"]["p_x"]),
            self.Ty(self._calib_prms["joint5"]["p_y"]),
            self.Tz(self._DH_used["joint5"]),  # added
            self.Ry(self._calib_prms["joint5"]["phi_y"]),
        ]

        H_56 = [
            self.Rx(q[5] + self._calib_prms["joint6"]["delta_x"]),
            self.Tx(self._DH_used["joint6"]),  # added
        ]

        H_tool = [np.identity(4)]
        if j is not None:
            if self._p_tool is None:
                raise ValueError('p_tool should be defined to use j index')
            H_tool[0][:3, 3] = self._p_tool[j] + self._calib_prms['tool'][j]

        H_total = H_base + H_01 + H_12 + H_23 + H_34 + H_45 + H_56 + H_tool

        if k is None:
            H_robot = multi_dot(H_total)
        elif k == 1:
            H_robot = H_total[0]
        else:
            H_robot = multi_dot(H_total[:k])

        return H_robot[0:3, 3], H_robot[0:3, 0:3]
