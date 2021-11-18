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
        base=np.zeros(6),
        delta=np.zeros(5),
        p_x=np.zeros(2),
        p_y=np.zeros(3),
        p_z=np.zeros(3),
        phi_y=np.zeros(5),
        phi_z=np.zeros(2),
        p_base=None,
        R_base=None,
        p_tool=None,
    ):

        # DH  = [a, alpha, d, theta] --- DH parameters of the UR10 arm
        self.DH = np.array(
            [
                [0, -math.pi / 2, 0.128, 0],
                [0.6127, 0, 0, 0],
                [0.5716, 0, 0, 0],
                [0, -math.pi / 2, 0.1639, 0],
                [0, math.pi / 2, 0.1157, 0],
                [0, 0, 0.0922, 0],
            ]
        )

        self.DH_used = {
            "base": self.DH[0, 2],  # 128
            "joint1": self.DH[3, 2] + self.DH[5, 2],  # 163.9 + 92.2
            "joint2": self.DH[1, 0],  # 621.7
            "joint3": self.DH[2, 0],  # 571.6
            "joint4": self.DH[4, 2],  # 115.7
        }
        # calibration parameters
        self.calib_prms = {
            "base": {
                "p_x": base[0],
                "p_y": base[1],
                "p_z": base[2],
                "phi_x": base[3],
                "phi_y": base[4],
                "phi_z": base[5],
            },
            "joint1": {"p_x": p_x[0], "p_y": p_y[0], "phi_y": phi_y[0]},
            "joint2": {"delta_x": delta[0], "p_z": p_z[0], "phi_y": phi_y[1], "phi_z": phi_z[0]},
            "joint3": {"delta_x": delta[1], "p_z": p_z[1], "phi_y": phi_y[2], "phi_z": phi_z[1]},
            "joint4": {"delta_x": delta[2], "p_y": p_y[1], "p_z": p_z[2], "phi_y": phi_y[3]},
            "joint5": {"delta_z": delta[3], "p_x": p_x[1], "p_y": p_y[2], "phi_y": phi_y[4]},
            "joint6": {"delta_x": delta[4]},
        }
        self.quater = quater
        self.p_base = p_base
        self.R_base = R_base
        if p_tool is None:
            self.p_tool = [
                np.array([277.23, -46.53, -93.87]) * 0.001,
                np.array([276.49, -48.25, 94.05]) * 0.001,
                np.array([278.44, 103.73, -2.17]) * 0.001,
            ]
        else:
            self.p_tool = p_tool

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

    def forward_kinematics(self, k=None, j=None, q=np.array([0, 0, 0, 0, 0, 0])):
        """
        Computes the forward kinematics of the UR10 arm robot
                :param q: (np.ndarray) the joint angles
                :return: (np.ndarray) the cartesian position and orientation of the end effector
        """

        if self.p_base is not None and self.R_base is not None:
            H_base = [np.identity(4)]
            H_base[0][:3, :3] = self.R_base
            H_base[0][:3, 3] = self.p_base
        else:
            if self.quater is not None:
                self.quater = np.quaternion(
                    self.quater[0],
                    self.quater[1],
                    self.quater[2],
                    self.quater[3])
                H_base = [
                    self.Tx(self.calib_prms["base"]["p_x"]),
                    self.Ty(self.calib_prms["base"]["p_y"]),
                    self.Tz(self.DH_used["base"] + self.calib_prms["base"]["p_z"]),
                    self.R_baseq(self.quater),
                ]
            else:
                H_base = [
                    self.Tx(self.calib_prms["base"]["p_x"]),
                    self.Ty(self.calib_prms["base"]["p_y"]),
                    self.Tz(self.DH_used["base"] + self.calib_prms["base"]["p_z"]),
                    self.Rx(self.calib_prms["base"]["phi_x"]),
                    self.Ry(self.calib_prms["base"]["phi_y"]),
                    self.Rz(self.calib_prms["base"]["phi_z"]),
                ]

        H_01 = [
            self.Rz(q[0]),
            self.Tx(self.DH_used["joint1"] + self.calib_prms["joint1"]["p_x"]),
            self.Ty(self.calib_prms["joint1"]["p_y"]),
            self.Ry(self.calib_prms["joint1"]["phi_y"]),
        ]

        H_12 = [
            self.Rx(q[1] + self.calib_prms["joint2"]["delta_x"]),
            self.Tz(self.DH_used["joint2"] + self.calib_prms["joint2"]["p_z"]),
            self.Ry(self.calib_prms["joint2"]["phi_y"]),
            self.Rz(self.calib_prms["joint2"]["phi_z"]),
        ]

        H_23 = [
            self.Rx(q[2] + self.calib_prms["joint3"]["delta_x"]),
            self.Tz(self.DH_used["joint3"] + self.calib_prms["joint3"]["p_z"]),
            self.Ry(self.calib_prms["joint3"]["phi_y"]),
            self.Rz(self.calib_prms["joint3"]["phi_z"]),
        ]

        H_34 = [
            self.Rx(q[3] + self.calib_prms["joint4"]["delta_x"]),
            self.Ty(self.calib_prms["joint4"]["p_y"]),
            self.Tz(self.DH_used["joint4"] + self.calib_prms["joint4"]["p_z"]),
            self.Ry(self.calib_prms["joint4"]["phi_y"]),
        ]

        H_45 = [
            self.Rz(q[4] + self.calib_prms["joint5"]["delta_z"]),
            self.Tx(self.calib_prms["joint5"]["p_x"]),
            self.Ty(self.calib_prms["joint5"]["p_y"]),
            self.Ry(self.calib_prms["joint5"]["phi_y"]),
        ]

        H_56 = [self.Rx(q[5] + self.calib_prms["joint6"]["delta_x"])]

        H_tool = [np.identity(4)]
        if j is not None:
            if self.p_tool is None:
                raise ValueError('p_tool should be defined to use j index')
            H_tool[0][:3, 3] = self.p_tool[j]

        H_total = H_base + H_01 + H_12 + H_23 + H_34 + H_45 + H_56 + H_tool
        print(H_total)

        if k is None:
            H_robot = multi_dot(H_total)
        elif k == 1:
            H_robot = H_total[0]
        else:
            H_robot = multi_dot(H_total[:k])

        return H_robot[0:3, 3], H_robot[0:3, 0:3]
