import numpy as np
from numpy.linalg import multi_dot
import math
import quaternion
from sqlalchemy import true

from deep_calibration.calibration.utils.jacobian import Jacobian
from deep_calibration.calibration.utils.calibParams import CalibParams


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
        ksi=np.zeros((5, 5)),
    ):

        # calibration parameters
        calibParams = CalibParams()
        # DH  = [a, alpha, d, theta] --- DH parameters of the UR10 arm
        self._DH = calibParams._DH
        self._DH_used = calibParams._DH_used

        self._calib_prms = calibParams.calib_prms(
            base_p,  base_phi, delta, p_x, p_y, p_z,
            phi_y, phi_z,
            tool, ksi
        )
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

        # relative position of the force ap plied
        self._p_force = np.array([277.23, -46.53, -93.87]) * 0.001

        # Jacobian of the robot without the applied force
        self._jacob = Jacobian(self)

        # the jacobian for the elastostaic model (joints only)
        prmsJ_theta = calibParams._prmsJ_theta
        self._prmsJ_theta = {joint: prmsJ_theta[joint]
                             for count, joint in enumerate(prmsJ_theta.keys()) if count < len(ksi)}
        self._jacob_theta = Jacobian(
            self,
            prms_J = self._prmsJ_theta
        )
        self._theta = np.zeros(len(self._prmsJ_theta))

        # the jacobian for the elastostaic model (links only)
        self._K = calibParams._K
        self._prmsJ_links = calibParams._prmsJ_links
        self._jacob_links = Jacobian(
            self,
            prms_J = self._prmsJ_links,
            jacob_link=True
        )
        # elastostatic deflections

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

    def forward_kinematics(
            self, k=None, j=None, q=np.array([0, 0, 0, 0, 0, 0]),
            ksi=None, ksi_link=None, F=None, f=None):
        """
        Computes the forward kinematics of the UR10 arm robot
                :param q: (np.ndarray) the joint angles
                :return: (np.ndarray) the cartesian position and orientation of the end effector
        """
        # elastostatic parameters
        if ksi is not None:
            if F is None:
                raise ValueError('You need to specify the value force F')
            J_if = self._jacob_theta.build_jacobian(q=q, j=j, f=True)
            self._theta = self._calib_prms["ksi"].dot((J_if.transpose()).dot(F))

        delta_t = 0
        if ksi_link is not None:
            J_ij = self._jacob_links.build_jacobian(q=q, j=j)
            delta_t = J_ij.dot(np.linalg.inv(self._K)).dot(J_ij.transpose()).dot(F)
            # print(delta_t)

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
                    self.Tz(self._DH_used["base"] + self._calib_prms["base"]["p_z"]),
                    self.Tx(self._calib_prms["base"]["p_x"]),
                    self.Ty(self._calib_prms["base"]["p_y"]),
                    self.Tz(self._calib_prms["base"]["p_z"]),
                    self.Rx(self._calib_prms["base"]["phi_x"]),
                    self.Ry(self._calib_prms["base"]["phi_y"]),
                    self.Rz(self._calib_prms["base"]["phi_z"]),
                ]
            if k is not None:
                k = k + 7

        H_01 = [
            self.Rz(q[0] + self._theta[0]),
            self.Tx(self._DH_used["joint1"]),
            self.Tx(self._calib_prms["joint1"]["p_x"]),
            self.Ty(self._calib_prms["joint1"]["p_y"]),
            self.Ry(self._calib_prms["joint1"]["phi_y"]),

            self.Tz(self._calib_prms["joint1"]["p_z"]),
            self.Rx(self._calib_prms["joint1"]["phi_x"]),
            self.Rz(self._calib_prms["joint1"]["phi_z"]),
        ]

        H_12 = [
            self.Rx(q[1] + self._calib_prms["joint2"]["delta_x"] + self._theta[1]),
            self.Tz(self._DH_used["joint2"]),
            self.Tz(self._calib_prms["joint2"]["p_z"]),
            self.Ry(self._calib_prms["joint2"]["phi_y"]),
            self.Rz(self._calib_prms["joint2"]["phi_z"]),

            self.Tx(self._calib_prms["joint2"]["p_x"]),
            self.Ty(self._calib_prms["joint2"]["p_y"]),
            self.Rx(self._calib_prms["joint2"]["phi_x"]),
        ]

        H_23 = [
            self.Rx(q[2] + self._calib_prms["joint3"]["delta_x"] + self._theta[2]),
            self.Tx(self._DH_used["joint3"]["x"]),  # added
            self.Tz(self._DH_used["joint3"]["z"]),
            self.Tz(self._calib_prms["joint3"]["p_z"]),
            self.Ry(self._calib_prms["joint3"]["phi_y"]),
            self.Rz(self._calib_prms["joint3"]["phi_z"]),

            self.Tx(self._calib_prms["joint3"]["p_x"]),  # added
            self.Ty(self._calib_prms["joint3"]["p_y"]),
            self.Rx(self._calib_prms["joint3"]["phi_x"])
        ]

        H_34 = [
            self.Rx(q[3] + self._calib_prms["joint4"]["delta_x"] + self._theta[3]),
            self.Tx(self._DH_used["joint4"]),  # added
            self.Ty(self._calib_prms["joint4"]["p_y"]),
            self.Tz(self._calib_prms["joint4"]["p_z"]),
            self.Ry(self._calib_prms["joint4"]["phi_y"]),

            self.Tx(self._calib_prms["joint4"]["p_x"]),  # added
            self.Rx(self._calib_prms["joint4"]["phi_x"]),
            self.Rz(self._calib_prms["joint4"]["phi_z"]),
        ]

        H_45 = [
            self.Rz(q[4] + self._calib_prms["joint5"]["delta_z"] + self._theta[4]),
            self.Tz(self._DH_used["joint5"]),  # added
            self.Tx(self._calib_prms["joint5"]["p_x"]),
            self.Ty(self._calib_prms["joint5"]["p_y"]),
            self.Ry(self._calib_prms["joint5"]["phi_y"]),

            self.Tz(self._calib_prms["joint5"]["p_z"]),  # added
            self.Rx(self._calib_prms["joint5"]["phi_x"]),
            self.Rz(self._calib_prms["joint5"]["phi_z"]),
        ]

        H_56 = [
            self.Rx(q[5] + self._calib_prms["joint6"]["delta_x"] + (self._theta[5] if len(self._theta) == 6 else 0)),
            # self.Rx(q[5] + self._calib_prms["joint6"]["delta_x"]),
            self.Tx(self._DH_used["joint6"]),  # added

            self.Tx(self._calib_prms["joint6"]["p_x"]),
            self.Ty(self._calib_prms["joint6"]["p_y"]),
            self.Tz(self._calib_prms["joint6"]["p_z"]),  # added
            self.Rx(self._calib_prms["joint6"]["phi_x"]),
            self.Ry(self._calib_prms["joint6"]["phi_y"]),
            self.Rz(self._calib_prms["joint6"]["phi_z"]),
        ]

        H_tool = [np.identity(4)]
        if j is not None:
            if self._p_tool is None:
                raise ValueError('p_tool should be defined to use j index')
            H_tool[0][:3, 3] = self._p_tool[j] + self._calib_prms['tool'][j]

        if f is not None:
            H_tool[0][:3, 3] = self._p_force

        H_total = H_base + H_01 + H_12 + H_23 + H_34 + H_45 + H_56 + H_tool

        if k is None:
            H_robot = multi_dot(H_total)
        elif k == 1:
            H_robot = H_total[0]
        else:
            H_robot = multi_dot(H_total[:k])

        return (H_robot[0:3, 3] + delta_t), H_robot[0:3, 0:3]
