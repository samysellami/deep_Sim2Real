import numpy as np
import math
import os

from deep_calibration.utils.kinematics import Kinematics
from deep_calibration.utils.jacobian import Jacobian
from numpy import linalg as LA

from deep_calibration import script_dir


class Calibration:
    """
    Calibration of the UR10 arm following the paper "Geometric and elastostatic calibration of robotic manipulator
                                                            using partial pose measurements"
    """

    def __init__(
        self,
        p_ij=[],
        configs=[
            np.array([0, 0, 0, 0, 0, 0]),
            np.array([-math.pi / 2, math.pi, -math.pi / 2, math.pi, math.pi / 2, -math.pi]),
            np.array([math.pi, math.pi / 2, math.pi, math.pi / 2, math.pi, math.pi / 2]),
            np.array([-math.pi / 2, math.pi / 2, -math.pi, math.pi, math.pi / 2, -math.pi / 2]),
            np.array([-math.pi / 2, -math.pi, -math.pi / 2, -math.pi / 2, -math.pi, -math.pi / 2]),
            np.array([-math.pi, -math.pi, -math.pi / 2, -math.pi, -math.pi / 2, -math.pi / 2]),
            np.array([math.pi / 2, math.pi / 2, math.pi / 2, math.pi / 2, math.pi / 2, math.pi / 2]),
            np.array([-math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2]),
            np.array([math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2]),
            np.array([-math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2]),
        ]
    ):
        self._p_tool = None
        self._p_base = None
        self._R_base = None
        self._delta = np.zeros(5)

        self._n = 3  # number of tools used for calibration
        # self._configs = self.setup_configs()  # robot configurations used for calibration
        self._configs = configs  # robot configurations used for calibration
        self._c = len(self._configs)  # number of robot configurations
        self._m = self._c  # number of measurements configurations
        self._noise_std = 0.00 * 0.001

        self._FK = Kinematics(
            delta=np.array([0.001, -0.002, 0.003, -0.002, 0.001])
        )
        self._goal_pos = self.build_goal_pos()
        self._p_ij = self.build_p_ij()
        self._FK = Kinematics(p_base=self._p_base, R_base=self._R_base, p_tool=self._p_tool, delta=self._delta)

    def noise(self):
        return (2 * np.random.rand() - 1.0) * self._noise_std

    def build_goal_pos(self):
        return [
            np.array(
                self.p_robot(i)
            )
            for i in range(self._c)
        ]  # goal position

    def build_p_ij(self):
        return [
            np.array(
                [
                    self.p_robot(i, j=0) + self.noise(),
                    self.p_robot(i, j=1) + self.noise(),
                    self.p_robot(i, j=2) + self.noise(),
                ]
            )
            for i in range(self._m)
        ]  # partial pose measurements

    def setup_configs(self):
        configs = []
        angle_limit = math.pi
        self._c = 10

        for m in range(self._c):
            q = np.array([
                (2 * np.random.rand() - 1.) * angle_limit,
                (2 * np.random.rand() - 1.) * angle_limit,
                (2 * np.random.rand() - 1.) * angle_limit,
                (2 * np.random.rand() - 1.) * angle_limit,
                (2 * np.random.rand() - 1.) * angle_limit,
                (2 * np.random.rand() - 1.) * angle_limit
            ])
            configs.append(q)
        return configs

    def config(self, i):
        return self._configs[i % self._c]

    def skew(self, phi):
        """
            Computes the skew symetric matrix of a certain set of angles
                :param qhi: (np.ndarray) the angles
                :return: (np.ndarray) the skew symetric matrix
        """
        return np.array([[0, -phi[2], phi[1]], [phi[2], 0, -phi[0]], [-phi[1], phi[0], 0]])

    def p_robot(self, i=0, j=None):
        if j is not None:
            return self._FK.forward_kinematics(q=self.config(i), j=j)[0]
        return self._FK.forward_kinematics(q=self.config(i))[0]

    def R_robot(self, i=0):
        return self._FK.forward_kinematics(q=self.config(i))[1]

    def delta_p(self, i=0, j=None, goal=None):
        if j is None:
            if goal is not None:
                return (self._goal_pos[i] - self.p_robot(i=i)).flatten()
            return (self._p_ij[i] - self.p_robot(i=i)).flatten()
        else:
            return (self._p_ij[i][j] - self.p_robot(i=i, j=j)).flatten()

    def construct_A(self, i):
        eye = np.identity(3)
        A = np.zeros((self._n * 3, self._n * 3 + 6))
        p_i = self.skew(self.p_robot(i=i))
        R_i = self.R_robot(i=i)

        for j in range(self._n):
            A[3 * j: 3 * (j + 1), 0:3] = eye
            A[3 * j: 3 * (j + 1), 3:6] = p_i
            A[3 * j: 3 * (j + 1), 3 * (j + 2): 3 * (j + 3)] = R_i
        return A

    def dist_to_goal(self):
        self._FK = Kinematics(p_base=self._p_base, R_base=self._R_base, p_tool=self._p_tool, delta=self._delta)
        dists_goal = []
        for i in range(self._m):
            dist_goal = np.mean(np.abs((np.array(self.delta_p(i=i, goal=1)))))
            dists_goal.append(dist_goal)

        return np.mean(dists_goal)

    def dist_to_goal_j(self):
        dists_goal = []
        for i in range(self._m):
            dist_goal = []
            for j in range(self._n):
                dist_goal.append(self.delta_p(i=i, j=j))
            dist_goal = np.mean(np.abs((np.array(dist_goal))))
            # dist_goal = LA.norm((np.array(dist_goal)))
            dists_goal.append(dist_goal)

        return np.mean(dists_goal)

    def identity_base_tool(self):
        """
            Identify the tool and base parameters
                :return: (np.ndarray) the base positional and rotoational parameters and the tool parameters
        """
        self._FK = Kinematics(p_base=np.zeros(3), R_base=np.identity(3))

        res1 = 0
        res2 = 0
        for i in range(self._m):
            A_i = self.construct_A(i)
            res1 += np.dot(A_i.transpose(), A_i)
            res2 += np.dot(A_i.transpose(), self.delta_p(i=i))

        res = np.dot(np.linalg.inv(res1), res2)
        p_base, phi_base, u_tool1, u_tool2, u_tool3 = res[:
                                                          3], res[3:6], res[6:9], res[9:12], res[12:]

        R_base = self.skew(phi_base) + np.identity(3)
        p_tool1 = np.dot(R_base.transpose(), u_tool1)
        p_tool2 = np.dot(R_base.transpose(), u_tool2)
        p_tool3 = np.dot(R_base.transpose(), u_tool3)

        return p_base, R_base, [p_tool1, p_tool2, p_tool3]

    def identify_calib_prms(self):
        """
            Identify the calibration parameters
                :param p_base, R_base, p_tool: (np.ndarray) base and tool parameters identified in the 1st step
                :return: (np.ndarray) the calibration parameters
        """
        self._FK = Kinematics(p_base=self._p_base, R_base=self._R_base, p_tool=self._p_tool, delta=self._delta)
        jacob = Jacobian(p_base=self._p_base, R_base=self._R_base, p_tool=self._p_tool, delta=self._delta)

        res1 = 0
        res2 = 0
        A = np.zeros(5)
        b = np.zeros(1)

        for i in range(self._m):
            for j in range(self._n):
                J_ij = jacob.build_jacobian(q=self.config(i), j=j)
                res1 += np.dot(J_ij.transpose(), J_ij)
                res2 += np.dot(J_ij.transpose(), self.delta_p(i=i, j=j))

                A = np.vstack((A, J_ij))
                b = np.hstack((b, self.delta_p(i=i, j=j)))

        # print(LA.cond(res1, np.inf))
        # calib_prms = np.dot(np.linalg.inv(res1), res2)
        # accuracy = np.linalg.inv(res1) * self._noise_std * 1000

        A = np.delete(A, 0, 0)
        b = np.delete(b, 0, 0)
        calib_prms = np.dot(np.linalg.pinv(A), b)
        return calib_prms


def main():

    np.set_printoptions(precision=7, suppress=True)
    calib = Calibration()
    print('distance to goal: ', calib.dist_to_goal() * 1000)

    # step 1 identification of p_base, phi_base and u_tool
    p_base, R_base, p_tool = calib.identity_base_tool()
    print('p_base:\n', p_base, ' \n R_base:\n', R_base, '\n p_tool:\n', p_tool)

    # step 2 identification of the calibration parameters
    calib._p_base = p_base
    calib._R_base = R_base
    calib._p_tool = p_tool

    for i in range(5):
        calib_prms = calib.identify_calib_prms()
        calib._delta += calib_prms

        print('delta_calib_prms:', calib_prms)
        print('distance to goal: ', calib.dist_to_goal() * 1000)

    print('calib_prms:', calib._delta)

    with open(f"{script_dir}/calibration/p_ij.npy", 'wb') as f:
        f.truncate(0)
        np.save(f, {
            'p_ij': calib._p_ij,
            'p_base': calib._p_base,
            'R_base': calib._R_base,
            'p_tool': calib._p_tool,
            'calib_prms': calib._delta,
            'goal_position': calib._goal_pos,
        }, allow_pickle=True)
    f.close()


if __name__ == "__main__":
    main()
