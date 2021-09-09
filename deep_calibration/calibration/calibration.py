import numpy as np
import math

from deep_calibration.utils.kinematics import Kinematics
from deep_calibration.utils.jacobian import Jacobian

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
            np.array([-math.pi / 8, math.pi / 3, math.pi / 4, math.pi / 5, math.pi / 6, -math.pi / 7]),
            np.array([-math.pi / 7, math.pi / 6, -math.pi / 5, math.pi / 4, math.pi / 3, -math.pi / 8]),
            np.array([math.pi / 5, math.pi / 3, math.pi / 4, math.pi / 8, math.pi / 7, math.pi / 6]),
            np.array([-math.pi / 5, math.pi / 6, -math.pi / 7, math.pi / 8, math.pi / 3, -math.pi / 4]),
            np.array([-math.pi / 3, -math.pi / 5, -math.pi / 8, -math.pi / 4, -math.pi / 6, -math.pi / 7]),
            np.array([-math.pi / 4, -math.pi / 5, -math.pi / 1, -math.pi / 2, -math.pi / 5, -math.pi / 7]),
            np.array([-math.pi / 4, -math.pi / 5, -math.pi / 3, -math.pi / 4, -math.pi / 2, -math.pi / 9]),
            np.array([-math.pi / 8, -math.pi / 2, -math.pi / 4, -math.pi / 2, -math.pi / 6, -math.pi / 7]),
            np.array([-math.pi / 7, -math.pi / 5, -math.pi / 6, -math.pi / 1, -math.pi / 2, -math.pi / 7]),
            np.array([-math.pi / 3, -math.pi / 5, -math.pi / 4, -math.pi / 10, -math.pi / 2, -math.pi / 8]),
        ],

        p_tool=[
            np.array([277.23, -46.53, -93.87]),
            np.array([276.49, -48.25, 94.05]),
            np.array([278.44, 103.73, -2.17]),
        ],
    ):
        self._configs = configs  # robot configurations used for calibration
        self._n = 3  # number of tools used for calibration
        self._c = len(self._configs)  # number of robot configurations
        self._m = self._c * 10
        self._FK = Kinematics(p_tool=p_tool)
        self._p_ij = self.build_p_ij()
        self._FK = Kinematics(p_base=np.zeros(3), R_base=np.identity(3))

    def build_p_ij(self):
        return [
            np.array(
                [
                    self.p_robot(i, j=0) + (2 * np.random.rand() - 1.0) * 0.3,
                    self.p_robot(i, j=1) + (2 * np.random.rand() - 1.0) * 0.3,
                    self.p_robot(i, j=2) + (2 * np.random.rand() - 1.0) * 0.3,
                ]
            )
            for i in range(self._m)
        ]  # partial pose measurements

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

    def delta_p(self, i=0, j=None):
        if j is None:
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

    def identity_base_tool(self):
        """
        Identify the tool and base parameters
                :return: (np.ndarray) the base positional and rotoational parameters and the tool parameters
        """
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

    def identify_calib_prms(self, p_base, R_base, p_tool):
        """
        Identify the calibration parameters
                :param p_base, R_base, p_tool: (np.ndarray) base and tool parameters identified in the 1st step
                :return: (np.ndarray) the calibration parameters
        """
        jacob = Jacobian(p_base=p_base, R_base=R_base, p_tool=p_tool)
        res1 = 0
        res2 = 0
        for i in range(self._m):
            for j in range(self._n):
                J_ij = jacob.build_jacobian(q=self.config(i), j=j)
                res1 += np.dot(J_ij.transpose(), J_ij)
                res2 += np.dot(J_ij.transpose(), self.delta_p(i=i, j=j))
                # print('jacobian=\n', J_ij)

        res = np.dot(np.linalg.inv(res1), res2)
        return res


def main():

    np.set_printoptions(precision=7, suppress=True)
    calib = Calibration()

    # step 1 identification of p_base, phi_base and u_tool
    p_base, R_base, p_tool = calib.identity_base_tool()
    print('calibration parameters:', p_base, R_base, p_tool)

    # step 2 identification of the calibration parameters
    calib._FK = Kinematics(p_base=p_base, R_base=R_base, p_tool=p_tool)
    calib_prms = calib.identify_calib_prms(p_base=p_base, R_base=R_base, p_tool=p_tool)
    print('calib_prms:', calib_prms)

    with open(f"{script_dir}/calibration/p_ij.npy", 'wb') as f:
        np.save(f, {
            'p_ij': calib._p_ij,
            'p_base': p_base,
            'R_base': R_base,
            'p_tool': p_tool,
            'calib_prms': calib_prms,
        }, allow_pickle=True)


if __name__ == "__main__":
    main()
