import numpy as np


class Jacobian():
    """
            Jacobian of the UR10 robot calibration
    """

    def __init__(self, FK,
                 prms_J={"joint2": "delta_x",
                         "joint3": "delta_x",
                         "joint4": "delta_x",
                         "joint5": "delta_z"}):

        self._FK = FK
        self._prms_J = prms_J
        self.len_prms = len(prms_J)
        self._RATIO = 1  # conversion ratio to meters
        self._x = np.array([1, 0, 0])
        self._y = np.array([0, 1, 0])
        self._z = np.array([0, 0, 1])
        self._prms = self._FK._calib_prms.copy()
        del self._prms["base"]
        del self._prms["tool"]
        del self._prms["ksi"]

    def jacobian(self, O_i, O_n, T_i, param):
        if param == "p_x":
            J = np.dot(T_i, self._x)
        elif param == "p_y":
            J = np.dot(T_i, self._y)
        elif param == "p_z":
            J = np.dot(T_i, self._z)
        elif param == "phi_x" or param == "delta_x":
            J = np.cross(np.dot(T_i, self._x), O_n - O_i)
        elif param == "phi_y":
            J = np.cross(np.dot(T_i, self._y), O_n - O_i)
        elif param == "phi_z" or param == "delta_z":
            J = np.cross(np.dot(T_i, self._z), O_n - O_i)
        return J

    def build_jacobian(self, q=np.array([0, 0, 0, 0, 0, 0]), j=None, f=None):
        Jac = np.zeros((3, self.len_prms))
        k = 0
        l = 0
        for (joint, params) in self._prms.items():
            for (param, _) in params.items():
                if param == self._prms_J.get(joint):
                    O_i = self._FK.forward_kinematics(q=q, k=k)[0]
                    O_n = self._FK.forward_kinematics(q=q, j=j, f=f)[0]
                    T_i = self._FK.forward_kinematics(q=q, k=k)[1]

                    Jac[:, l] = self.jacobian(O_i=O_i, O_n=O_n, T_i=T_i, param=param)
                    l += 1
                k += 1
        return Jac
