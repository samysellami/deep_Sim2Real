from deep_calibration.utils.kinematics import Kinematics
import numpy as np


class Jacobian(Kinematics):
	"""
		Jacobian of the UR10 robot calibration
	"""

	def __init__(self, p_base = None, R_base = None, p_tool = None):
		super().__init__(p_base = p_base, R_base = R_base, p_tool = p_tool)

		self._x = np.array([1, 0, 0])
		self._y = np.array([0, 1, 0])
		self._z = np.array([0, 0, 1])
		self._prms = self.calib_prms.copy()
		del self._prms["base"]

	def jacobian(self, O_i, O_n, param):
		if param == "p_x":
			J = self._x
		elif param == "p_y":
			J = self._y
		elif param == "p_z":
			J = self._z
		elif param == "phi_x" or param == "delta_x":
			J = np.cross(self._x, O_n - O_i)
		elif param == "phi_y":
			J = np.cross(self._y, O_n - O_i)
		elif param == "phi_z" or param == "delta_z":
			J = np.cross(self._z, O_n - O_i)
		return J

	def build_jacobian(self, q = np.array([0, 0, 0, 0, 0, 0]), j = None):
		Jac = np.zeros((3, 5))
		k = 0
		l = 0
		for (_, joint) in self._prms.items():
			for (param, _) in joint.items():
				if param == 'delta_x' or  param == 'delta_z':
					O_i = self.forward_kinematics(q = q, k = k + 2)[0]
					O_n = self.forward_kinematics(q = q, j = j)[0]
					Jac[:, l] = self.jacobian(O_i = O_i, O_n = O_n, param = param)
					l += 1
				k += 1
		return Jac
