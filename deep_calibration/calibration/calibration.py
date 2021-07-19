import numpy as np
import math

from deep_calibration.utils.kinematics import Kinematics
from deep_calibration.utils.jacobian import Jacobian

class Calibration():
	"""
		Calibratioon of the UR10 arm following the paper "Geometric and elastostatic calibration of robotic manipulator
									using partial pose measurements" 
	"""

	def __init__(self,  
		p_ij = [],
		configs = [
			np.array([-math.pi/8,  math.pi/3,  math.pi/4,  math.pi/5,  math.pi/6, -math.pi/7]), 
			np.array([-math.pi/7,  math.pi/6, -math.pi/5,  math.pi/4,  math.pi/3, -math.pi/8]),
			np.array([ math.pi/5,  math.pi/3,  math.pi/4,  math.pi/8,  math.pi/7,  math.pi/6]),
			np.array([-math.pi/5,  math.pi/6, -math.pi/7,  math.pi/8,  math.pi/3, -math.pi/4]), 
			np.array([-math.pi/3, -math.pi/5, -math.pi/8, -math.pi/4, -math.pi/6, -math.pi/7]),
	    ]
	):
		self._FK = Kinematics()
		self._configs = configs
		self._n = 3
		self._m = len(self._configs)
		self._p_ij = [ 
			np.array(
				[[self.p_robot(i) + (2 * np.random.rand() - 1.) * 0.03], 
				[self.p_robot(i)  + (2 * np.random.rand() - 1.) * 0.03], 
				[self.p_robot(i)  + (2 * np.random.rand() - 1.) * 0.03]]
			)  for i in range(self._m)
		]

	def skew(self, phi):
		return np.array([[0, -phi[2], phi[1]], [phi[2], 0, -phi[0]], [-phi[1], phi[0], 0]])

	def p_robot(self, i):
		return self._FK.forward_kinematics(q = self._configs[i])[0]

	def R_robot(self, i):
		return self._FK.forward_kinematics(q = self._configs[i])[1]

	def delta_p(self, i, j = None):
		if j is None:
			return (self._p_ij[i] - self.p_robot(i)).flatten()
		else:
			return (self._p_ij[i][j] - self.p_robot(i)).flatten()

	def construct_A(self, i):
		eye = np.identity(3)
		A = np.zeros((self._n * 3, self._n * 3 + 6))
		p_i = self.skew(self.p_robot(i))
		R_i = self.R_robot(i)

		for j in range(self._n):
			A[3 * j : 3 * (j+1),                  0 : 3] = eye
			A[3 * j : 3 * (j+1),                  3 : 6] = p_i
			A[3 * j : 3 * (j+1),   3 * (j+2) :3 * (j+3)] = R_i
		return A

	def identity_base_tool(self):
		res1 = 0
		res2 = 0
		for i in range(self._m):
			A_i = self.construct_A(i)
			res1 += np.dot(A_i.transpose(), A_i)
			res2 += np.dot(A_i.transpose(), self.delta_p(i))

		res = np.dot( np.linalg.inv(res1), res2 )
		p_base, phi_base, u_tool1, u_tool2, u_tool3 = res[:3], res[3:6], res[6:9], res[9:12], res[12:]   
		
		R_base = self.skew(phi_base) + np.identity(3)
		p_tool1 = np.dot( R_base.transpose(), u_tool1)
		p_tool2 = np.dot( R_base.transpose(), u_tool2)
		p_tool3 = np.dot( R_base.transpose(), u_tool3)
		
		return p_base, R_base, [p_tool1, p_tool2, p_tool3] 

	def identify_calib_prms(self, p_base, R_base, p_tool):
		jacob = Jacobian(p_base, R_base, p_tool)

		res1 = 0
		res2 = 0
		for i in range(self._m):
			for j in range(self._n):
				J_ij = jacob.build_jacobian(q = self._configs[i], j = j)
				res1 += np.dot(J_ij.transpose(), J_ij)
				res2 += np.dot(J_ij.transpose(), self.delta_p(i, j))
		
		res = np.dot( np.linalg.inv(res1), res2 )
		return res

def main():

	np.set_printoptions(precision=4, suppress=True)
	calib = Calibration()

	# step 1 identification of p_base, phi_base and u_tool
	p_base, R_base, p_tool = calib.identity_base_tool()

	# step 2 identification of the calibration parameters
	calib_prms = calib.identify_calib_prms(p_base, R_base, p_tool)
	print(calib_prms.shape)


if __name__ == "__main__":
    main()


