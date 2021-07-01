import numpy as np
from numpy.linalg import multi_dot
import math
import quaternion

class Kinematics():
	"""
			Kinematics of the UR10 arm 
		:param delta: (np.ndarray) the delta calibration parameters of the UR10 arm
		:param p_x, p_y, p_z: (np.ndarray) the linear calibration parameters of the UR10 arm
		:param phi_x, phi_y, phi_z: (np.ndarray) the angular calibration parameters of the UR10 arm
		:param quater: (np.ndarray) the quaternion parameters for the base orientation of the UR10 arm
	"""

	def __init__(
		self, quater = np.zeros(4), delta = np.zeros(5), p_x = np.zeros(3), p_y = np.zeros(4), p_z = np.zeros(4), 
		phi_x = np.zeros(1), phi_y = np.zeros(6), phi_z = np.zeros(3)
	):  
		
		# DH  = [a, alpha, d, theta] --- DH parameters of the UR10 arm
		self.DH = np.array(
			[[0,    -math.pi/2, 128,   0], 
			[612.7, 0,          0,     0], 
			[571.6, 0,          0,     0], 
			[0,     -math.pi/2, 163.9, 0], 
			[0,     math.pi/2,  115.7, 0], 
			[0,     0,          92.2,  0]]
		)
		# calibration parameters   
		self.calib_prms = { 
			'base'  : {'p_x': p_x[0], 'p_y': p_y[0], 'p_z': p_z[0],'phi_x': phi_x[0],'phi_y': phi_y[0],'phi_z': phi_z[0]},
			'joint1': {'p_x': p_x[1], 'p_y': p_y[1], 'phi_y': phi_y[1]}, 
			'joint2': {'delta': delta[0], 'p_z': p_z[1], 'phi_y': phi_y[2], 'phi_z': phi_z[1]}, 
			'joint3': {'delta': delta[1], 'p_z': p_z[2], 'phi_y': phi_y[3], 'phi_z': phi_z[2]}, 
			'joint4': {'delta': delta[2], 'p_y': p_y[2], 'p_z': p_z[3], 'phi_y': phi_y[4]}, 
			'joint5': {'delta': delta[3], 'p_x': p_x[2], 'p_y': p_y[3], 'phi_y': phi_y[5]}, 
			'joint6': {'delta': delta[4]},
		}
		self.quater = np.quaternion(quater[0],quater[1],quater[2],quater[3])

	def R_base(self, q):
		Rb = np.zeros((4,4))
		Rb[:3,:3] = quaternion.as_rotation_matrix(q) 
		Rb[3,3] = 1
		return Rb

		
	def Rx(self, theta):
		return np.array(
			[[1, 0, 0, 0], [0, np.cos(theta), -np.sin(theta), 0], [0, np.sin(theta), np.cos(theta), 0], [0, 0, 0, 1]]
		)	

	def Ry(self, theta):
		return np.array(
			[[np.cos(theta), 0, np.sin(theta), 0], [0, 1, 0, 0], [-np.sin(theta), 0, np.cos(theta), 0], [0, 0, 0, 1]]
		)	

	def Rz(self, theta):
		return np.array(
			[[np.cos(theta), -np.sin(theta), 0, 0], [np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
		)	

	def Tx(self, x):		
		return np.array(
			[[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
		)	

	def Ty(self, y):
		return np.array(
			[[1, 0, 0, 0], [0, 1, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]]
		)	

	def Tz(self, z):
		return np.array(
			[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, z], [0, 0, 0, 1]]
		)	

	def forward_kinematics(self, q = np.array([0,0,0,0,0,0]), quater = False):
		"""
			Comutes the forward kinematics of the UR10 arm robot
			:param q: (np.ndarray) the joint angles
			:return: (np.ndarray) the cartesian position of the end effector
		"""
		d1 = self.DH[0,2]
		a2 = self.DH[1,0]
		a3 = self.DH[2,0]
		d4 = self.DH[3,2]
		d5 = self.DH[4,2]
		d6 = self.DH[5,2]
		
		if quater == True:
			H_base = multi_dot(
				[self.Tx(self.calib_prms['base']['p_x']),
				self.Ty(self.calib_prms['base']['p_y']),
				self.Tz(d1 + self.calib_prms['base']['p_z']),
				self.R_base(self.quater)]
			)
		else:
			H_base = multi_dot(
				[self.Tx(self.calib_prms['base']['p_x']),
				self.Ty(self.calib_prms['base']['p_y']),
				self.Tz(d1 + self.calib_prms['base']['p_z']),
				self.Rx(self.calib_prms['base']['phi_x']),  
				self.Ry(self.calib_prms['base']['phi_y']),
				self.Rz(self.calib_prms['base']['phi_z'])]
			)

		H_01 = multi_dot(
			[self.Rz(q[0]), 
			self.Tx(d4 + d6  + self.calib_prms['joint1']['p_x']), 
			self.Ty(self.calib_prms['joint1']['p_y']), 
			self.Ry(self.calib_prms['joint1']['phi_y'])]
		)

		H_12 = multi_dot(
			[self.Rx(q[1] + self.calib_prms['joint2']['delta']), 
			self.Tz(a2 + self.calib_prms['joint2']['p_z']), 
			self.Ry(self.calib_prms['joint2']['phi_y']), 
			self.Rz(self.calib_prms['joint2']['phi_z'])]
		)

		H_23 = multi_dot(
			[self.Rx(q[2] + self.calib_prms['joint3']['delta']), 
			self.Tz(a3 + self.calib_prms['joint3']['p_z']), 
			self.Ry(self.calib_prms['joint3']['phi_y']), 
			self.Rz(self.calib_prms['joint3']['phi_z'])]
		)

		H_34 = multi_dot(
			[self.Rx(q[3] + self.calib_prms['joint4']['delta']), 
			self.Ty(self.calib_prms['joint4']['p_y']), 
			self.Tz(d5 + self.calib_prms['joint4']['p_z']), 
			self.Ry(self.calib_prms['joint4']['phi_y'])]
		)

		H_45 = multi_dot(
			[self.Rz(q[4] + self.calib_prms['joint5']['delta']), 
			self.Tx(self.calib_prms['joint5']['p_x']), 
			self.Ty(self.calib_prms['joint5']['p_y']), 
			self.Ry(self.calib_prms['joint5']['phi_y'])]
		)

		H_56 = self.Rx(
			q[5] + self.calib_prms['joint6']['delta']
		)

		H_robot = multi_dot(
			[H_base, H_01, H_12, H_23, H_34, H_45, H_56]
		)
		return H_robot[0:3, 3], H_robot[0:3, 0:3]