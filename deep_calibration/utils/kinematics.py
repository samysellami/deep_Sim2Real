import numpy as np
from numpy.linalg import multi_dot
import math


class Kinematics():
	"""
		Kinematics of the UR10 arm 
		:param delta: ([float]) the initial delta calibration parameters of the UR10 arm
		:param joints: ([float]) the initial joints calibration parameters of the UR10 arm
	"""

	def __init__(self, delta = [0,0,0,0,0], joints = np.zeros((5,3))):

		# DH  = [a, alpha, d, theta] --- DH parameters of the UR10 arm
		self.DH = np.array(
			[[0, -math.pi/2, 128, 0], 
			[612.7, 0, 0, 0], 
			[571.6, 0, 0, 0], 
			[0, -math.pi/2, 163.9, 0], 
			[0, math.pi/2, 115.7, 0], 
			[0, 0, 92.2, 0]]
		)
		# calibration parameters   
		self.calib_prms = { 
			'joint1': {'p_x': joints[0,0], 'p_y': joints[0,1], 'phi_y': joints[0,2]}, 
			'joint2': {'delta': delta[0], 'p_z': joints[1,0], 'phi_y': joints[1,1], 'phi_z': joints[1,2]}, 
			'joint3': {'delta': delta[1], 'p_z': joints[2,0], 'phi_y': joints[2,1], 'phi_z': joints[2,2]}, 
			'joint4': {'delta': delta[2], 'p_y': joints[3,0], 'p_z': joints[3,1], 'phi_y': joints[3,2]}, 
			'joint5': {'delta': delta[3], 'p_x': joints[4,0], 'p_y': joints[4,1], 'phi_y': joints[4,2]}, 
			'joint6': {'delta': delta[4]},
		}
		
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

	def forward_kinematcis(self, q):
		a2 = self.DH[1,0]
		a3 = self.DH[2,0]
		d4 = self.DH[3,2]
		d5 = self.DH[4,2]
		d6 = self.DH[5,2]
		H_01 = multi_dot(
			[self.Rz(q[0]), self.Tx(d4 + d6  + self.calib_prms['joint1']['p_x']), 
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
			[H_01, H_12, H_23, H_34, H_45, H_56]
		)
		return H_robot[0:3, 3]