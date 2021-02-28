import numpy as np
from deep_calibration.kinematics import Kinematics
import math

class Calibration():
    def __init__(self):
        # super().__init__(visualize=visualize, mode=mode)
        self.p = np.zeros(8) # positional residuals
        self.phi = np.zeros(7) # angular residuals
        self.delta = np.zeros(4) # joint residuals
        self.p_base = np.zeros(3)
        self.R_base = np.zeros((3,3))
        self.p_tool = [np.zeros(3), np.zeros(3), np.zeros(3)]  


if __name__ == "__main__":
	
    calib = Calibration()
    DH = np.array([[0, -math.pi/2, 128, 0], [612.7, 0, 0, 0], [571.6, 0, 0, 0], \
        [0, -math.pi/2, 163.9, 0], [0, math.pi/2, 115.7, 0], [0, 0, 92.2, 0]])
    q = np.array([0, math.pi/2, 0, 0, 0, 0])
    
    FK = Kinematics(DH)
    print('forward kinematics: =\n', FK.forward_kinematcis(q))


