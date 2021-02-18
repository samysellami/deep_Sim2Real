import numpy as np



class Calibration():
    def __init__(self):
        # super().__init__(visualize=visualize, mode=mode)

        self.p = np.zeros(10)
        self.p_base = np.zeros(3)
        self.R_base = np.zeros((3,3))
        self.p_tool = [np.zeros(3), np.zeros(3), np.zeros(3)]  


if __name__ == "__main__":
	calib = Calibration()
	print(calib.p_tool)