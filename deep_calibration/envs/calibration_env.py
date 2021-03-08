import numpy as np
import logging
from numpy import linalg as LA
import math

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import Space

from deep_calibration.utils.kinematics import Kinematics


class CalibrationEnv(gym.Env): 
  """
    Gym environment for the deep calibration
    :param q: (np.ndarray) the initial joint position of the UR10 arm
  """
  metadata = {'render.modes': ['human']}

  def __init__(self, q = np.array([0,0,0,0,0,0])):
    
    # action encodes the calibration parameters
    self._action_space = spaces.Box(-0.1, 0.1, shape=(26,), dtype='float32')
    
    # observation encodes the x, y, z position of the end effector
    self._observation_space = spaces.Box(
      np.array([-1500, -1500, 0, -2*math.pi, -2*math.pi, -2*math.pi, -2*math.pi, -2*math.pi, -2*math.pi]),
      np.array([1500, 1500, 1500, 2*math.pi, 2*math.pi, 2*math.pi, 2*math.pi, 2*math.pi, 2*math.pi]), 
      dtype='float32'
    )
    self._q = q
    self._delta = np.zeros((1,5))
    self._joints = np.zeros((5,3))
    self._base = np.zeros(6)
    self._goal = self.get_position()
    self._count = 0 # total time_steps
    self._reset = 1 # reset the environment from the initial joint position

  @property
  def observation_space(self) -> Space:
      return self._observation_space

  @property
  def action_space(self) -> Space:
      return self._action_space

  def step(self, action):
    observation = self.get_observation(action)
    reward = - LA.norm(self.get_position(action) - self._goal)
    done = self.compute_done(reward)
    info = {}

    return observation, reward, done, {}

  def reset(self):
    logging.info("Episode reset...")
    self.count = 0
    if self._reset == 0:
      self.setup_joints()  
    observation = self.get_observation()
    
    return observation

  def render(self, mode='human'):
    ...
  def close(self):
    ...

# -------------- all the methods above are required for any Gym environment, everything below is env-specific --------------

  def get_position(self, action = np.zeros(26)):
    """
      Return the end effector position
      :param action: (np.ndarray) the calibration parameters 
      :return: (np.ndarray) the position of the end effector
    """
    self._delta = action[0:5]
    self._joints[0,:] = action[5:8]
    self._joints[1,:] = action[8:11]
    self._joints[2,:] = action[11:14]
    self._joints[3,:] = action[14:17]
    self._joints[4,:] = action[17:20]
    self._base = action[20:]

    FK = Kinematics(self._delta, self._joints, self._base)
    return FK.forward_kinematcis(self._q)

  def get_observation(self, action = np.zeros(26)):
    pos = self.get_position(action)
    return np.hstack((pos,self._q)) 
  
  def setup_joints(self):
    step_limit = math.pi/10
    self._q = np.array(
        [self._q[0] + (2 * np.random.rand() - 1.) * step_limit,
        self._q[1] + (2 * np.random.rand() - 1.) * step_limit,
        self._q[2] + (2 * np.random.rand() - 1.) * step_limit,
        self._q[3] + (2 * np.random.rand() - 1.) * step_limit,
        self._q[4] + (2 * np.random.rand() - 1.) * step_limit,
        self._q[5] + (2 * np.random.rand() - 1.) * step_limit]
    )

  def compute_done(self, reward):
    self._count = self._count + 1
    done = False 
    
    if self._count == 100000:
      logging.info('--------Reset: Timeout--------')
      done = True
    # elif -reward <0.001:
    #   logging.info('--------Reset: Convergence--------')
    #   done = True
    elif -reward > 100:
      logging.info('--------Reset: Divergence--------')
      done = True
    
    return done