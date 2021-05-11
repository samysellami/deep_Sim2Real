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

  def __init__(
    self, config = [
      np.array([0 ,0 ,0 ,0 ,0 ,0]), 
      np.array([0, math.pi/2, 0, math.pi/2, 0, math.pi/2]), 
      np.array([math.pi/2, 0, math.pi/2, 0, math.pi/2, 0]),
      np.array([0, math.pi/3, 0, math.pi/3, 0, math.pi/3]),
      np.array([math.pi/3, 0, math.pi/3, 0, math.pi/3, 0]), 
      # np.array([0, math.pi/4, 0, math.pi/4, 0, math.pi/4]),
      # np.array([math.pi/4, 0, math.pi/4, 0, math.pi/4, 0]), 
      # np.array([0, math.pi/5, 0, math.pi/5, 0, math.pi/5]),
      # np.array([math.pi/5, 0, math.pi/5, 0, math.pi/5, 0]),
      # np.array([math.pi/6, 0, math.pi/6, 0, math.pi/6, 0]),
      # np.array([0, math.pi/6, 0, math.pi/6, 0, math.pi/6]),
      # np.array([math.pi/7, 0, math.pi/7, 0, math.pi/7, 0]),
      # np.array([0, math.pi/7, 0, math.pi/7, 0, math.pi/7]), 
      # np.array([math.pi/8, 0, math.pi/8, 0, math.pi/8, 0]),
      # np.array([0, math.pi/8, 0, math.pi/8, 0, math.pi/8]), 
    ],  
    delta = np.array([0.001, -0.001, 0.001, -0.001, 0.001]), 
    p_x = np.array([0.2, -0.2, 0.2]), p_y = np.array([-0.3, -0.2, 0.2, -0.2]), p_z = np.array([-0.1, -0.2, 0.2, -0.2]), 
    phi_x = np.array([0.02]), phi_y = np.array([0.02, -0.02, 0.02, -0.02, 0.02, -0.02]), 
    phi_z = np.array([0.02, -0.02, 0.02])
  ):
    

    self.pos = 0.500
    self.ori = 0.050
    # action encodes the calibration parameters (positional and rotational)
    # self._action_space = spaces.Dict({
    #   'position'   : gym.spaces.Box(low = -0.5, high = 0.5, shape=(11,), dtype='float32'),
    #   'orientation': gym.spaces.Box(low = -0.03, high = 0.03, shape=(15,), dtype='float32')
    # })
    # self.action_space = spaces.Box(
    #   np.array(
    #     [-self.pos, -self.pos, -self.pos, -self.pos, -self.pos, -self.pos, -self.pos, -self.pos, -self.pos, -self.pos, -self.pos,
    #     -self.ori, -self.ori, -self.ori, -self.ori, -self.ori, -self.ori, -self.ori, -self.ori, -self.ori, -self.ori, -self.ori, 
    #     -self.ori, -self.ori, -self.ori, -self.ori]
    #   ),
    #   np.array(
    #     [self.pos, self.pos, self.pos, self.pos, self.pos, self.pos, self.pos, self.pos, self.pos, self.pos, self.pos,
    #     self.ori, self.ori, self.ori, self.ori, self.ori, self.ori, self.ori, self.ori, self.ori, self.ori, self.ori, 
    #     self.ori, self.ori, self.ori, self.ori]
    #   ),
    #   dtype='float32'
    # )

    self.action_space = spaces.Box(
      np.array(
        [-self.pos, -self.pos, -self.pos]
      ),
      np.array(
        [self.pos, self.pos, self.pos]
      ),
      dtype='float32'
    )

    # the observation encodes the x, y, z position of the end effector and the joint angles
    self.observation_space = spaces.Box(
      np.array(
        [-2*math.pi, -2*math.pi, -2*math.pi, -2*math.pi, -2*math.pi, -2*math.pi]
      ),
      np.array(
        [ 2*math.pi, 2*math.pi, 2*math.pi, 2*math.pi, 2*math.pi, 2*math.pi]
      ), 
      dtype='float32'
    )
    self._configs = config
    # self._config = []
    self._config = self._configs
    self._default_action = self.get_default_action(
        delta = delta, p_x = p_x, p_y = p_y, p_z = p_z, 
        phi_x = phi_x, phi_y = phi_y, phi_z = phi_z
    )
    self._q = config[0]
    self._i = -1
    self.rand = 0
    self._delta = delta
    self._p_x = p_x; self._p_y = p_y; self._p_z = p_z 
    self._phi_x = phi_x; self._phi_y = phi_y; self._phi_z = phi_z    
    self._count = 0 # total time_steps
    self._reset = 0 # reset the environment from the initial joint position
    self._goal_pos = self.get_position()
    self._prev_distance = None

  @property
  def config(self):
      return self._config

# ---------------------------- Gym specific methods  -----------------------------------

  def step(self, action):
    # print('--------Environment step--------')

    if self._prev_distance is None:
      self._prev_distance = self.distance_to_goal(action)

    observation = self.get_observation(action)
    reward = self.compute_reward(action) 
    self._prev_distance = self.distance_to_goal(action)
    
    done = self.compute_done(action)
    info = {}
    return observation, reward, done, {}




  def reset(self):
    # print('--------Episode reset--------')
    # self._count = 0
    j = len(self._config)
    if self._count % 1000000 == 0 and j < len(self._configs):
      print('--------changing configs --------')
      self._config.append(self._configs[j])


    self._prev_distance  = None
    if self._reset == 0:
      self.setup_joints()  
      self._goal_pos = self.get_position()

    observation = self.get_observation()
    return observation

  def render(self, mode='human'):
    ...
  def close(self):
    ...

# ----------------------------  env-specific methods -----------------------------------

  def setup_joints(self):
    """
      pertubate the joint angles for the reset function
    """
    self._i = (self._i + 1)  % len(self._config)
    if self.rand == 0:
      self._q = self._config[self._i]
    else:
      step_limit = math.pi/50
      self._q = np.array([
          self._q[0] + (2 * np.random.rand() - 1.) * step_limit,
          self._q[1] + (2 * np.random.rand() - 1.) * step_limit,
          self._q[2] + (2 * np.random.rand() - 1.) * step_limit,
          self._q[3] + (2 * np.random.rand() - 1.) * step_limit,
          self._q[4] + (2 * np.random.rand() - 1.) * step_limit,
          self._q[5] + (2 * np.random.rand() - 1.) * step_limit
      ])

  def get_default_action(self, delta, p_x, p_y, p_z, 
                        phi_x, phi_y, phi_z):
    action = np.zeros(3)
    action[0] = p_x[0]
    action[1] = p_y[0]
    action[2] = p_z[0]

    # action = np.zeros(26)    
    # action[0:3] = p_x  
    # action[3:7] = p_y   
    # action[7:11] = p_z   
    # action[11:16] = delta 
    # action[16:17] = phi_x 
    # action[17:23] = phi_y  
    # action[23:] = phi_z  
    return action


  def get_position(self, action = None):
    """
      Return the end effector position
      :param action: (np.ndarray) the calibration parameters 
      :return: (np.ndarray) the position of the end effector
    """
    if action is None:
      action = self._default_action

    self._p_x[0] = action[0]
    self._p_y[0] = action[1] 
    self._p_z[0] = action[2] 

    # self._p_x = action[0:3]
    # self._p_y = action[3:7] 
    # self._p_z = action[7:11] 
    # self._delta = action[11:16]
    # self._phi_x = action[16:17] 
    # self._phi_y = action[17:23] 
    # self._phi_z = action[23:] 
    
    FK = Kinematics(
      delta = self._delta, p_x = self._p_x, p_y = self._p_y, p_z = self._p_z, 
      phi_x = self._phi_x, phi_y = self._phi_y, phi_z = self._phi_z
    )
    return FK.forward_kinematcis(self._q)

  def get_observation(self, action = None):
    """
      Return the environment observation
      :param action: (np.ndarray) the calibration parameters 
      :return: (np.ndarray) the environment observation
    """
    if action is None:
      action = self._default_action

    pos = self.get_position(action)
    return self._q
    # return np.hstack((pos,self._q)) 
  

  def distance_to_goal(self, action):
    """
      Compute the distance to the goal
      :param action: (np.ndarray) the calibration parameters 
    """
    return LA.norm(self.get_position(action) - self._goal_pos)

  def compute_reward(self, action):
    """
      Compute the reward value for the step function
      :param action: (np.ndarray) the calibration parameters 
    """
    if self._prev_distance is None:
      self._prev_distance = self.distance_to_goal(action)
    
    dist_goal = self.distance_to_goal(action)
    reward = (self._prev_distance - dist_goal)/self._prev_distance + (10/dist_goal)
    if math.isnan(reward):
      reward = 1000
    return reward
  
  def compute_done(self, action):
    """
      Compute the done boolean for the step function
      :param reward: (float) the reward of the given step 
      :return: (float) the done flag
    """
    self._count += 1
    done = False   
    if self.distance_to_goal(action) > 300:
      print('--------Reset: Divergence--------')
      done = True
    elif self.distance_to_goal(action) < 0.00001:
      print('--------Reset: Convergence--------')
      done = True
    elif self._count % 1000 == 0:  
      print('--------Reset: Timeout--------')
      done = True

    return done