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
      # np.array([0 ,0 ,0 ,0 ,0 ,0]), 
      np.array([-math.pi/8, math.pi/3, math.pi/4, math.pi/5, math.pi/6, -math.pi/7]), 
      np.array([-math.pi/7, math.pi/6, -math.pi/5, math.pi/4, math.pi/3, -math.pi/8]),
      np.array([math.pi/5, math.pi/3, math.pi/4, math.pi/8, math.pi/7, math.pi/6]),
      np.array([-math.pi/5, math.pi/6, -math.pi/7, math.pi/8, math.pi/3, -math.pi/4]), 
      np.array([-math.pi/3, -math.pi/5, -math.pi/8, -math.pi/4, -math.pi/6, -math.pi/7]),
    ],  
    quater = np.array([0.9998, 0.0100, 0.0098, 0.0100]),
    delta = np.array([0.001, -0.001, 0.001, -0.001, 0.001]), 
    p_x = np.array([0.2, -0.2, 0.2]), p_y = np.array([-0.1, -0.2, 0.2, -0.2]), p_z = np.array([0.2, -0.2, 0.2, -0.2]), 
    phi_x = np.array([0.02]), phi_y = np.array([0.02, -0.02, 0.02, -0.02, 0.02, -0.02]), phi_z = np.array([0.02, -0.02, 0.02])
  ):
    # action encodes the calibration parameters (positional and rotational)
    # self._action_space = spaces.Dict({
    #   'position'   : gym.spaces.Box(low = -0.5, high = 0.5, shape=(11,), dtype='float32'),
    #   'orientation': gym.spaces.Box(low = -0.03, high = 0.03, shape=(15,), dtype='float32')
    # })
    self.pos = 0.300
    self.ori = 0.030
    self.action_space = spaces.Box(
      np.array(
        [0.1, -0.2, 0.1, 0.98, -0.012, -0.012, -0.012]
      ),
      np.array(
        [0.3, 0.0, 0.3, 1, 0.012, 0.012, 0.012]
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
    self._config = config
    self._n_config = len(self._config)
    self._i = -1
    # self._config = self.setup_configs()
    self._default_action = self.get_default_action(
        quater = quater, delta = delta, p_x = p_x, p_y = p_y, p_z = p_z, 
        phi_x = phi_x, phi_y = phi_y, phi_z = phi_z
    )
    self._q = config[0]
    self.rand = 0
    self._quater = quater; self._delta = delta
    self._p_x = p_x; self._p_y = p_y; self._p_z = p_z 
    self._phi_x = phi_x; self._phi_y = phi_y; self._phi_z = phi_z    
    self._count = 0 # total time_steps
    self._goal_pos = self.get_position()
    self._prev_distance = None
    self._best_action = np.zeros((self._n_config, self.action_space.shape[0] + 1))
    self.update_best_action(init = True)
    self.all_config = True
  
  @property
  def config(self):
      return self._config

# ---------------------------- Gym specific methods  -----------------------------------

  def step(self, action):
    # print('--------Environment step--------')

    observation = self.get_observation(action)
    reward = self.compute_reward(action) 
    self._prev_distance = self.distance_to_goal(action)

    done = self.compute_done(action)
    info = {}
    return observation, reward, done, {}


  def reset(self):
    # print('--------Episode reset--------')

    j = len(self._config)
    if self._count % 10000000 == 0 and j < len(self._configs):
      print('--------changing configs --------')
      self._config.append(self._configs[j])

    # if self._count % 10000 == 0:
    #   print(f'--------best actions --------: {self._best_action}')

    self._prev_distance  = None
    self.setup_joints()  

    observation = self.get_observation()
    return observation

  def render(self, mode='human'):
    ...
  def close(self):
    ...

# ----------------------------  env-specific methods -----------------------------------

  def update_best_action(self, action = None, init = False):    
    if init == True:
      for k in range(self._best_action.shape[0]):
        self._best_action[k, :-1] = self.action_space.sample()
        self._best_action[k, -1] = self.distance_to_goal(self._best_action[k, :-1])
    else:  
      if self.distance_to_goal(action) < self._best_action[self._i, -1]:
        self._best_action[self._i, :-1] = action
        self._best_action[self._i, -1] = self.distance_to_goal(action)



  def setup_configs(self):
    configs = []
    self._q = np.zeros(6)
    self.rand = 1

    for m in range(self._n_config):
      self.setup_joints()
      configs.append(self._q)
    configs = np.array(configs)

    return configs



  def setup_joints(self):
    """
      pertubate the joint angles for the reset function and computes the goal position
    """
    if self.rand == 0:
      self._i = (self._i + 1)  % len(self._config)
      self._q = self._config[self._i]
    else:
      step_limit = math.pi/2
      self._q = np.array([
          self._q[0] + (2 * np.random.rand() - 1.) * step_limit,
          self._q[1] + (2 * np.random.rand() - 1.) * step_limit,
          self._q[2] + (2 * np.random.rand() - 1.) * step_limit,
          self._q[3] + (2 * np.random.rand() - 1.) * step_limit,
          self._q[4] + (2 * np.random.rand() - 1.) * step_limit,
          self._q[5] + (2 * np.random.rand() - 1.) * step_limit
      ])
    self._goal_pos = self.get_position()


  def get_default_action(self, quater, delta, p_x, p_y, p_z, 
                        phi_x, phi_y, phi_z):
    action = np.zeros(7)
    action[0] = p_x[0]
    action[1] = p_y[0]
    action[2] = p_z[0]
    action[3] = quater[0]
    action[4] = quater[1]
    action[5] = quater[2]
    action[6] = quater[3]

    return action


  def get_position(self, action = None, noise = False):
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
    self._quater =  action[3:]
    
    FK = Kinematics( quater = self._quater,
      delta = self._delta, p_x = self._p_x, p_y = self._p_y, p_z = self._p_z, 
      phi_x = self._phi_x, phi_y = self._phi_y, phi_z = self._phi_z
    )
    
    noise = (2 * np.random.rand() - 1.) * 0.03
    pos = FK.forward_kinematics(self._q, quater = True)[0]
    if noise == True:
       pos = pos + noise
    return pos

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

    if self.all_config== True:
      dists_goal = []
      for l in range(self._n_config):
        dists_goal.append(self.distance_to_goal(action))
        self.setup_joints()
      dist_goal = np.mean(np.array(dists_goal))
    else:
      dist_goal = self.distance_to_goal(action)
    
    actions_diff = LA.norm(self._best_action[self._best_action[:,-1] < dist_goal, :-1] - action)
    actions_diff = 0
    if self._prev_distance is None:
      self._prev_distance = self.distance_to_goal(action)
      reward =  (10/dist_goal)
    else:
      if actions_diff == 0:
        reward = (self._prev_distance - dist_goal)/self._prev_distance + (10/dist_goal)
      else:
        reward = (self._prev_distance - dist_goal)/self._prev_distance + (10/dist_goal) + (1/actions_diff)
    
    self.update_best_action(action)
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
    if self.distance_to_goal(action) > 500:
      print('--------Reset: Divergence--------')
      done = True
    elif self.distance_to_goal(action) < 0.00001:
      print('--------Reset: Convergence--------')
      done = True
    elif self._count % 1000 == 0:  
      # print('--------Reset: Timeout--------')
      done = True


    return done