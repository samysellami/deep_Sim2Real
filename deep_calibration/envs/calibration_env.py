import numpy as np
import logging
from numpy import linalg as LA
import math

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import Space

from deep_calibration.utils.kinematics import Kinematics
from deep_calibration import script_dir


class CalibrationEnv(gym.Env):
    """
        Gym environment for the deep calibration
            :param q: (np.ndarray) the initial joint position of the UR10 arm
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self, config=[
            np.array([0, 0, 0, 0, 0, 0]),
            np.array([-math.pi / 2, math.pi, -math.pi / 2, math.pi, math.pi / 2, -math.pi]),
            np.array([math.pi, math.pi / 2, math.pi, math.pi / 2, math.pi, math.pi / 2]),
            np.array([-math.pi / 2, math.pi / 2, -math.pi, math.pi, math.pi / 2, -math.pi / 2]),
            np.array([-math.pi / 2, -math.pi, -math.pi / 2, -math.pi / 2, -math.pi, -math.pi / 2]),
            np.array([-math.pi, -math.pi, -math.pi / 2, -math.pi, -math.pi / 2, -math.pi / 2]),
            np.array([math.pi / 2, math.pi / 2, math.pi / 2, math.pi / 2, math.pi / 2, math.pi / 2]),
            np.array([-math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2]),
            np.array([math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2]),
            np.array([-math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2]),
        ],
        quater=np.array([0.9998, 0.0100, 0.0098, 0.0100]),
        delta=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        p_x=np.array([-0.2, 0.2]),
        p_y=np.array([-0.2, 0.2, -0.2]),
        p_z=np.array([-0.2, 0.2, -0.2]),
        phi_y=np.array([-0.02, 0.02, -0.02, 0.02, -0.02]),
        phi_z=np.array([-0.02, 0.02])
    ):
        # the action encodes the calibration parameters (positional and rotational)
        # lim = 0.0001
        lim = 0.005
        self.action_space = spaces.Box(
            np.array(
                [-lim, -lim, -lim, -lim, -lim]
            ),
            np.array(
                [lim, lim, lim, lim, lim]
            ),
            dtype='float32'
        )

        # the observation encodes  the joint angles
        self.observation_space = spaces.Box(
            np.array(
                [-2*math.pi, -2*math.pi, -2*math.pi, -2*math.pi, -2*math.pi, -2*math.pi]
            ),
            np.array(
                [2*math.pi, 2*math.pi, 2*math.pi, 2*math.pi, 2*math.pi, 2*math.pi]
            ),
            dtype='float32'
        )
        self._configs = config
        self._config = config  # robot configurations
        self._n_config = len(self._config)  # number of configurations
        self.rand = 0  # use random configuration
        self._count = 0  # total time_steps
        self._i = -1  # configuration number
        self._n_episode = -1  # number of episodes
        # self._config = self.setup_configs()

        # calibration parameters
        self._p_base = None
        self._R_base = None
        self._p_tool = None
        self._quater = quater
        self._delta = delta
        self._p_x = p_x
        self._p_y = p_y
        self._p_z = p_z
        # self._phi_x = phi_x
        self._phi_y = phi_y
        self._phi_z = phi_z
        self._prev_distance = None  # previous distance to goal used to compute the reward
        self._all_config = True  # if True compute the mean distance to goal using all configurations
        self._from_p_ij = True  # if True compute the goal position from the real measurements data
        self._form_goal_pos = True  # if True compute the goal position from the kinematics without tools
        self._tune = False

        if self._from_p_ij:
            with open(f"{script_dir}/calibration/p_ij.npy", 'rb') as f:
                self.identified_prms = np.load(f, allow_pickle=True).item()
                self._p_ij = self.identified_prms['p_ij']
                self._p_base = self.identified_prms['p_base']
                self._R_base = self.identified_prms['R_base']
                self._p_tool = self.identified_prms['p_tool']
                self._calib_prms = self.identified_prms['calib_prms']
                self._goal_position = self.identified_prms['goal_position']
        f.close()

        # setup joints and actions
        self._default_action = self.get_default_action(
            quater=quater, delta=delta, p_x=p_x, p_y=p_y, p_z=p_z,
            phi_y=phi_y, phi_z=phi_z
        )
        self.setup_joints()
        self._best_action = np.zeros(
            (self._n_config, self.action_space.shape[0] + 1)
        )
        self.update_best_action(init=True)

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

        # if self._count % 10000 == 0:
        #   print(f'--------best actions --------: {self._best_action}')

        self._prev_distance = None
        self.setup_joints()

        observation = self.get_observation()
        return observation

    def render(self, mode='human'):
        ...

    def close(self):
        ...

# ----------------------------  env-specific methods -----------------------------------

    def update_best_action(self, action=None, init=False):
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
            self._n_episode += 1
            self._i = (self._i + 1) % len(self._config)
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
        self._goal_pos = self.get_goal_position()

    def get_default_action(self, quater, delta, p_x, p_y, p_z,
                           phi_y, phi_z):
        action = np.zeros(5)
        action = delta + self._calib_prms

        return action

    def get_position(self, action=None, tool=None):
        """
          Return the end effector position
            :param action: (np.ndarray) the calibration parameters 
            :return: (np.ndarray) the position of the end effector
        """
        if action is None:
            action = self._default_action

        if self._tune:
            self._delta = action + self._calib_prms
        else:
            self._delta = action

        if tool is not None:
            FK = Kinematics(
                p_base=self._p_base, R_base=self._R_base, p_tool=self._p_tool,
                delta=self._delta
            )
            pos = []
            for j in range(3):
                pos.append(FK.forward_kinematics(q=self._q, j=j)[0])
        else:
            FK = Kinematics(
                p_base=self._p_base, R_base=self._R_base, p_tool=self._p_tool,
                delta=self._delta
            )
            pos = FK.forward_kinematics(q=self._q)[0]

        return np.array(pos)

    def get_goal_position(self):
        if self._from_p_ij:
            if self._form_goal_pos:
                return self._goal_position[self._i]

            return self._p_ij[self._i]
            # return self._p_ij[self._n_episode % len(self._p_ij)]

        return self.get_position()

    def distance_to_goal(self, action=None):
        """
            Compute the distance to the goal
                :param action: (np.ndarray) the calibration parameters 
        """
        if self._all_config:
            dists_goal = []
            for j in range(self._n_config):
                if self._form_goal_pos:
                    dists_goal.append(np.mean(np.abs(self.get_position(action) - self._goal_pos)))
                else:
                    # dists_goal.append(LA.norm(self.get_position(action) - self._goal_pos))
                    dists_goal.append(np.mean(np.abs(self.get_position(action, tool=True) - self._goal_pos)))
                self.setup_joints()
            dist_goal = np.mean(dists_goal)
        else:
            dist_goal = LA.norm(self.get_position(action) - self._goal_pos)

        return dist_goal

    def get_observation(self, action=None):
        """
            Return the environment observation
                :param action: (np.ndarray) the calibration parameters 
                :return: (np.ndarray) the environment observation
        """
        if action is None:
            action = self._default_action

        # pos = self.get_position(action)
        return self._q
        # return np.hstack((pos,self._q))

    def compute_reward(self, action):
        """
            Compute the reward value for the step function
                :param action: (np.ndarray) the calibration parameters 
        """

        dist_goal = self.distance_to_goal(action)

        actions_diff = LA.norm(
            self._best_action[self._best_action[:, -1] < dist_goal, :-1] - action
        )
        actions_diff = 0

        if self._prev_distance is None:
            self._prev_distance = self.distance_to_goal(action)
            reward = (1/dist_goal)
        else:
            if actions_diff == 0:
                reward = (self._prev_distance - dist_goal) / \
                    self._prev_distance + (1/dist_goal)
            else:
                reward = (self._prev_distance - dist_goal) / \
                    self._prev_distance + (1/dist_goal) + (1/actions_diff)

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


def main():
    env = CalibrationEnv()
    print('distance to goal: ', env.distance_to_goal(env._calib_prms) * 1000)


if __name__ == "__main__":
    main()
