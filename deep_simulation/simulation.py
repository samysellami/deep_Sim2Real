
from deep_simulation.environment import CoppeliaSimEnvWrapper
import time
import numpy as np

if __name__ == "__main__":
	sim = CoppeliaSimEnvWrapper()
	
	# time.sleep(10)
	# observation = env._observation_space.sample()
	# action = env._action_space.sample()
	# print("resetting the environment !!!")
	# sim.reset()
	time.sleep(10)							
	DELTA = 0.01
	sim.env.start()

	print('Planning path  ...')
	path = sim.UR10_arm.get_path(position= sim.goal.get_position(),
                            quaternion= sim.gripper_dummy.get_quaternion(), ignore_collisions=True)

	path.visualize()  # Let's see what the path looks like
	print('Executing plan ...')
	done = False
	while not done:
	    done = path.step()
	    print(done)
	    sim.env.step()
	path.clear_visualization()

	# print('Closing left gripper ...')
	# while not sim.baxter_gripper.actuate(0.0, 0.1):
	# 	sim.env.step()
	# sim.baxter_gripper.grasp(cup)

	# goal_pos = sim.goal.get_position()
	# sim.gripper.set_position(goal_pos)
	# sim.env.step()
	# time.sleep(10)	

