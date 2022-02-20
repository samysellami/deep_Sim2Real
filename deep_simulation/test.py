from pyrep import PyRep
from pyrep.robots.arms.ur10 import UR10
from pyrep.robots.arms.panda import Panda
from os.path import dirname, join, abspath
import time
import os

SCENE_FILE = join(dirname(abspath(__file__)), 'scenes/scene_UR10.ttt')
# SCENE_FILE = join(dirname(abspath(__file__)), 'scenes/scene_panda_reach_target.ttt')
DELTA = 0.01
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()
# agent = Panda()
agent = UR10()

starting_joint_positions = agent.get_joint_positions()
pos, quat = agent.get_tip().get_position(), agent.get_tip().get_quaternion()
print('start =', starting_joint_positions)

# new_joint_angles = [0, 0, 0, 0, 0, 0]
# new_joint_angles = [x + y for x, y in zip(starting_joint_positions, [-0.1, 0, 0, 0, 0, 0, 0])]
# agent.set_joint_target_positions(new_joint_angles)
# print('new =', new_joint_angles)

# pr.step()  # Step physics simulation

done = False


def move(index, delta):
    pos[index] += delta
    # new_joint_angles = agent.solve_ik(pos, quaternion=quat)
    # new_joint_angles = agent.solve_ik_via_jacobian(pos, quaternion=quat)
    # new_joint_angles = [0, 0, 0, 0, 0, 0, 0]
    new_joint_angles = [0, 0, 0, 0, 0, 0]

    agent.set_joint_target_positions(new_joint_angles)
    pr.step()


[move(2, -DELTA) for _ in range(2000)]
# [move(1, -DELTA) for _ in range(20)]
# [move(2, DELTA) for _ in range(10)]
# [move(1, DELTA) for _ in range(20)]

# time.sleep(10)
# print('Done ...')
# input('Press enter to finish ...')
# time.sleep(10)

done = True
pr.stop()
pr.shutdown()
