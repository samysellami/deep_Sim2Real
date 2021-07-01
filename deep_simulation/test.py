from pyrep import PyRep
from pyrep.robots.arms.ur10 import UR10
import os 

SCENE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scenes/scene_UR10.ttt')
DELTA = 0.01
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()
agent = UR10()

starting_joint_positions = agent.get_joint_positions()
pos, quat = agent.get_tip().get_position(), agent.get_tip().get_quaternion()
print('start =', starting_joint_positions)

new_joint_angles  = [x + y for x, y in zip(starting_joint_positions, [1.57, 0, 0, 0, 0, 0])] 

agent.set_joint_target_positions(new_joint_angles)
pr.step()
print('new =',agent.get_joint_positions())



def move(index, delta):
    pos[index] += delta
    new_joint_angles = agent.solve_ik(pos, quaternion=quat)
    agent.set_joint_target_positions(new_joint_angles)
    pr.step()


# [move(1, DELTA) for _ in range(1)]

pr.stop()
pr.shutdown()



