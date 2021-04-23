import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np
from os import path
import RobotDART as rd
import dartpy # OSX breaks if this is imported before RobotDART


# Load robot
robot = rd.Robot("arm.urdf", "arm", False)
robot.fix_to_world()

# Create PD controller to control the arm
control = rd.PDControl([-4.0, 5.5, -3.5, -2.6], False)
# Add it to the robot
robot.add_controller(control, 1.)
control.set_pd(200., 20.)

# Print initial positions of the robot
print(robot.positions())

# Create simulator object
simu = rd.RobotDARTSimu(0.001)

# Create graphics
graphics = rd.gui.Graphics()
simu.set_graphics(graphics)

# Add robot and floor to the simulation
simu.add_robot(robot)
simu.add_checkerboard_floor()

simu.run(5.)


print(robot.positions())