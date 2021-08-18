import csv
import os
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import RobotDART as rd
import dartpy



class FrankaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # RobotDART initialization data
        dt = 0.001
        Kp = 2.
        Ki = 0.01
        self.simu = rd.RobotDARTSimu(dt)
        self.simu.set_collision_detector("fcl")

        # Load Franka
        packages = [("franka_description", "franka/franka_description")]
        self.robot = rd.Robot("franka/franka.urdf", packages)
        self.robot.set_color_mode("material")

        # Load box
        self.box_size = [0.06, 0.06, 0.06]
        self.tf = dartpy.math.Isometry3()
        self.tf.set_rotation(dartpy.math.eulerZYXToMatrix([0., 0., 0.]))
        self.tf.set_translation([0.7, 0.3, self.box_size[2] / 2.])
        self.box_pose = rd.math.logMap(self.tf.rotation()).tolist() + self.tf.translation().tolist()
        self.box = rd.Robot.create_box(self.box_size, self.box_pose, "free", mass=0.01, color=[0.1, 0.2, 0.9, 1.0])

        # Position Franka and box
        self.robot.set_actuator_types("servo")     # Control each joint by giving velocity commands
        self.robot.fix_to_world()
        self.simu.add_robot(self.robot)
        self.simu.add_robot(self.box)
        self.simu.add_floor()

        # Get end-effector and box pose
        self.eef_link_name = "panda_hand"
        self.target_eef_pose = self.robot.body_pose(self.eef_link_name)
        self.box_body_pose = self.box.base_pose()

        # Initialize controller
        self.controller = PITask(self.target_eef_pose, dt, Kp, Ki)

        # Rest initialization data
        self._step = 0

        self.min_velocity = -2.
        self.max_velocity = 2.
        self.min_finger_velocity = -0.2
        self.max_finger_velocity = 0.2
        self.min_theta_x = -0.855
        self.max_theta_x = 0.855
        self.min_theta_y = -0.36
        self.max_theta_y = 1.19
        self.min_theta_z = -0.855
        self.max_theta_z = 0.855

        # Spaces
        action_low = np.array([self.min_velocity, self.min_velocity, self.min_velocity, self.min_finger_velocity], dtype=np.float32)
        action_high = np.array([self.max_velocity, self.max_velocity, self.max_velocity, self.max_finger_velocity], dtype=np.float32)

        obs_low = np.array([[self.min_theta_x, self.min_theta_y, self.min_theta_z], [self.min_velocity, self.min_velocity, self.min_velocity]], dtype=np.float32)
        obs_high = np.array([[self.max_theta_x, self.max_theta_y, self.max_theta_z], [self.max_velocity, self.max_velocity, self.max_velocity]], dtype=np.float32)

        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            shape=(4,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        self.seed()

    def step(self, vel2):

        # At each step get the end effector pose
        current_eef_pose = self.robot.body_pose(self.eef_link_name)

        # Gripper open (negative velocities) or close (positive velocities)
        gripper_vel = vel2[3]

        vel1 = self.controller.update(current_eef_pose)
        vel = [vel1[0], vel1[1], vel1[2], vel2[0], vel2[1], vel2[2]]
        jac = self.robot.jacobian(self.eef_link_name)
        jac_pinv = damped_pseudoinverse(jac)
        cmd = jac_pinv @ vel
        cmd[7] = gripper_vel

        self.robot.set_commands(cmd)

        for i in range(20):
            # Run one simulated step
            self.simu.step_world()


        current_eef_pos = current_eef_pose.translation()
        current_eef_vel = self.robot.body_velocity(self.eef_link_name)
        current_eef_vel = [current_eef_vel[3], current_eef_vel[4], current_eef_vel[5]]

        self.state = np.array([current_eef_pos, current_eef_vel])


        # Reward function
        current_box_pos = self.box_body_pose.translation()
        dist = euclidean(current_eef_pos, current_box_pos)
        reaching_reward = dist

        # Two phases of the task
        # 0: Reaching the box
        # 1: Grasping the box and lifting it up
        if self.phase == 0:
            reward = reaching_reward

            # Gripper open reward
            if gripper_vel < 0:
                reward += abs(gripper_vel)

            if dist < 0.025:
                self.phase = 1

        elif self.phase == 1:
            reward = reaching_reward

            # Grasping reward
            if self.has_grasp:
                reward += 1.0

        # Success reward
        if self.check_success():
            reward += 2.0


        self._step += 1
        done = False
        if self._step >= 250:
            done = True

        return self.state, reward, done, {}

    def reset(self):
        # Set initial joint positions and velocities (fingers are open)
        joint_pos = [0., 0., 0., -1.5708, 0., 1.5708, 0., 0.04, 0.04]
        joint_vel = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.robot.reset()
        self.robot.set_positions(joint_pos)
        self.robot.set_velocities(joint_vel)

        eef_pos = self.target_eef_pose.translation()
        eef_vel = self.robot.body_velocity(self.eef_link_name)
        eef_vel = [eef_vel[3], eef_vel[4], eef_vel[5]]

        self.state = np.array([eef_pos, eef_vel])

        self.phase = 0
        self.has_grasp = False
        self._step = 0

        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def check_success(self):
        # Returns True if task has been completed
        current_box_pos = self.box_body_pose.translation()

        return self.has_grasp and current_box_pos[2] > 0.86


class PITask:
    def __init__(self, target, dt, Kp=10., Ki=0.1):
        self._target = target
        self._dt = dt
        self._Kp = Kp
        self._Ki = Ki
        self._sum_error = 0

    def set_target(self, target):
        self._target = target

    # function to compute error
    def error(self, tf):
        # compute error directly in world frame
        rot_error = rd.math.logMap(self._target.rotation() @ tf.rotation().T)
        lin_error = self._target.translation() - tf.translation()
        return np.r_[rot_error, lin_error]

    def update(self, current):
        error_in_world_frame = self.error(current)

        self._sum_error = self._sum_error + error_in_world_frame * self._dt

        return self._Kp * error_in_world_frame + self._Ki * self._sum_error


def error(tf, tf_desired):
    return np.linalg.norm(rd.math.logMap(tf.inverse().multiply(tf_desired)))


def damped_pseudoinverse(jac, l=0.01):
    m, n = jac.shape
    if n >= m:
        return jac.T @ np.linalg.inv(jac @ jac.T + l * l * np.eye(m))
    return np.linalg.inv(jac.T @ jac + l * l * np.eye(n)) @ jac.T



# Instantiate the simulated environment with domain randomization
env = FrankaEnv()

# Check for warnings
# check_env(env)


# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)

timesteps = int(50000)

# For every 10 episodes of learning, test to the real environment (with domain randomization)
model.learn(total_timesteps=timesteps, log_interval=40)

