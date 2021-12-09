import csv
import os
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import RobotDART as rd
import dartpy
import math



class FrankaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, viewer):
        # RobotDART initialization data
        dt = 0.001
        self.simu = rd.RobotDARTSimu(dt)
        self.simu.set_collision_detector("fcl")

        # Load Franka
        packages = [("franka_description", "franka/franka_description")]
        self.robot = rd.Robot("franka/franka.urdf", packages)
        self.robot.set_color_mode("material")

        # Load box
        self.box_size = [0.04, 0.04, 0.04]
        self.tf = dartpy.math.Isometry3()
        self.tf.set_rotation(dartpy.math.eulerZYXToMatrix([0., 0., 0.]))
        self.tf.set_translation([0.4, 0.3, 0.02])
        self.box_pose = rd.math.logMap(self.tf.rotation()).tolist() + self.tf.translation().tolist()
        self.box = rd.Robot.create_box(self.box_size, self.box_pose, "free", mass=0.4, color=[0.1, 0.2, 0.9, 1.0])

        # Position Franka and box
        self.robot.set_actuator_types("torque")                     # Control each joint by giving torque commands
        self.robot.set_actuator_types("servo", ["panda_joint7"])    # Control gripper joint by giving velocity commands
        self.robot.fix_to_world()
        self.simu.add_robot(self.robot)
        self.simu.add_robot(self.box)
        self.simu.add_floor(floor_width=40.0)

        # Visualization
        self.visualize(viewer)

        # Rest initialization data
        self._step = 0

        # Limits
        # End effector
        self.eef_min_vel = -2.
        self.eef_max_vel = 2.
        self.finger_min_vel = -0.2
        self.finger_max_vel = 0.2
        self.eef_min_pos_x = -0.855
        self.eef_max_pos_x = 0.855
        self.eef_min_pos_y = -0.855
        self.eef_max_pos_y = 0.855
        self.eef_min_pos_z = 0.
        self.eef_max_pos_z = 1.190
        # Βox
        self.box_min_pos_x = -2.5
        self.box_max_pos_x = 2.5
        self.box_min_pos_y = -2.5
        self.box_max_pos_y = 2.5
        self.box_min_pos_z = 0.02
        self.box_max_pos_z = 1.190

        # Spaces
        action_low = np.array([self.eef_min_vel, self.eef_min_vel, self.eef_min_vel, self.finger_min_vel], dtype=np.float32)
        action_high = np.array([self.eef_max_vel, self.eef_max_vel, self.eef_max_vel, self.finger_max_vel], dtype=np.float32)

        obs_low = np.array([[self.eef_min_pos_x, self.eef_min_pos_y, self.eef_min_pos_z], [self.eef_min_vel, self.eef_min_vel, self.eef_min_vel],
                            [self.box_min_pos_x, self.box_min_pos_y, self.box_min_pos_z]], dtype=np.float32)
        obs_high = np.array([[self.eef_max_pos_x, self.eef_max_pos_y, self.eef_max_pos_z], [self.eef_max_vel, self.eef_max_vel, self.eef_max_vel],
                             [self.box_max_pos_x, self.box_max_pos_y, self.box_max_pos_z]], dtype=np.float32)

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

        for i in range(20):
            # At each step get the last end effector pose
            last_eef_pose = self.robot.body_pose(self.eef_link_name)

            vel1 = self.controller.update(last_eef_pose)
            vel = [vel1[0], vel1[1], vel1[2], vel2[0], vel2[1], vel2[2]]
            jac = self.robot.jacobian(self.eef_link_name)
            jac_pinv = damped_pseudoinverse(jac)
            cmd = jac_pinv @ vel

            fingers_vel = vel2[3]
            cmd[7] = fingers_vel
            cmd[8] = 0.

            self.robot.set_commands(cmd)

            # Run one simulated step
            self.simu.step_world()


        # At each step get the current end effector position and velocity
        current_eef_pos = self.robot.body_pose(self.eef_link_name).translation()

        current_eef_vel = self.robot.body_velocity(self.eef_link_name)
        current_eef_vel = np.array([current_eef_vel[3], current_eef_vel[4], current_eef_vel[5]])

        # At each step get the current box position
        current_box_pos = self.box.base_pose_vec()
        current_box_pos = np.array([current_box_pos[3], current_box_pos[4], current_box_pos[5]])

        self.state = np.array([current_eef_pos, current_eef_vel, current_box_pos])


        # Reward function
        reward = -0.1 * np.linalg.norm(current_eef_pos-current_box_pos)

        # if cube is lifted
        if current_box_pos[2] > 0.05:
            reward += 5.0                                                   # bonus for lifting the cube
            reward += -0.5*np.linalg.norm(current_eef_pos-self.target_pos)  # make gripper go to target
            reward += -0.5*np.linalg.norm(current_box_pos-self.target_pos)  # make cube go to target

        # BONUS
        if np.linalg.norm(current_box_pos-self.target_pos) < 0.15:
            reward += 10.0                                                  # bonus for cube close to target

        if np.linalg.norm(current_box_pos-self.target_pos) < 0.1:
            reward += 20.0                                                  # bonus for cube very close to target

        if np.linalg.norm(current_box_pos-self.target_pos) < 0.05:
            reward += 30.0                                                  # bonus for cube extremely close to target


        self._step += 1
        done = False
        if self._step >= 500:
            done = True

        if math.isnan(reward):
            print("current eef pos:", current_eef_pos)
            print("current cube pos:", current_box_pos)
            print("velocities:", vel, fingers_vel)
            print("cmd:", cmd)
            exit()

        return self.state, reward, done, {}

    def reset(self):
        # Robot initial joint positions (fingers are close)
        joint_pos = [0., 0., 0., -1.5708, 0., 1.5708, 0.8, 0., 0.]

        # Reset robot and box
        self.robot.reset()
        self.robot.set_positions(joint_pos)

        self.box.reset()
        self.box.set_base_pose(self.tf)

        # Get eef target pose
        self.eef_link_name = "panda_hand"
        self.eef_target_pose = self.robot.body_pose(self.eef_link_name)

        # Reset controller
        self.controller = PITask(self.eef_target_pose)

        eef_pos = self.robot.body_pose(self.eef_link_name).translation()

        eef_vel = self.robot.body_velocity(self.eef_link_name)
        eef_vel = np.array([eef_vel[3], eef_vel[4], eef_vel[5]])

        box_pos = self.box.base_pose_vec()
        box_pos = np.array([box_pos[3], box_pos[4], box_pos[5]])

        self.target_pos = box_pos + [0., 0., 0.18]

        self.state = np.array([eef_pos, eef_vel, box_pos])
        self._step = 0

        return self.state

    def visualize(self, viewer):
        if viewer:
            # Create graphics
            gconfig = rd.gui.GraphicsConfiguration(720, 480)
            graphics = rd.gui.Graphics(gconfig)
            self.simu.set_graphics(graphics)
            graphics.look_at([2., 0., 1.])
            self.simu.scheduler().set_sync(False)

            graphics.camera().record(True)
            graphics.record_video("sac_franka.mp4")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]


class PITask:
    def __init__(self, target, dt=0.001, Kp=200., Ki=2.):
        self._target = target
        self._dt = dt
        self._Kp = Kp
        self._Ki = Ki
        self._sum_error = 0

    def set_target(self, target):
        self._target = target

    def error(self, tf):
        rot_error = rd.math.logMap(self._target.rotation() @ tf.rotation().T)
        lin_error = self._target.translation() - tf.translation()
        return np.r_[rot_error, lin_error]

    def update(self, current):
        error_in_world_frame = self.error(current)

        self._sum_error = self._sum_error + error_in_world_frame * self._dt

        return self._Kp * error_in_world_frame + self._Ki * self._sum_error


def damped_pseudoinverse(jac, l=0.01):
    m, n = jac.shape
    if n >= m:
        return jac.T @ np.linalg.inv(jac @ jac.T + l * l * np.eye(m))
    return np.linalg.inv(jac.T @ jac + l * l * np.eye(n)) @ jac.T





# Instantiate the simulated environment
env = FrankaEnv(False)

# The noise objects for SAC
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Save a checkpoint every 250000 steps
checkpoint_callback = CheckpointCallback(save_freq=250000, save_path='./logs/')

model = SAC(policy="MlpPolicy", env=env, action_noise=action_noise, verbose=1, device="cuda")

timesteps = int(6000000)
model.learn(total_timesteps=timesteps, callback=checkpoint_callback, log_interval=100)

model.save("sac_franka_model")
model.save_replay_buffer("sac_franka_replay_buffer")

del model # remove to demonstrate saving and loading


# Visualization
env = FrankaEnv(True)

# Load the trained model to the same environment
model = SAC.load("sac_franka_model")
model.load_replay_buffer("sac_franka_replay_buffer")


# Run the learned policy one time to see what happens
obs = env.reset()
episode_reward = 0.
for i in range(500):
    action, states = model.predict(obs)
    obs, reward, done, info = env.step(action)

    episode_reward += reward
    if done:
        print("Reward:", episode_reward)
        episode_reward = 0.