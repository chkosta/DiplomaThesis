import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines3 import TD3
import RobotDART as rd
import numpy as np



class ArmEnv(gym.Env):

    def __init__(self, rand_value, viewer):
        # RobotDART initialization data
        self.simu = rd.RobotDARTSimu(0.001)

        # Load arm
        self.robot = rd.Robot("arm.urdf")

        # Position arm
        self.robot.set_actuator_types("torque")  # Control each joint by giving torque commands
        self.robot.fix_to_world()
        self.simu.add_robot(self.robot)
        self.simu.add_floor()

        # Randomization
        self.randomization = rand_value

        # Visualization
        self.visualize(viewer)

        # Rest initialization data
        self._step = 0

        # Limits
        self.max_torque = 5.
        self.max_velocity = 5.
        self.max_pos1 = np.radians(180)   # 3.1416
        self.max_pos_rest = np.radians(90)    # 1.5708

        # Spaces
        action_high = np.array([self.max_torque, self.max_torque, self.max_torque, self.max_torque], dtype=np.float32)

        obs_high = np.array([[self.max_pos1, self.max_pos_rest, self.max_pos_rest, self.max_pos_rest],
                             [self.max_velocity, self.max_velocity, self.max_velocity, self.max_velocity]], dtype=np.float32)

        self.action_space = spaces.Box(
            low=-action_high,
            high=action_high,
            shape=(4,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32
        )

        self.seed()

    def step(self, cmd):
        for i in range(20):
            cmd = np.clip(cmd, -self.max_torque, self.max_torque)

            # Set torque command
            self.robot.set_commands(cmd)

            # Run one simulated step
            self.simu.step_world()


        current_pos = self.robot.positions()
        current_vel = self.robot.velocities()

        self.state = np.array([current_pos, current_vel])

        # Reward function
        reward = -np.linalg.norm(current_pos-self.target_pos)

        self._step += 1
        done = False
        if self._step >= 250:
            done = True

        return self.state, reward, done, {}

    def reset(self):
        # Randomize initial joint positions and set velocities to zero
        self.state = np.array([self.generate_random_pos(), [0., 0., 0., 0.]])
        self.robot.set_positions(self.state[0])
        self.robot.set_velocities(self.state[1])

        # Randomization
        self.randomize(self.randomization)

        # Target joint positions
        self.target_pos = [0., 1.57, -0.5, 0.7]
        self._step = 0

        return self.state

    def generate_random_pos(self):
        pos = np.zeros(4)
        pos[0] = self.np_random.uniform(-self.max_pos1, self.max_pos1)
        pos[1] = self.np_random.uniform(-self.max_pos_rest, self.max_pos_rest)
        pos[2] = self.np_random.uniform(-self.max_pos_rest, self.max_pos_rest)
        pos[3] = self.np_random.uniform(-self.max_pos_rest, self.max_pos_rest)
        return pos

    def randomize(self, randomization):
        if randomization:
            # Randomize arm link's masses in [0, 1]
            mass_low = np.array([0., 0., 0., 0.])
            mass_high = np.array([1., 1., 1., 1.])
            mass = self.np_random.uniform(low=mass_low, high=mass_high)

            self.robot.set_body_mass("arm_link_1", mass[0])
            self.robot.set_body_mass("arm_link_2", mass[1])
            self.robot.set_body_mass("arm_link_3", mass[2])
            self.robot.set_body_mass("arm_link_4", mass[3])
        else:
            # Set fixed values to 2
            self.robot.set_body_mass("arm_link_1", 2.)  # 0.19
            self.robot.set_body_mass("arm_link_2", 2.)  # 0.29
            self.robot.set_body_mass("arm_link_3", 2.)  # 0.22
            self.robot.set_body_mass("arm_link_4", 2.)  # 0.16

    def visualize(self, viewer):
        if viewer:
            # Create graphics
            gconfig = rd.gui.GraphicsConfiguration(720, 480)
            graphics = rd.gui.Graphics(gconfig)
            self.simu.set_graphics(graphics)
            graphics.look_at([0., -3.3, 0.3])
            self.simu.scheduler().set_sync(True)

            graphics.camera().record(True)
            graphics.record_video("./videos/Source Environment/randomized_model.mp4")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class TestArmEnv(gym.Env):

    def __init__(self, viewer):
        # RobotDART initialization data
        self.simu = rd.RobotDARTSimu(0.001)

        # Load arm
        self.robot = rd.Robot("arm.urdf")

        # Position arm
        self.robot.set_actuator_types("torque")  # Control each joint by giving torque commands
        self.robot.fix_to_world()
        self.simu.add_robot(self.robot)
        self.simu.add_floor()

        # Visualization
        self.visualize(viewer)

        # Rest initialization data
        self._step = 0

        # Limits
        self.max_torque = 5.
        self.max_velocity = 5.
        self.max_pos1 = np.radians(180)  # 3.1416
        self.max_pos_rest = np.radians(90)  # 1.5708

        # Spaces
        action_high = np.array([self.max_torque, self.max_torque, self.max_torque, self.max_torque], dtype=np.float32)

        obs_high = np.array([[self.max_pos1, self.max_pos_rest, self.max_pos_rest, self.max_pos_rest],
                             [self.max_velocity, self.max_velocity, self.max_velocity, self.max_velocity]],
                            dtype=np.float32)

        self.action_space = spaces.Box(
            low=-action_high,
            high=action_high,
            shape=(4,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32
        )

        self.seed()

    def step(self, cmd):
        for i in range(20):
            cmd = np.clip(cmd, -self.max_torque, self.max_torque)

            # Set torque command
            self.robot.set_commands(cmd)

            # Run one simulated step
            self.simu.step_world()


        current_pos = self.robot.positions()
        current_vel = self.robot.velocities()

        self.state = np.array([current_pos, current_vel])

        # Reward function
        reward = -np.linalg.norm(current_pos-self.target_pos)

        self._step += 1
        done = False
        if self._step >= 250:
            done = True

        return self.state, reward, done, {}

    def reset(self):
        # Randomize initial joint positions and set velocities to zero
        self.state = np.array([self.generate_random_pos(), [0., 0., 0., 0.]])
        self.robot.set_positions(self.state[0])
        self.robot.set_velocities(self.state[1])

        # Target joint positions
        self.target_pos = [0., 1.57, -0.5, 0.7]
        self._step = 0

        return self.state

    def generate_random_pos(self):
        pos = np.zeros(4)
        pos[0] = self.np_random.uniform(-self.max_pos1, self.max_pos1)
        pos[1] = self.np_random.uniform(-self.max_pos_rest, self.max_pos_rest)
        pos[2] = self.np_random.uniform(-self.max_pos_rest, self.max_pos_rest)
        pos[3] = self.np_random.uniform(-self.max_pos_rest, self.max_pos_rest)
        return pos

    def visualize(self, viewer):
        if viewer:
            # Create graphics
            gconfig = rd.gui.GraphicsConfiguration(720, 480)
            graphics = rd.gui.Graphics(gconfig)
            self.simu.set_graphics(graphics)
            graphics.look_at([0., -3.3, 0.3])
            self.simu.scheduler().set_sync(True)

            graphics.camera().record(True)
            graphics.record_video("./videos/Target Environment/randomized_model.mp4")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



# Visualization
rand_value = True
viewer = True
env = ArmEnv(rand_value, viewer)

# Load the trained model to the same environment
model = TD3.load("./logs/Randomized/best_model.zip")

# Run the learned policy one time to see what happens
obs = env.reset()
episode_reward = 0.
for i in range(250):
    action, states = model.predict(obs)
    obs, reward, done, info = env.step(action)

    episode_reward += reward
    if done:
        print("Reward:", episode_reward)
        episode_reward = 0.