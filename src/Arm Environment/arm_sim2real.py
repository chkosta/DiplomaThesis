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



class RandomizedArmEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # RobotDART initialization data
        self.simu = rd.RobotDARTSimu(0.001)
        self.robot = rd.Robot("arm.urdf")
        self.robot.set_actuator_types("torque")  # Control each joint by giving torque commands
        self.robot.fix_to_world()
        self.simu.add_robot(self.robot)
        self.simu.add_floor()

        # Rest initialization data
        self._step = 0
        self.max_torque = 5.
        self.max_velocity = 5.
        self.max_theta1 = np.radians(180)   # 3.1416
        self.max_theta_rest = np.radians(90)    # 1.5708

        # Spaces
        action_high = np.array([self.max_torque, self.max_torque, self.max_torque, self.max_torque], dtype=np.float32)
        obs_high = np.array([[self.max_theta1, self.max_theta_rest, self.max_theta_rest, self.max_theta_rest],
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
        # Target joint positions
        target_pos = [0., 1.57, -0.5, 0.7]

        cmd = np.clip(cmd, -self.max_torque, self.max_torque)

        # Set torque command
        self.robot.set_commands(cmd)

        for i in range(20):
            # Run one simulated step
            self.simu.step_world()

        # Calculate reward
        current_pos = np.round(self.robot.positions(), 2)
        current_vel = np.round(self.robot.velocities(), 2)

        costs = euclidean(current_pos, target_pos)
        reward = -costs

        self.state = np.array([current_pos, current_vel])

        self._step += 1
        done = False
        if self._step >= 250:
            done = True

        return self.state, reward, done, {}

    def reset(self):
        # Randomize initial joint positions and velocities
        self.state = np.array([self.generate_random_pos(), [0., 0., 0., 0.]])
        self.robot.set_positions(self.state[0])
        self.robot.set_velocities(self.state[1])

        # Randomize link's masses
        self.mass = self.generate_random_mass()
        self.robot.set_body_mass("arm_link_1", self.mass[0])
        self.robot.set_body_mass("arm_link_2", self.mass[1])
        self.robot.set_body_mass("arm_link_3", self.mass[2])
        self.robot.set_body_mass("arm_link_4", self.mass[3])

        self._step = 0
        return self.state

    def generate_random_pos(self):
        pos = np.zeros(4)
        pos[0] = self.np_random.uniform(-self.max_theta1, self.max_theta1)
        pos[1] = self.np_random.uniform(-self.max_theta_rest, self.max_theta_rest)
        pos[2] = self.np_random.uniform(-self.max_theta_rest, self.max_theta_rest)
        pos[3] = self.np_random.uniform(-self.max_theta_rest, self.max_theta_rest)
        return pos

    def generate_random_mass(self):
        mass_low = np.array([0.10, 0.10, 0.10, 0.10])
        mass_high = np.array([0.60, 0.70, 0.63, 0.57])
        mass = self.np_random.uniform(low=mass_low, high=mass_high)
        return mass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class NonRandomizedArmEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # RobotDART initialization data
        self.simu = rd.RobotDARTSimu(0.001)
        self.robot = rd.Robot("arm.urdf")
        self.robot.set_actuator_types("torque")  # Control each joint by giving torque commands
        self.robot.fix_to_world()
        self.simu.add_robot(self.robot)
        self.simu.add_floor()

        # Set link's masses
        self.robot.set_body_mass("arm_link_1", 0.59)        # 0.19
        self.robot.set_body_mass("arm_link_2", 0.69)        # 0.29
        self.robot.set_body_mass("arm_link_3", 0.62)        # 0.22
        self.robot.set_body_mass("arm_link_4", 0.56)        # 0.16

        # Rest initialization data
        self._step = 0
        self.max_torque = 5.
        self.max_velocity = 5.
        self.max_theta1 = np.radians(180)   # 3.1416
        self.max_theta_rest = np.radians(90)    # 1.5708

        # Spaces
        action_high = np.array([self.max_torque, self.max_torque, self.max_torque, self.max_torque], dtype=np.float32)
        obs_high = np.array([[self.max_theta1, self.max_theta_rest, self.max_theta_rest, self.max_theta_rest],
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
        # Target joint positions
        target_pos = [0., 1.57, -0.5, 0.7]

        cmd = np.clip(cmd, -self.max_torque, self.max_torque)

        # Set torque command
        self.robot.set_commands(cmd)

        for i in range(20):
            # Run one simulated step
            self.simu.step_world()

        # Calculate reward
        current_pos = np.round(self.robot.positions(), 2)
        current_vel = np.round(self.robot.velocities(), 2)

        costs = euclidean(current_pos, target_pos)
        reward = -costs

        self.state = np.array([current_pos, current_vel])

        self._step += 1
        done = False
        if self._step >= 250:
            done = True

        return self.state, reward, done, {}

    def reset(self):
        # Randomize initial joint positions and velocities
        self.state = np.array([self.generate_random_pos(), [0., 0., 0., 0.]])
        self.robot.set_positions(self.state[0])
        self.robot.set_velocities(self.state[1])

        self._step = 0
        return self.state

    def generate_random_pos(self):
        pos = np.zeros(4)
        pos[0] = self.np_random.uniform(-self.max_theta1, self.max_theta1)
        pos[1] = self.np_random.uniform(-self.max_theta_rest, self.max_theta_rest)
        pos[2] = self.np_random.uniform(-self.max_theta_rest, self.max_theta_rest)
        pos[3] = self.np_random.uniform(-self.max_theta_rest, self.max_theta_rest)
        return pos

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class TestArmEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # RobotDART initialization data
        self.simu = rd.RobotDARTSimu(0.001)
        self.robot = rd.Robot("arm.urdf")
        self.robot.set_actuator_types("torque")  # Control each joint by giving torque commands
        self.robot.fix_to_world()
        self.simu.add_robot(self.robot)
        self.simu.add_floor()

        # Rest initialization data
        self._step = 0
        self.max_torque = 5.
        self.max_velocity = 5.
        self.max_theta1 = np.radians(180)   # 3.1416
        self.max_theta_rest = np.radians(90)    # 1.5708

        # Spaces
        action_high = np.array([self.max_torque, self.max_torque, self.max_torque, self.max_torque], dtype=np.float32)
        obs_high = np.array([[self.max_theta1, self.max_theta_rest, self.max_theta_rest, self.max_theta_rest],
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
        # Target joint positions
        target_pos = [0., 1.57, -0.5, 0.7]

        cmd = np.clip(cmd, -self.max_torque, self.max_torque)

        # Set torque command
        self.robot.set_commands(cmd)

        for i in range(20):
            # Run one simulated step
            self.simu.step_world()

        # Calculate reward
        current_pos = np.round(self.robot.positions(), 2)
        current_vel = np.round(self.robot.velocities(), 2)

        costs = euclidean(current_pos, target_pos)
        reward = -costs

        self.state = np.array([current_pos, current_vel])

        self._step += 1
        done = False
        if self._step >= 250:
            done = True

        return self.state, reward, done, {}

    def reset(self):
        # Randomize initial joint positions and velocities
        self.state = np.array([self.generate_random_pos(), [0., 0., 0., 0.]])
        self.robot.set_positions(self.state[0])
        self.robot.set_velocities(self.state[1])

        self._step = 0
        return self.state

    def generate_random_pos(self):
        pos = np.zeros(4)
        pos[0] = self.np_random.uniform(-self.max_theta1, self.max_theta1)
        pos[1] = self.np_random.uniform(-self.max_theta_rest, self.max_theta_rest)
        pos[2] = self.np_random.uniform(-self.max_theta_rest, self.max_theta_rest)
        pos[3] = self.np_random.uniform(-self.max_theta_rest, self.max_theta_rest)
        return pos

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


def load(dir):
    df = open(dir)
    csv_df = csv.reader(df)
    next(csv_df)
    next(csv_df)
    reward = []
    reward_non = []

    if dir == "./logs/Randomized/monitor.csv":
        for row in csv_df:
            reward.append(float(row[0]))
        reward_list.append(reward)
        print(reward_list)

    if dir == "./logs/NonRandomized/monitor.csv":
        for row in csv_df:
            reward_non.append(float(row[0]))
        reward_list_non.append(reward_non)
        print(reward_list_non)


def perc(reward_list):
    rwd_med_list = np.median(reward_list, axis=0)
    rwd_perc_25 = np.percentile(reward_list, 25, axis=0)
    rwd_perc_75 = np.percentile(reward_list, 75, axis=0)
    return rwd_med_list, rwd_perc_25, rwd_perc_75



# Create log directories
log_randomized = "./logs/Randomized/"
os.makedirs(log_randomized, exist_ok=True)
log_non_randomized = "./logs/NonRandomized/"
os.makedirs(log_non_randomized, exist_ok=True)


reward_list = []
reward_list_non = []


for i in range(20):
    # Instantiate the simulated environment with domain randomization
    randomized_env = RandomizedArmEnv()

    # Instantiate the simulated environment without domain randomization
    non_randomized_env = NonRandomizedArmEnv()

    # Instantiate the real environment and wrap it
    env_test = TestArmEnv()
    env_test1 = Monitor(env_test, log_randomized)
    env_test2 = Monitor(env_test, log_non_randomized)

    # Check for warnings
    # check_env(env)


    # The noise objects for TD3
    n_actions = randomized_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    randomized_model = TD3("MlpPolicy", randomized_env, action_noise=action_noise, verbose=1)
    non_randomized_model = TD3("MlpPolicy", non_randomized_env, action_noise=action_noise, verbose=1)

    timesteps = int(50000)

    # For every 10 episodes of learning, test to the real environment (with domain randomization)
    randomized_model.learn(total_timesteps=timesteps, log_interval=50, eval_env=env_test1, eval_freq=2000, n_eval_episodes=1, eval_log_path=log_randomized)

    # For every 10 episodes of learning, test to the real environment (without domain randomization)
    non_randomized_model.learn(total_timesteps=timesteps, log_interval=50, eval_env=env_test2, eval_freq=2000, n_eval_episodes=1, eval_log_path=log_non_randomized)


    # Dataframe split to get only the important data (rewards)
    dir_randomized = "./logs/Randomized/monitor.csv"
    load(dir_randomized)
    dir_non_randomized = "./logs/NonRandomized/monitor.csv"
    load(dir_non_randomized)



# Plot the results using the first type of learning/testing
# Compute the median and 25/75 percentiles with domain randomization
rwd_med_list, rwd_perc_25, rwd_perc_75 = perc(reward_list)

# Compute the median and 25/75 percentiles without domain randomization
rwd_med_list_non, rwd_perc_25_non, rwd_perc_75_non = perc(reward_list_non)

# Timesteps list
tmps_list = list(range(2000, timesteps + 2000, 2000))

plt.fill_between(tmps_list, rwd_perc_25, rwd_perc_75, alpha=0.25, linewidth=2, color='#006BB2')
plt.fill_between(tmps_list, rwd_perc_25_non, rwd_perc_75_non, alpha=0.25, linewidth=2, color='#B22400')

plt.plot(tmps_list, rwd_med_list)
plt.plot(tmps_list, rwd_med_list_non)
plt.legend(["Domain Randomization", "No Domain Randomization"])
plt.title("TD3 Arm")
plt.xlabel("Timesteps")
plt.ylabel("Median Rewards")
plt.show()