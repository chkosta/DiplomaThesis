import csv
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import RobotDART as rd



class ArmEnv(gym.Env):

    def __init__(self, rand_value):
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
        reward = -euclidean(current_pos, self.target_pos)

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
            # Randomize arm link's masses +-80%
            mass_low = np.array([0.04, 0.06, 0.04, 0.03])
            mass_high = np.array([0.34, 0.52, 0.40, 0.29])
            mass = self.np_random.uniform(low=mass_low, high=mass_high)

            self.robot.set_body_mass("arm_link_1", mass[0])
            self.robot.set_body_mass("arm_link_2", mass[1])
            self.robot.set_body_mass("arm_link_3", mass[2])
            self.robot.set_body_mass("arm_link_4", mass[3])

        else:
            # Set fixed values +80%
            self.robot.set_body_mass("arm_link_1", 0.34)  # 0.19
            self.robot.set_body_mass("arm_link_2", 0.52)  # 0.29
            self.robot.set_body_mass("arm_link_3", 0.40)  # 0.22
            self.robot.set_body_mass("arm_link_4", 0.29)  # 0.16

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class TestArmEnv(gym.Env):

    def __init__(self):
        # RobotDART initialization data
        self.simu = rd.RobotDARTSimu(0.001)

        # Load arm
        self.robot = rd.Robot("arm.urdf")

        # Position arm
        self.robot.set_actuator_types("torque")  # Control each joint by giving torque commands
        self.robot.fix_to_world()
        self.simu.add_robot(self.robot)
        self.simu.add_floor()


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
        reward = -euclidean(current_pos, self.target_pos)

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

    if dir == "./logs/Boxplot/monitor.csv":
        for row in csv_df:
            reward.append(float(row[0]))

        # Calculate median rewards (for boxplotting)
        reward = np.median(reward)

        reward_list.append(reward)
        print(reward_list)


def perc(reward_list):
    rwd_med_list = np.median(reward_list, axis=0)
    rwd_perc_25 = np.percentile(reward_list, 25, axis=0)
    rwd_perc_75 = np.percentile(reward_list, 75, axis=0)
    return rwd_med_list, rwd_perc_25, rwd_perc_75




reward_list = []
reward_list_non = []

# Run 10 times
for i in range(10):

    # Instantiate the simulated environment with domain randomization
    randomization = True
    randomized_env = ArmEnv(randomization)
    # Instantiate the real environment
    test_rand_env = TestArmEnv()
    # Monitor the real environment
    test_rand_mon = Monitor(test_rand_env, "./logs/Randomized/")

    # The noise objects for TD3
    n_actions = randomized_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    randomized_model = TD3("MlpPolicy", randomized_env, action_noise=action_noise, verbose=1)

    timesteps = int(500000)
    # For every 10 episodes of learning, test to the real environment (with domain randomization)
    randomized_model.learn(total_timesteps=timesteps, log_interval=500, eval_env=test_rand_mon, eval_freq=2500, n_eval_episodes=1, eval_log_path="./logs/Randomized/")



    # Instantiate the simulated environment without domain randomization
    randomization = False
    non_randomized_env = ArmEnv(randomization)
    # Instantiate the real environment
    test_nonrand_env = TestArmEnv()
    # Monitor the real environment
    test_nonrand_mon = Monitor(test_nonrand_env, "./logs/NonRandomized/")

    # The noise objects for TD3
    n_actions = non_randomized_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    non_randomized_model = TD3("MlpPolicy", non_randomized_env, action_noise=action_noise, verbose=1)

    non_randomized_env.reset()
    # For every 10 episodes of learning, test to the real environment (without domain randomization)
    non_randomized_model.learn(total_timesteps=timesteps, log_interval=500, eval_env=test_nonrand_mon, eval_freq=2500, n_eval_episodes=1, eval_log_path="./logs/NonRandomized/")



    # # Instantiate the simulated environment with domain randomization
    # randomization = True
    # randomized_env = ArmEnv(randomization)
    # # Instantiate the real environment
    # test_rand_env = TestArmEnv()
    # # Monitor the real environment
    # test_rand_mon = Monitor(test_rand_env, "./logs/Boxplot/")
    #
    # # The noise objects for TD3
    # n_actions = randomized_env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    #
    # model = TD3("MlpPolicy", randomized_env, action_noise=action_noise, verbose=1)
    #
    # timesteps = 500000
    # randomized_env.reset()
    # # After 500000 steps of learning, test to the real environment
    # model.learn(total_timesteps=timesteps, log_interval=500, eval_env=test_rand_mon, eval_freq=500000, n_eval_episodes=1, eval_log_path="./logs/Boxplot/")


    choice = 1
    load("./logs/Randomized/monitor.csv")
    load("./logs/NonRandomized/monitor.csv")

    # choice = 2
    # load("./logs/Boxplot/monitor.csv")



if choice == 1:
    # Plot the results using the first type of learning/testing
    # Compute the median and 25/75 percentiles with domain randomization
    rwd_med_list, rwd_perc_25, rwd_perc_75 = perc(reward_list)

    # Compute the median and 25/75 percentiles without domain randomization
    rwd_med_list_non, rwd_perc_25_non, rwd_perc_75_non = perc(reward_list_non)

    # Iterations list
    iter_list = list(range(10, 2010, 10))

    plt.fill_between(iter_list, rwd_perc_25, rwd_perc_75, alpha=0.25, linewidth=2, color='#006BB2')
    plt.fill_between(iter_list, rwd_perc_25_non, rwd_perc_75_non, alpha=0.25, linewidth=2, color='#B22400')

    plt.plot(iter_list, rwd_med_list)
    plt.plot(iter_list, rwd_med_list_non)
    plt.legend(["Domain Randomization", "No Domain Randomization"])
    plt.title("Learning Curves (Arm)")
    plt.xlabel("Iteration")
    plt.ylabel("Expected Return")
    plt.show()

if choice == 2:
    # Boxplot the results using the second type of learning/testing
    plt.boxplot(reward_list)
    plt.title("Boxplot (Arm)")
    plt.xlabel("Box No.")
    plt.ylabel("Expected Return")
    plt.show()