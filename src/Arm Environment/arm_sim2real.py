# Libraries that are imported
import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import RobotDART as rd
from pylab import *
import numpy as np
import csv



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
        box_reward_list.append(reward[-1])

        datafile = open("./logs/Randomized/data_target.csv", "a")
        for element in reward:
            datafile.write(str(element) + "\n")
        datafile.write("\n")
        datafile.close()


    if dir == "./logs/NonRandomized/monitor.csv":
        for row in csv_df:
            reward_non.append(float(row[0]))
        reward_list_non.append(reward_non)
        box_reward_list_non.append(reward_non[-1])

        datafile = open("./logs/NonRandomized/data_target.csv", "a")
        for element in reward_non:
            datafile.write(str(element) + "\n")
        datafile.write("\n")
        datafile.close()


def perc(reward_list):
    rwd_med_list = np.median(reward_list, axis=0)
    rwd_perc_25 = np.percentile(reward_list, 25, axis=0)
    rwd_perc_75 = np.percentile(reward_list, 75, axis=0)
    return rwd_med_list, rwd_perc_25, rwd_perc_75



reward_list = []
reward_list_non = []
box_reward_list = []
box_reward_list_non = []

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

    randomized_model = TD3("MlpPolicy", randomized_env, action_noise=action_noise, verbose=1, device="cuda")

    timesteps = int(625000)
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

    non_randomized_model = TD3("MlpPolicy", non_randomized_env, action_noise=action_noise, verbose=1, device="cuda")

    # For every 10 episodes of learning, test to the real environment (without domain randomization)
    non_randomized_model.learn(total_timesteps=timesteps, log_interval=500, eval_env=test_nonrand_mon, eval_freq=2500, n_eval_episodes=1, eval_log_path="./logs/NonRandomized/")


    # Dataframe split to get only the important data (rewards)
    load("./logs/Randomized/monitor.csv")
    load("./logs/NonRandomized/monitor.csv")



# Plot the results using learning curves
# Compute the median and 25/75 percentiles with domain randomization
rwd_med_list, rwd_perc_25, rwd_perc_75 = perc(reward_list)

# Compute the median and 25/75 percentiles without domain randomization
rwd_med_list_non, rwd_perc_25_non, rwd_perc_75_non = perc(reward_list_non)

# Iteration list
iter_list = list(range(10, 2510, 10))

plt.fill_between(iter_list, rwd_perc_25, rwd_perc_75, alpha=0.25, linewidth=2, color='#006BB2')
plt.fill_between(iter_list, rwd_perc_25_non, rwd_perc_75_non, alpha=0.25, linewidth=2, color='#B22400')

plt.plot(iter_list, rwd_med_list)
plt.plot(iter_list, rwd_med_list_non)
plt.legend(["Randomized Model", "Non Randomized Model"])
plt.title("Learning Curves (Arm)")
plt.xlabel("Iteration")
plt.ylabel("Expected Return")
plt.show()


# Plot the results using boxplots
fig = figure()
ax = fig.add_subplot()
bp = ax.boxplot([box_reward_list, box_reward_list_non])

colors = ['#43A2CA', '#FDAE6B']
for i in range(0, len(bp['boxes'])):
    bp['boxes'][i].set_color(colors[i])
    bp['whiskers'][i * 2].set_color(colors[i])
    bp['whiskers'][i * 2 + 1].set_color(colors[i])
    bp['whiskers'][i * 2].set_linewidth(2)
    bp['whiskers'][i * 2 + 1].set_linewidth(2)
    bp['fliers'][i].set(markerfacecolor=colors[i], marker='o', alpha=0.75, markersize=6, markeredgecolor='none')
    bp['medians'][i].set_color('black')
    bp['medians'][i].set_linewidth(3)
    for c in bp['caps']:
        c.set_linewidth(2)
for i in range(len(bp['boxes'])):
    box = bp['boxes'][i]
    box.set_linewidth(0)
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        boxPolygon = Polygon(boxCoords, facecolor=colors[i], linewidth=0)
        ax.add_patch(boxPolygon)

plt.title("Boxplots (Arm)")
plt.xlabel("Boxes")
plt.ylabel("Expected Return")
ax.set_xticklabels(["Randomized Model", "Non Randomized Model"])
plt.show()