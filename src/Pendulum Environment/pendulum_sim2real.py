# Libraries that are imported
import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from pylab import *
import numpy as np
import csv



class PendulumEnv(gym.Env):

    def __init__(self, rand_value, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self._step = 0

        # Randomization
        self.randomization = rand_value

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()


    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]

        # Reward function
        reward = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)


        newthdot = thdot + (-3*g/ (2*l) * np.sin(th+np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])


        self._step += 1
        done = False
        if self._step >= 200:
            done = True

        return self.get_obs(), -reward, done, {}


    def reset(self):
        # State Randomization
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)

        # Randomization
        self.randomize(randomization)

        self._step = 0
        return self.get_obs()


    def get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])


    def randomize(self, randomization):
        if randomization:
            # Randomize pendulum mass +-30%
            self.m = self.np_random.uniform(0.70, 1.30)
            # Randomize pendulum length +-30%
            self.l = self.np_random.uniform(0.70, 1.30)
        else:
            # Set fixed mass, length values +30%
            self.m = 1.30
            self.l = 1.30


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



class TestPendulumEnv(gym.Env):

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self._step = 0

        # Damping parameter
        self.b = 0.2

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()


    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        b = self.b

        u = np.clip(u, -self.max_torque, self.max_torque)[0]

        # Reward function
        reward = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)


        newthdot = thdot + (-3*g / (2*l) * np.sin(th+np.pi) + 3./(m*l**2) * (u-b*thdot)) * dt       # add damping to angular velocity
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])


        self._step += 1
        done = False
        if self._step >= 200:
            done = True

        return self.get_obs(), -reward, done, {}


    def reset(self):
        # State Randomization
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)

        self._step = 0
        return self.get_obs()


    def get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


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

        datafile = open("./logs/Randomized/data.csv", "a")
        for element in reward:
            datafile.write(str(element) + "\n")
        datafile.write("\n")
        datafile.close()


    if dir == "./logs/NonRandomized/monitor.csv":
        for row in csv_df:
            reward_non.append(float(row[0]))
        reward_list_non.append(reward_non)
        box_reward_list_non.append(reward_non[-1])

        datafile = open("./logs/NonRandomized/data.csv", "a")
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
    randomized_env = PendulumEnv(randomization)
    # Instantiate the real environment
    test_rand_env = TestPendulumEnv()
    # Monitor the real environment
    test_rand_mon = Monitor(test_rand_env, "./logs/Randomized/")

    # The noise objects for TD3
    n_actions = randomized_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    randomized_model = TD3("MlpPolicy", randomized_env, action_noise=action_noise, verbose=1)

    timesteps = int(500000)
    # For every 10 episodes of learning, test to the real environment (with domain randomization)
    randomized_model.learn(total_timesteps=timesteps, log_interval=500, eval_env=test_rand_mon, eval_freq=2000, n_eval_episodes=1, eval_log_path="./logs/Randomized/")



    # Instantiate the simulated environment without domain randomization
    randomization = False
    non_randomized_env = PendulumEnv(randomization)
    # Instantiate the real environment
    test_nonrand_env = TestPendulumEnv()
    # Monitor the real environment
    test_nonrand_mon = Monitor(test_nonrand_env, "./logs/NonRandomized/")

    # The noise objects for TD3
    n_actions = non_randomized_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    non_randomized_model = TD3("MlpPolicy", non_randomized_env, action_noise=action_noise, verbose=1)

    # For every 10 episodes of learning, test to the real environment (without domain randomization)
    non_randomized_model.learn(total_timesteps=timesteps, log_interval=500, eval_env=test_nonrand_mon, eval_freq=2000, n_eval_episodes=1, eval_log_path="./logs/NonRandomized/")


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
plt.title("Learning Curves (Pendulum)")
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

plt.title("Boxplots (Pendulum)")
plt.xlabel("Boxes")
plt.ylabel("Expected Return")
ax.set_xticklabels(["Randomized Model", "Non Randomized Model"])
plt.show()