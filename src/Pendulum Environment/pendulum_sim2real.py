import csv
import os
from os import path
import gym
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import numpy as np



class RandomizedPendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self._step = 0

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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        self._step += 1
        done = False
        if self._step >= 200:
            done = True
        return self.get_obs(), -costs, done, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.m = self.np_random.uniform(0.7, 1.3)       # uniform sample from mass range
        self.l = self.np_random.uniform(0.7, 1.3)       # uniform sample from length range
        self._step = 0
        return self.get_obs()

    def get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])


class NonRandomizedPendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        # Fixed mass, length values
        self.m = 1.3
        self.l = 1.3
        self._step = 0

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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        self._step += 1
        done = False
        if self._step >= 200:
            done = True
        return self.get_obs(), -costs, done, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self._step = 0
        return self.get_obs()

    def get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])


class TestPendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self._step = 0
        self.b = 0.2        # damping parameter

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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        b = self.b

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * (u - b * thdot)) * dt       # add damping to angular velocity
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        self._step += 1
        done = False
        if self._step >= 200:
            done = True
        return self.get_obs(), -costs, done, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self._step = 0
        return self.get_obs()

    def get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])


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
    randomized_env = RandomizedPendulumEnv()

    # Instantiate the simulated environment without domain randomization
    non_randomized_env = NonRandomizedPendulumEnv()

    # Instantiate the real environment and wrap it
    env_test = TestPendulumEnv()
    # env_test1 = Monitor(env_test, "./logs/Randomized/")
    # env_test2 = Monitor(env_test, "./logs/NonRandomized/")
    env_test3 = Monitor(env_test, "./logs/Boxplot/")

    # Check for warnings
    # check_env(randomized_env)


    # The noise objects for TD3
    n_actions = randomized_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # randomized_model = TD3("MlpPolicy", randomized_env, action_noise=action_noise, verbose=1)
    # non_randomized_model = TD3("MlpPolicy", non_randomized_env, action_noise=action_noise, verbose=1)
    model = TD3("MlpPolicy", randomized_env, action_noise=action_noise, verbose=1)

    timesteps = int(500000)

    # # For every 10 episodes of learning, test to the real environment (with domain randomization)
    # randomized_model.learn(total_timesteps=timesteps, log_interval=500, eval_env=env_test1, eval_freq=2000, n_eval_episodes=1, eval_log_path="./logs/Randomized/")
    # # For every 10 episodes of learning, test to the real environment (without domain randomization)
    # non_randomized_model.learn(total_timesteps=timesteps, log_interval=500, eval_env=env_test2, eval_freq=2000, n_eval_episodes=1, eval_log_path="./logs/NonRandomized/")

    # After 500000 steps of learning, test to the real environment
    model.learn(total_timesteps=timesteps, log_interval=500, eval_env=env_test3, eval_freq=500000, n_eval_episodes=1, eval_log_path="./logs/Boxplot/")


    # Dataframe split to get only the important data (rewards)
    # choice = 1
    # dir_randomized = "./logs/Randomized/monitor.csv"
    # load(dir_randomized)
    # dir_non_randomized = "./logs/NonRandomized/monitor.csv"
    # load(dir_non_randomized)

    choice = 2
    dir = "./logs/Boxplot/monitor.csv"
    load(dir)



if choice == 1:
    # Plot the results using the first type of learning/testing
    # Compute the median and 25/75 percentiles with domain randomization
    rwd_med_list, rwd_perc_25, rwd_perc_75 = perc(reward_list)

    # Compute the median and 25/75 percentiles without domain randomization
    rwd_med_list_non, rwd_perc_25_non, rwd_perc_75_non = perc(reward_list_non)


    # Timesteps list
    tmps_list = list(range(2000, timesteps+2000, 2000))

    plt.fill_between(tmps_list, rwd_perc_25, rwd_perc_75, alpha=0.25, linewidth=2, color='#006BB2')
    plt.fill_between(tmps_list, rwd_perc_25_non, rwd_perc_75_non, alpha=0.25, linewidth=2, color='#B22400')

    plt.plot(tmps_list, rwd_med_list)
    plt.plot(tmps_list, rwd_med_list_non)
    plt.legend(["Domain Randomization", "No Domain Randomization"])
    plt.title("TD3 Pendulum")
    plt.xlabel("Timesteps")
    plt.ylabel("Median Rewards")
    plt.show()

else:
    # Boxplot the results using the second type of learning/testing
    plt.boxplot(reward_list)
    plt.title("TD3 Pendulum")
    plt.xlabel("Box No.")
    plt.ylabel("Median Rewards")
    plt.show()
