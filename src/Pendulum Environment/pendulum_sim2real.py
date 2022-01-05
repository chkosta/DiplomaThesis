import csv
import gym
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
import numpy as np



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


        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
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


        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * (u - b * thdot)) * dt       # add damping to angular velocity
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



    # # Instantiate the simulated environment with domain randomization
    # randomization = True
    # randomized_env = PendulumEnv(randomization)
    # # Instantiate the real environment
    # test_rand_env = TestPendulumEnv()
    # # Monitor the real environment
    # test_rand_mon = Monitor(test_rand_env, "./logs/Boxplot/")
    #
    #
    # # The noise objects for TD3
    # n_actions = randomized_env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    #
    # model = TD3("MlpPolicy", randomized_env, action_noise=action_noise, verbose=1)
    #
    # timesteps = int(500000)
    # # After 500000 steps of learning, test to the real environment (with domain randomization)
    # model.learn(total_timesteps=timesteps, log_interval=500, eval_env=test_rand_mon, eval_freq=500000, n_eval_episodes=1, eval_log_path="./logs/Boxplot/")


    # Dataframe split to get only the important data (rewards)
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


    # Iteration list
    iter_list = list(range(10, 2510, 10))

    plt.fill_between(iter_list, rwd_perc_25, rwd_perc_75, alpha=0.25, linewidth=2, color='#006BB2')
    plt.fill_between(iter_list, rwd_perc_25_non, rwd_perc_75_non, alpha=0.25, linewidth=2, color='#B22400')

    plt.plot(iter_list, rwd_med_list)
    plt.plot(iter_list, rwd_med_list_non)
    plt.legend(["Domain Randomization", "No Domain Randomization"])
    plt.title("Learning Curves (Pendulum)")
    plt.xlabel("Iteration")
    plt.ylabel("Expected Return")
    plt.show()

if choice == 2:
    # Boxplot the results using the second type of learning/testing
    plt.boxplot(reward_list)
    plt.title("Boxplot (Pendulum)")
    plt.xlabel("Box No.")
    plt.ylabel("Expected Return")
    plt.show()