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



class CustomPendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.viewer = None
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
        self.last_u = u  # for rendering
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
        self.m = self.np_random.uniform(0.7, 1.2)       # uniform sample from mass range
        self.l = self.np_random.uniform(0.7, 1.2)       # uniform sample from length range
        self.last_u = None
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
        self.viewer = None
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
        self.last_u = u  # for rendering
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
        self.last_u = None
        self._step = 0
        return self.get_obs()

    def get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)



# Create log directories
log_dir1 = "./logs/logs1/"
os.makedirs(log_dir1, exist_ok=True)
log_dir2 = "./logs/logs2/"
os.makedirs(log_dir2, exist_ok=True)

reward_list = []
tmps_list = []

# Run 10 times
for i in range(10):

    # Instantiate the simulated environment
    env = CustomPendulumEnv()
    # Instantiate the real environment and wrap it
    env_test = TestPendulumEnv()
    env_test1 = Monitor(env_test, log_dir1)
    env_test2 = Monitor(env_test, log_dir2)

    # Check for warnings
    # check_env(env)


    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)

    timesteps = int(50000)

    # For every 10 episodes of learning, test to the real environment
    # model.learn(total_timesteps=timesteps, log_interval=50, eval_env=env_test1, eval_freq=2000, n_eval_episodes=1, eval_log_path=log_dir1)

    # After 50000 steps of learning, test to the real environment
    model.learn(total_timesteps=timesteps, log_interval=50, eval_env=env_test2, eval_freq=50000, n_eval_episodes=10, eval_log_path=log_dir2)

    # Dataframe split to get only the important data (rewards, timesteps)
    df = open("./logs/logs2/monitor.csv")
    csv_df = csv.reader(df)
    next(csv_df)
    next(csv_df)
    reward = []
    tmps = []
    for row in csv_df:
        reward.append(float(row[0]))
        tmps.append(float(row[2]))

    # Calculate mean episode rewards (for boxplotting)
    reward = [np.mean(reward)]
    # Calculate mean episode timesteps (for boxplotting)
    tmps = [np.mean(tmps)]

    reward_list.append(reward)
    tmps_list.append(tmps)
    print(reward_list)

# # Calculate mean curve
# x_mean = np.mean(tmps_list, axis=0)
# y_mean = np.mean(reward_list, axis=0)
#
#
#
# # Plot the results with the first type of learning/testing
# plt.plot(x_mean, y_mean)
# plt.title("TD3 Pendulum")
# plt.xlabel("Timesteps")
# plt.ylabel("Episode Rewards")
# plt.show()

# Boxplot with the second type of learning/testing
plt.boxplot(reward_list)
plt.title("TD3 Pendulum")
plt.xlabel("Experiment No.")
plt.ylabel("Mean Episode Rewards")
plt.show()