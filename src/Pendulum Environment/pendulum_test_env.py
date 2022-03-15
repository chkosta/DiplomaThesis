from os import path
import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines3 import TD3
import numpy as np



class PendulumEnv(gym.Env):

    def __init__(self, rand_value, g=10.0):
        metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30}

        self.max_theta = 1.
        self.max_speed = 8.
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self._step = 0

        # Randomization
        self.randomization = rand_value

        # Visualization
        self.viewer = None

        # Spaces
        action_high = self.max_torque
        obs_high = np.array([self.max_theta, self.max_theta, self.max_speed], dtype=np.float32)

        self.action_space = spaces.Box(
            low=-action_high,
            high=action_high,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32
        )

        self.seed()


    def step(self, torque):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        torque = np.clip(torque, -self.max_torque, self.max_torque)[0]

        # Reward function
        reward = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (torque ** 2)


        self.last_u = True      # for rendering

        newthdot = thdot + (-3*g/ (2*l) * np.sin(th+np.pi) + 3./(m*l**2)*torque) * dt
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


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class TestPendulumEnv(gym.Env):

    def __init__(self, g=10.0):
        metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30}

        self.max_theta = 1.
        self.max_speed = 8.
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self._step = 0

        # Damping parameter
        self.b = 0.2

        # Visualization
        self.viewer = None

        # Spaces
        action_high = self.max_torque
        obs_high = np.array([self.max_theta, self.max_theta, self.max_speed], dtype=np.float32)

        self.action_space = spaces.Box(
            low=-action_high,
            high=action_high,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32
        )

        self.seed()


    def step(self, torque):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        b = self.b

        torque = np.clip(torque, -self.max_torque, self.max_torque)[0]

        # Reward function
        reward = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (torque ** 2)


        self.last_u = True      # for rendering

        newthdot = thdot + (-3*g / (2*l) * np.sin(th+np.pi) + 3./(m*l**2) * (torque-b*thdot)) * dt       # add damping to angular velocity
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


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)



randomization = True
env = PendulumEnv(randomization)

# Load the trained model to the same environment
model = TD3.load("./logs/Randomized/best_model.zip")

# Run the learned policy one time to see what happens
obs = env.reset()
episode_reward = 0.
for i in range(200):
    action, states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()

    episode_reward += reward
    if done:
        print("Reward:", episode_reward)
        episode_reward = 0.