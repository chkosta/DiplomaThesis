import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from scipy.spatial.distance import euclidean
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise
import RobotDART as rd



class RandomizedArmEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # RobotDART initialization data
        self.simu = rd.RobotDARTSimu(0.001)
        self.robot = rd.Robot("arm.urdf")
        self.robot.set_actuator_types("torque")  # Control each joint by giving velocity commands
        self.robot.fix_to_world()
        self.simu.add_robot(self.robot)
        self.simu.add_floor()

        # Rest initialization data
        self._step = 0
        self.max_velocity = 5.
        self.max_theta1 = np.radians(180)   # 3.1416
        self.max_theta_rest = np.radians(90)    # 1.5708

        # Spaces
        action_high = np.array([self.max_velocity, self.max_velocity, self.max_velocity, self.max_velocity], dtype=np.float32)
        obs_high = np.array([self.max_theta1, self.max_theta_rest, self.max_theta_rest, self.max_theta_rest], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-action_high,
            high=action_high,
            shape=(4,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            shape=(4,),
            dtype=np.float32
        )

        self.seed()

    def step(self, cmd):
        # Target joint positions
        target_pos = [0., 1.57, -0.5, 0.7]

        cmd = np.clip(cmd, -self.max_velocity, self.max_velocity)

        # Set velocity command
        self.robot.set_commands(cmd)

        for i in range(20):
            # Run one simulated step
            self.simu.step_world()

        # Calculate reward
        current_pos = np.round(self.robot.positions(), 2)
        costs = euclidean(current_pos, target_pos)
        reward = -costs

        self.state = current_pos

        self._step += 1
        done = False
        if self._step >= 250:
            done = True

        return self.state, reward, done, {}

    def reset(self):
        # Randomize initial joint positions
        self.state = self.generate_random_pos()
        self.robot.set_positions(self.state)

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
        mass_low = np.array([0.1, 0.1, 0.1, 0.1])
        mass_high = np.array([0.5, 0.5, 0.5, 0.5])
        mass = self.np_random.uniform(low=mass_low, high=mass_high)
        return mass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



# Instantiate the simulated environment with domain randomization
randomized_env = RandomizedArmEnv()

# Check for warnings
# check_env(env)


# The noise objects for TD3
n_actions = randomized_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

randomized_model = TD3("MlpPolicy", randomized_env, action_noise=action_noise, verbose=1)

timesteps = int(50000)
randomized_model.learn(total_timesteps=50000, log_interval=40)

