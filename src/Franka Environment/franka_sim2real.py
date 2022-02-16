# Libraries that are imported
import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from pylab import *
import RobotDART as rd
import dartpy
import csv



class FrankaEnv(gym.Env):

    def __init__(self, viewer, rand_value):
        # RobotDART initialization data
        dt = 0.001
        self.simu = rd.RobotDARTSimu(dt)
        self.simu.set_collision_detector("fcl")

        # Load Franka
        packages = [("franka_description", "franka/franka_description")]
        self.robot = rd.Robot("franka/franka.urdf", packages)
        self.robot.set_color_mode("material")
        self.model = rd.Robot("franka/franka.urdf", packages)

        # Load box
        self.box_size = [0.04, 0.04, 0.04]
        self.tf = dartpy.math.Isometry3()
        self.tf.set_rotation(dartpy.math.eulerZYXToMatrix([0., 0., 0.]))
        self.tf.set_translation([0.4, 0.3, 0.02])
        self.box_pose = rd.math.logMap(self.tf.rotation()).tolist() + self.tf.translation().tolist()
        self.box = rd.Robot.create_box(self.box_size, self.box_pose, "free", mass=0.4, color=[0.1, 0.2, 0.9, 1.0], box_name="box")

        # Position Franka and box
        self.robot.set_actuator_types("torque")                                                                   # Control each joint by giving torque commands
        self.robot.set_actuator_types("servo", ["panda_joint7", "panda_finger_joint1", "panda_finger_joint2"])    # Control gripper joints by giving velocity commands
        self.robot.fix_to_world()
        self.model.fix_to_world()
        self.simu.add_robot(self.robot)
        self.simu.add_robot(self.box)
        self.simu.add_floor(floor_width=40.0)

        # Visualization
        self.visualize(viewer)

        # Randomization
        self.randomization = rand_value

        # Rest initialization data
        self._step = 0

        # Limits for end effector
        self.eef_min_vel = -2.
        self.eef_max_vel = 2.
        self.finger_min_vel = -0.2
        self.finger_max_vel = 0.2
        self.eef_min_pos_x = -0.855
        self.eef_max_pos_x = 0.855
        self.eef_min_pos_y = -0.855
        self.eef_max_pos_y = 0.855
        self.eef_min_pos_z = 0.
        self.eef_max_pos_z = 1.190
        # Limits for box
        self.box_min_pos_x = -2.5
        self.box_max_pos_x = 2.5
        self.box_min_pos_y = -2.5
        self.box_max_pos_y = 2.5
        self.box_min_pos_z = 0.02
        self.box_max_pos_z = 1.190

        # Spaces
        action_low = np.array([self.eef_min_vel, self.eef_min_vel, self.eef_min_vel, self.finger_min_vel], dtype=np.float32)
        action_high = np.array([self.eef_max_vel, self.eef_max_vel, self.eef_max_vel, self.finger_max_vel], dtype=np.float32)

        obs_low = np.array([[self.eef_min_pos_x, self.eef_min_pos_y, self.eef_min_pos_z], [self.eef_min_vel, self.eef_min_vel, self.eef_min_vel],
                            [self.box_min_pos_x, self.box_min_pos_y, self.box_min_pos_z]], dtype=np.float32)
        obs_high = np.array([[self.eef_max_pos_x, self.eef_max_pos_y, self.eef_max_pos_z], [self.eef_max_vel, self.eef_max_vel, self.eef_max_vel],
                             [self.box_max_pos_x, self.box_max_pos_y, self.box_max_pos_z]], dtype=np.float32)

        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            shape=(4,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        self.seed()

    def step(self, vel2):

        for i in range(20):
            # At each step get the last end effector pose
            last_eef_pose = self.robot.body_pose(self.eef_link_name)

            vel1 = self.controller.update(last_eef_pose)
            vel = [vel1[0], vel1[1], vel1[2], vel2[0], vel2[1], vel2[2]]
            jac = self.robot.jacobian(self.eef_link_name)
            jac_pinv = damped_pseudoinverse(jac)
            cmd = jac_pinv @ vel

            fingers_vel = vel2[3]
            cmd[7] = fingers_vel
            cmd[8] = 0.


            self.model.set_positions(self.robot.positions())
            self.model.set_velocities(self.robot.velocities())

            # Gravity Compensation to achieve equilibrium
            c = self.model.coriolis_gravity_forces(["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6"])

            grav_comp_cmds = [cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5]]
            grav_comp_cmds = grav_comp_cmds + c

            for i in range(6):
                cmd[i] = grav_comp_cmds[i]

            self.robot.set_commands(cmd)

            # Run one simulated step
            self.simu.step_world()


        # At each step get the current end effector position and velocity
        current_eef_pos = self.robot.body_pose(self.eef_link_name).translation()

        current_eef_vel = self.robot.body_velocity(self.eef_link_name)
        current_eef_vel = np.array([current_eef_vel[3], current_eef_vel[4], current_eef_vel[5]])

        # At each step get the current box position
        current_box_pos = self.box.base_pose_vec()
        current_box_pos = np.array([current_box_pos[3], current_box_pos[4], current_box_pos[5]])

        self.state = np.array([current_eef_pos, current_eef_vel, current_box_pos])


        # Reward function
        reward = -0.1 * np.linalg.norm(current_eef_pos-current_box_pos)

        # if cube is lifted
        if current_box_pos[2] > 0.05:
            reward += 5.0                                                   # bonus for lifting the cube
            reward += -0.5*np.linalg.norm(current_eef_pos-self.target_pos)  # make gripper go to target
            reward += -0.5*np.linalg.norm(current_box_pos-self.target_pos)  # make cube go to target

        # BONUS
        if np.linalg.norm(current_box_pos-self.target_pos) < 0.15:
            reward += 10.0                                                  # bonus for cube close to target
        if np.linalg.norm(current_box_pos-self.target_pos) < 0.1:
            reward += 20.0                                                  # bonus for cube very close to target
        if np.linalg.norm(current_box_pos-self.target_pos) < 0.05:
            reward += 30.0                                                  # bonus for cube extremely close to target


        self._step += 1
        done = False
        if self._step >= 500:
            done = True

        return self.state, reward, done, {}

    def reset(self):
        # Robot initial joint positions (fingers are close)
        joint_pos = [0., 0., 0., -1.5708, 0., 1.5708, 0.8, 0., 0.]

        # Reset robot and box
        self.robot.reset()
        self.robot.set_positions(joint_pos)

        self.box.reset()
        self.box.set_base_pose(self.tf)

        # Get eef target pose
        self.eef_link_name = "panda_hand"
        self.eef_target_pose = self.robot.body_pose(self.eef_link_name)

        # Reset controller
        self.controller = PITask(self.eef_target_pose)

        eef_pos = self.robot.body_pose(self.eef_link_name).translation()

        eef_vel = self.robot.body_velocity(self.eef_link_name)
        eef_vel = np.array([eef_vel[3], eef_vel[4], eef_vel[5]])

        box_pos = self.box.base_pose_vec()
        box_pos = np.array([box_pos[3], box_pos[4], box_pos[5]])

        self.target_pos = box_pos + [0., 0., 0.18]

        self.state = np.array([eef_pos, eef_vel, box_pos])
        self._step = 0

        # Randomization
        self.randomize(self.randomization)

        return self.state

    def visualize(self, viewer):
        if viewer:
            # Create graphics
            gconfig = rd.gui.GraphicsConfiguration(720, 480)
            graphics = rd.gui.Graphics(gconfig)
            self.simu.set_graphics(graphics)
            graphics.look_at([2., 0., 1.])
            self.simu.scheduler().set_sync(False)

            graphics.camera().record(True)
            graphics.record_video("sac_franka.mp4")

    def randomize(self, randomization):
        if randomization:
            # Randomize franka link's masses +-15%
            franka_mass_low = np.array([2.60, 1.99, 2.01, 2.02, 2.07, 2.97, 1.25])
            franka_mass_high = np.array([3.52, 2.69, 2.71, 2.74, 2.79, 4.03, 1.69])
            franka_mass = self.np_random.uniform(low=franka_mass_low, high=franka_mass_high)

            self.robot.set_body_mass("panda_link0", franka_mass[0])
            self.robot.set_body_mass("panda_link1", franka_mass[1])
            self.robot.set_body_mass("panda_link2", franka_mass[2])
            self.robot.set_body_mass("panda_link3", franka_mass[3])
            self.robot.set_body_mass("panda_link4", franka_mass[4])
            self.robot.set_body_mass("panda_link5", franka_mass[5])
            self.robot.set_body_mass("panda_link6", franka_mass[6])

            # Randomize box mass +-15%
            box_mass = self.np_random.uniform(low=0.34, high=0.46)
            self.box.set_body_mass("box", box_mass)
        else:
            # Set fixed values +15%
            self.robot.set_body_mass("panda_link0", 3.52)   # 3.06
            self.robot.set_body_mass("panda_link1", 2.69)   # 2.34
            self.robot.set_body_mass("panda_link2", 2.71)   # 2.36
            self.robot.set_body_mass("panda_link3", 2.74)   # 2.38
            self.robot.set_body_mass("panda_link4", 2.79)   # 2.43
            self.robot.set_body_mass("panda_link5", 4.03)   # 3.50
            self.robot.set_body_mass("panda_link6", 1.69)   # 1.47

            self.box.set_body_mass("box", 0.46)             # 0.40

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class TestFrankaEnv(gym.Env):

    def __init__(self, viewer):
        # RobotDART initialization data
        dt = 0.001
        self.simu = rd.RobotDARTSimu(dt)
        self.simu.set_collision_detector("fcl")

        # Load Franka
        packages = [("franka_description", "franka/franka_description")]
        self.robot = rd.Robot("franka/franka.urdf", packages)
        self.robot.set_color_mode("material")
        self.model = rd.Robot("franka/franka.urdf", packages)

        # Load box
        self.box_size = [0.04, 0.04, 0.04]
        self.tf = dartpy.math.Isometry3()
        self.tf.set_rotation(dartpy.math.eulerZYXToMatrix([0., 0., 0.]))
        self.tf.set_translation([0.4, 0.3, 0.02])
        self.box_pose = rd.math.logMap(self.tf.rotation()).tolist() + self.tf.translation().tolist()
        self.box = rd.Robot.create_box(self.box_size, self.box_pose, "free", mass=0.4, color=[0.1, 0.2, 0.9, 1.0], box_name="box")

        # Position Franka and box
        self.robot.set_actuator_types("torque")                                                                   # Control each joint by giving torque commands
        self.robot.set_actuator_types("servo", ["panda_joint7", "panda_finger_joint1", "panda_finger_joint2"])    # Control gripper joints by giving velocity commands
        self.robot.fix_to_world()
        self.model.fix_to_world()
        self.simu.add_robot(self.robot)
        self.simu.add_robot(self.box)
        self.simu.add_floor(floor_width=40.0)

        # Visualization
        self.visualize(viewer)

        # Rest initialization data
        self._step = 0
        self.rand_success_rate = []
        self.nonrand_success_rate = []

        # Limits for end effector
        self.eef_min_vel = -2.
        self.eef_max_vel = 2.
        self.finger_min_vel = -0.2
        self.finger_max_vel = 0.2
        self.eef_min_pos_x = -0.855
        self.eef_max_pos_x = 0.855
        self.eef_min_pos_y = -0.855
        self.eef_max_pos_y = 0.855
        self.eef_min_pos_z = 0.
        self.eef_max_pos_z = 1.190
        # Limits for box
        self.box_min_pos_x = -2.5
        self.box_max_pos_x = 2.5
        self.box_min_pos_y = -2.5
        self.box_max_pos_y = 2.5
        self.box_min_pos_z = 0.02
        self.box_max_pos_z = 1.190

        # Spaces
        action_low = np.array([self.eef_min_vel, self.eef_min_vel, self.eef_min_vel, self.finger_min_vel], dtype=np.float32)
        action_high = np.array([self.eef_max_vel, self.eef_max_vel, self.eef_max_vel, self.finger_max_vel], dtype=np.float32)

        obs_low = np.array([[self.eef_min_pos_x, self.eef_min_pos_y, self.eef_min_pos_z], [self.eef_min_vel, self.eef_min_vel, self.eef_min_vel],
                            [self.box_min_pos_x, self.box_min_pos_y, self.box_min_pos_z]], dtype=np.float32)
        obs_high = np.array([[self.eef_max_pos_x, self.eef_max_pos_y, self.eef_max_pos_z], [self.eef_max_vel, self.eef_max_vel, self.eef_max_vel],
                             [self.box_max_pos_x, self.box_max_pos_y, self.box_max_pos_z]], dtype=np.float32)

        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            shape=(4,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        self.seed()

    def step(self, vel2):

        for i in range(20):
            # At each step get the last end effector pose
            last_eef_pose = self.robot.body_pose(self.eef_link_name)

            vel1 = self.controller.update(last_eef_pose)
            vel = [vel1[0], vel1[1], vel1[2], vel2[0], vel2[1], vel2[2]]
            jac = self.robot.jacobian(self.eef_link_name)
            jac_pinv = damped_pseudoinverse(jac)
            cmd = jac_pinv @ vel

            fingers_vel = vel2[3]
            cmd[7] = fingers_vel
            cmd[8] = 0.


            self.model.set_positions(self.robot.positions())
            self.model.set_velocities(self.robot.velocities())

            # Gravity Compensation to achieve equilibrium
            c = self.model.coriolis_gravity_forces(["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6"])

            grav_comp_cmds = [cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5]]
            grav_comp_cmds = grav_comp_cmds + c

            for i in range(6):
                cmd[i] = grav_comp_cmds[i]

            self.robot.set_commands(cmd)

            # Run one simulated step
            self.simu.step_world()


        # At each step get the current end effector position and velocity
        current_eef_pos = self.robot.body_pose(self.eef_link_name).translation()

        current_eef_vel = self.robot.body_velocity(self.eef_link_name)
        current_eef_vel = np.array([current_eef_vel[3], current_eef_vel[4], current_eef_vel[5]])

        # At each step get the current box position
        current_box_pos = self.box.base_pose_vec()
        current_box_pos = np.array([current_box_pos[3], current_box_pos[4], current_box_pos[5]])

        self.state = np.array([current_eef_pos, current_eef_vel, current_box_pos])


        # Reward function
        reward = -0.1 * np.linalg.norm(current_eef_pos-current_box_pos)

        # if cube is lifted
        if current_box_pos[2] > 0.05:
            reward += 5.0                                                   # bonus for lifting the cube
            reward += -0.5*np.linalg.norm(current_eef_pos-self.target_pos)  # make gripper go to target
            reward += -0.5*np.linalg.norm(current_box_pos-self.target_pos)  # make cube go to target

        # BONUS
        if np.linalg.norm(current_box_pos-self.target_pos) < 0.15:
            reward += 10.0                                                  # bonus for cube close to target
        if np.linalg.norm(current_box_pos-self.target_pos) < 0.1:
            reward += 20.0                                                  # bonus for cube very close to target
        if np.linalg.norm(current_box_pos-self.target_pos) < 0.05:
            reward += 30.0                                                  # bonus for cube extremely close to target


        self._step += 1
        done = False
        if self._step >= 500:
            done = True

        return self.state, reward, done, {}

    def reset(self):
        # Robot initial joint positions (fingers are close)
        joint_pos = [0., 0., 0., -1.5708, 0., 1.5708, 0.8, 0., 0.]

        # Reset robot and box
        self.robot.reset()
        self.robot.set_positions(joint_pos)

        self.box.reset()
        self.box.set_base_pose(self.tf)

        # Get eef target pose
        self.eef_link_name = "panda_hand"
        self.eef_target_pose = self.robot.body_pose(self.eef_link_name)

        # Reset controller
        self.controller = PITask(self.eef_target_pose)

        eef_pos = self.robot.body_pose(self.eef_link_name).translation()

        eef_vel = self.robot.body_velocity(self.eef_link_name)
        eef_vel = np.array([eef_vel[3], eef_vel[4], eef_vel[5]])

        box_pos = self.box.base_pose_vec()
        box_pos = np.array([box_pos[3], box_pos[4], box_pos[5]])

        self.target_pos = box_pos + [0., 0., 0.18]

        self.state = np.array([eef_pos, eef_vel, box_pos])
        self._step = 0

        return self.state

    def visualize(self, viewer):
        if viewer:
            # Create graphics
            gconfig = rd.gui.GraphicsConfiguration(720, 480)
            graphics = rd.gui.Graphics(gconfig)
            self.simu.set_graphics(graphics)
            graphics.look_at([2., 0., 1.])
            self.simu.scheduler().set_sync(False)

            graphics.camera().record(True)
            graphics.record_video("sac_franka.mp4")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class PITask:
    def __init__(self, target, dt=0.001, Kp=300., Ki=2.):
        self._target = target
        self._dt = dt
        self._Kp = Kp
        self._Ki = Ki
        self._sum_error = 0

    def set_target(self, target):
        self._target = target

    def error(self, tf):
        rot_error = rd.math.logMap(self._target.rotation() @ tf.rotation().T)
        lin_error = self._target.translation() - tf.translation()
        return np.r_[rot_error, lin_error]

    def update(self, current):
        error_in_world_frame = self.error(current)
        self._sum_error = self._sum_error + error_in_world_frame * self._dt
        return self._Kp * error_in_world_frame + self._Ki * self._sum_error


def damped_pseudoinverse(jac, l=0.01):
    m, n = jac.shape
    if n >= m:
        return jac.T @ np.linalg.inv(jac @ jac.T + l * l * np.eye(m))
    return np.linalg.inv(jac.T @ jac + l * l * np.eye(n)) @ jac.T


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
    viewer, randomization = False, True
    randomized_env = FrankaEnv(viewer, randomization)
    # Instantiate the real environment
    test_rand_env = TestFrankaEnv(viewer)
    # Monitor the real environment
    test_rand_mon = Monitor(test_rand_env, "./logs/Randomized/")

    # The noise objects for SAC
    n_actions = randomized_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Save a checkpoint every 250000 steps
    checkpoint_callback = CheckpointCallback(save_freq=250000, save_path="./logs/Randomized/")
    randomized_model = SAC(policy="MlpPolicy", env=randomized_env, action_noise=action_noise, verbose=1, device="cuda")

    timesteps = int(5000000)
    # For every 40 episodes of learning, test to the real environment (with domain randomization)
    randomized_model.learn(total_timesteps=timesteps, callback=checkpoint_callback, log_interval=100, eval_env=test_rand_mon, eval_freq=20000, n_eval_episodes=1, eval_log_path="./logs/Randomized/")



    # Instantiate the simulated environment without domain randomization
    viewer, randomization = False, False
    non_randomized_env = FrankaEnv(viewer, randomization)
    # Instantiate the real environment
    test_nonrand_env = TestFrankaEnv(viewer)
    # Monitor the real environment
    test_nonrand_mon = Monitor(test_nonrand_env, "./logs/NonRandomized/")

    # The noise objects for SAC
    n_actions = non_randomized_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Save a checkpoint every 250000 steps
    checkpoint_callback = CheckpointCallback(save_freq=250000, save_path="./logs/NonRandomized/")
    non_randomized_model = SAC(policy="MlpPolicy", env=non_randomized_env, action_noise=action_noise, verbose=1, device="cuda")

    # For every 40 episodes of learning, test to the real environment (without domain randomization)
    non_randomized_model.learn(total_timesteps=timesteps, callback=checkpoint_callback, log_interval=100, eval_env=test_nonrand_mon, eval_freq=20000, n_eval_episodes=1, eval_log_path="./logs/NonRandomized/")


    # Dataframe split to get only the important data (rewards)
    load("./logs/Randomized/monitor.csv")
    load("./logs/NonRandomized/monitor.csv")



# Plot the results using learning curves
# Compute the median and 25/75 percentiles with domain randomization
rand_med_list, rand_perc_25, rand_perc_75 = perc(reward_list)

# Compute the median and 25/75 percentiles without domain randomization
nonrand_med_list, nonrand_perc_25, nonrand_perc_75 = perc(reward_list_non)

# Iteration list
iter_list = list(range(40, 10040, 40))

plt.fill_between(iter_list, rand_perc_25, rand_perc_75, alpha=0.25, linewidth=2, color='#006BB2')
plt.fill_between(iter_list, nonrand_perc_25, nonrand_perc_75, alpha=0.25, linewidth=2, color='#B22400')

plt.plot(iter_list, rand_med_list)
plt.plot(iter_list, nonrand_med_list)
plt.legend(["Randomized Model", "Non Randomized Model"])
plt.title("Learning Curves (Franka Emika)")
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

plt.title("Boxplots (Franka Emika)")
plt.xlabel("Boxes")
plt.ylabel("Expected Return")
ax.set_xticklabels(["Randomized Model", "Non Randomized Model"])
plt.show()