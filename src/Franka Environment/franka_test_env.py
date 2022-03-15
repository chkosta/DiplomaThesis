import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines3 import SAC
import dartpy
import RobotDART as rd
import numpy as np



class FrankaEnv(gym.Env):

    def __init__(self, rand_value, viewer):
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

        # Randomization
        self.randomization = rand_value

        # Visualization
        self.visualize(viewer)

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

    def visualize(self, viewer):
        if viewer:
            # Create graphics
            gconfig = rd.gui.GraphicsConfiguration(720, 480)
            graphics = rd.gui.Graphics(gconfig)
            self.simu.set_graphics(graphics)
            graphics.look_at([2., 0., 1.])
            self.simu.scheduler().set_sync(True)

            graphics.camera().record(True)
            graphics.record_video("./videos/Source Environment/randomized_model.mp4")

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
        self.box = rd.Robot.create_box(self.box_size, self.box_pose, "free", mass=0.4, color=[0.1, 0.2, 0.9, 1.0])

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

        # Limits
        # End effector
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
        # Βox
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
            self.simu.scheduler().set_sync(True)

            graphics.camera().record(True)
            graphics.record_video("./videos/Target Environment/randomized_model.mp4")

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



# Visualization
rand_value = True
viewer = True
env = FrankaEnv(viewer, rand_value)

# Load the trained model to the same environment
model = SAC.load("./logs/Randomized/rl_model_5000000_steps.zip")

# Run the learned policy one time to see what happens
obs = env.reset()
episode_reward = 0.
for i in range(500):
    action, states = model.predict(obs)
    obs, reward, done, info = env.step(action)

    episode_reward += reward
    if done:
        print("Reward:", episode_reward)
        episode_reward = 0.