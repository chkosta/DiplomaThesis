import numpy as np
import RobotDART as rd
import dartpy # OSX breaks if this is imported before RobotDART

########## Create simulator object ##########
time_step = 0.001
simu = rd.RobotDARTSimu(time_step)
# simulate for 20 seconds only
simulation_time = 20.0

########## Create Graphics ##########
graphics = rd.gui.Graphics()
simu.set_graphics(graphics)
graphics.look_at([0., 1.8, 2.], [0., -0.5, 0.])

########## Create 1st robot arm ##########
robot1 = rd.Robot("arm.urdf", robot_name="robot_1")
robot1.set_position_enforced(True) # If True, limits of joints are not violated
robot1.set_actuator_types("servo") # Control each joint by giving velocity commands
robot1.fix_to_world()
tf = dartpy.math.Isometry3() # identity transformation -> no translation/rotation
robot1.set_base_pose(tf)
robot1.set_positions([0., 0., 0., np.pi/2.]) # Set the initial position of the joints
dofs1 = robot1.dof_names(True, True, True) # Get names of controllable DoFs
robot1_links = robot1.body_names()
robot1.set_draw_axis(robot1_links[-1], 0.25)
simu.add_robot(robot1)

print("Number of DoFs of robot arm: " + str(len(dofs1)))
print(dofs1)

########### Create 2nd robot arm ##########
robot2 = rd.Robot("arm.urdf", robot_name="robot_2")
robot2.set_position_enforced(True)
robot2.set_actuator_types("servo")
robot2.fix_to_world()
# Create tranformation matrix to translate robot2
tf = dartpy.math.Isometry3()
tf.set_translation([0., -0.5, 0.])
robot2_pose = rd.math.logMap(tf.rotation()).tolist() + tf.translation().tolist()
robot2.set_base_pose(robot2_pose)
robot2.set_positions([0., 0., 0., np.pi/2.])
dofs2 = robot2.dof_names(True, True, True)
robot2_links = robot2.body_names()
robot2.set_draw_axis(robot2_links[-1], 0.25)
simu.add_robot(robot2)

########## Create 3rd robot arm ##########
robot3 = rd.Robot("arm.urdf", robot_name="robot_3")
robot3.set_position_enforced(True)
robot3.set_actuator_types("servo")
robot3.fix_to_world()
# Create tranformation matrix to translate robot3
tf = dartpy.math.Isometry3()
tf.set_translation([0., -1., 0.])
robot3_pose = rd.math.logMap(tf.rotation()).tolist() + tf.translation().tolist()
robot3.set_base_pose(robot3_pose)
robot3.set_positions([0., 0., 0., np.pi/2.])
dofs3 = robot3.dof_names(True, True, True)
robot3_links = robot3.body_names()
robot3.set_draw_axis(robot3_links[-1], 0.25)
simu.add_robot(robot3)

########## Controllers (not really :P) ##########
previous_position1 = np.round(robot1.positions()[3], 2)
previous_position2 = np.round(robot2.positions()[3], 2)
previous_position3 = np.round(robot3.positions()[3], 2)
cmds1 = [0., 0., 0., -1.] # Velocity commands for each controllable DoF of robot1
cmds2 = [0., 0., 0., -2.]
cmds3 = [0., 0., 0., -4.]

# the commands are NOT reset, no need to set them at every time-step
robot1.set_commands(cmds1)
robot2.set_commands(cmds2)
robot3.set_commands(cmds3)

########## Add floor ##########
simu.add_floor()

########## Run simulation ##########
while simu.scheduler().next_time() < simulation_time:
    if (np.round(robot1.positions()[3], 2) == -previous_position1):
        cmds1[3] = -cmds1[3]
        previous_position1 = np.round(robot1.positions()[3], 2)
        robot1.set_commands(cmds1)

    if (np.round(robot2.positions()[3], 2) == -previous_position2):
        cmds2[3] = -cmds2[3]
        previous_position2 = np.round(robot2.positions()[3], 2)
        robot2.set_commands(cmds2)

    if (np.round(robot3.positions()[3], 2) == -previous_position3):
        cmds3[3] = -cmds3[3]
        previous_position3 = np.round(robot3.positions()[3], 2)
        robot3.set_commands(cmds3)

    if (simu.step_world()):
        break