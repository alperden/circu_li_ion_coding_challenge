import numpy as np
import imageio
import robosuite
from robosuite.controllers import load_controller_config
from robosuite.robots import Bimanual
from robosuite.models.grippers import GripperModel
import robosuite.macros as macros

# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"
# create a video writer with imageio
writer = imageio.get_writer("video.mp4", fps=20)
frames = []

print("Coding Challenge: Simulating Autonomous Battery Disassembly")

# load default controller parameters for Operational Space Control (OSC)
controller_name = "OSC_POSE"
controller_config = load_controller_config(default_controller = controller_name)

# Define the pre-defined controller actions to use (action_dim, num_test_steps, test_value)
controller_settings = {
    "OSC_POSE": [6, 6, 0.1],
    "IK_POSE": [6, 6, 0.01],
}

# Define variables for each controller test
action_dim = controller_settings[controller_name][0]
num_test_steps = controller_settings[controller_name][1]
test_value = controller_settings[controller_name][2]

# Define the number of timesteps to use per controller action as well as timesteps in between actions
steps_per_action = 75
steps_per_rest = 75

# create an environment to visualize on-screen
env = robosuite.make(
    "Lift",
    robots="IIWA",                          # load robot
    gripper_types="Robotiq140Gripper",      # load gripper
    controller_configs=controller_config,   # each arm is controlled using OSC
    env_configuration="single-arm-opposed", # single or two-arm selection
    has_renderer=True,                      # on-screen rendering
    render_camera="frontview",              # visualize the "frontview" camera 'frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'
    has_offscreen_renderer=True,           # no off-screen rendering
    ignore_done=True,
    control_freq=20,                        # 20 hz control for applied actions
    horizon=(steps_per_action + steps_per_rest) * num_test_steps,
    use_object_obs=True,                   # no observations needed
    use_camera_obs=False,                   # no observations needed
    camera_names="agentview",
    camera_heights=512,
    camera_widths=512,
)

# reset the environment
env.reset()
# env.viewer.set_camera(camera_id=0)
observables = env._setup_observables()

# z
# |
# |___ y  front view (x points outside of screen)
differ = env.model.mujoco_arena.table_offset - env._eef_xpos
action_approach_hor = np.array([differ[0], differ[1], 0, 0, 0, -0.13, -0.2])
action_approach_ver = np.array([0, 0, differ[2], 0, 0, 0, 0])
# action_approach = np.array([0.05, 0, -0.1, 0, 0, -0.15, 0])
action_rest     = np.array([0, 0, 0, 0, 0, 0, 0.2])
action_pickup   = np.array([0, 0, 0.1, 0, 0, 0, 0])

# actions: approach, gripper open -> rest, gripper close -> pick up, gripper close
total_action = np.concatenate((action_approach_hor.reshape(7,1), action_approach_ver.reshape(7,1), action_rest.reshape(7,1), action_pickup.reshape(7,1)),axis=1)
print("total_action: ", total_action)
# Loop through controller space
no_actions = total_action.shape[1]
print(env._eef_xpos)
print(env.model.mujoco_arena.table_offset)
for i in range(no_actions):
    # print(observables["cube_pos"].obs)
    for j in range(steps_per_action):
        # print(observables["gripper_to_cube_pos"].obs)
        # print(env._eef_xpos)
        obs, reward, done, info = env.step(total_action[:,i])
        env.render()
        # frame = obs["agentview" + "_image"]
        # writer.append_data(frame)
env.close()
# writer.close()
# while count < num_test_steps:
#     for i in range(steps_per_action):
#         # if controller_name in {"IK_POSE", "OSC_POSE"} and count > 2:
#         #     # Set this value to be the scaled axis angle vector
#         #     vec = np.zeros(3)
#         #     vec[count - 3] = test_value
#         #     action[3:6] = vec
#         # else:
#         #     action[count] = test_value
#         total_action = np.tile(action_approach, n)
#         # robot.grip_action(gripper = robot.gripper, gripper_action = np.tile(-1,n))
#         env.step(total_action)
#         env.render()
#     for i in range(steps_per_rest):
#         total_action = np.tile(neutral, n)
#         # robot.grip_action(gripper = robot.gripper, gripper_action = np.tile(1,n))
#         env.step(total_action)
#         env.render()
#     count += 1
#     print("count: ", count)

# # Shut down this env before starting the next test
# env.close()