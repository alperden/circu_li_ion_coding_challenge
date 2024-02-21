import argparse
import sys
import numpy as np
import imageio
from robosuite import make
from robosuite.controllers import load_controller_config
from robosuite.robots import Bimanual
from robosuite.models.grippers import GripperModel
import robosuite.macros as macros

# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"

if __name__ == "__main__":
    
    print("Coding Challenge: Simulating Autonomous Battery Disassembly")

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="video.mp4")
    args = parser.parse_args()

    # load default controller parameters for Operational Space Control (OSC)
    controller_name = "OSC_POSE"
    controller_config = load_controller_config(default_controller = controller_name)

    # Define the number of timesteps to use per controller action as well as timesteps in between actions
    steps_per_action = 75

    if "--video_path" in sys.argv:
        # create an environment to visualize on-screen
        env = make(
            "Lift",
            robots="IIWA",                          # load robot
            gripper_types="Robotiq140Gripper",      # load gripper
            controller_configs=controller_config,   # each arm is controlled using OSC
            env_configuration="single-arm-opposed", # single or two-arm selection
            has_renderer=False,                      # on-screen rendering
            ignore_done=True,
            control_freq=20,                        # 20 hz control for applied actions
            # horizon=(steps_per_action + steps_per_rest) * num_test_steps,
            use_object_obs=False,                   # no observations needed
            use_camera_obs=True,                   # no observations needed
            camera_names="agentview",
            camera_heights=512,
            camera_widths=512,
        )
        # reset the environment
        obs = env.reset()
        # env.viewer.set_camera(camera_id=0)
        observables = env._setup_observables()

        # z
        # |
        # |___ y  front view (x points outside of screen)
        differ = env.model.mujoco_arena.table_offset - env._eef_xpos
        action_approach_hor = np.array([differ[0], differ[1], 0, 0, 0, -0.12, -0.2])
        action_approach_ver = np.array([0, 0, differ[2]+0.02, 0, 0, 0, 0])
        action_rest     = np.array([0, 0, 0, 0, 0, 0, 0.2])
        action_pickup   = np.array([0, 0, 0.1, 0, 0, 0, 0])

        # actions: approach, gripper open -> rest, gripper close -> pick up, gripper close
        total_action = np.concatenate((action_approach_hor.reshape(7,1), action_approach_ver.reshape(7,1), action_rest.reshape(7,1), action_pickup.reshape(7,1)),axis=1)
        print("total_action: ", total_action)
        # Loop through controller space
        no_actions = total_action.shape[1]
        print(env._eef_xpos)
        print(env.model.mujoco_arena.table_offset)

        # create a video writer with imageio
        writer = imageio.get_writer(args.video_path, fps=20)
        frames = []

        for i in range(no_actions):
            # print(observables["cube_pos"].obs)
            for j in range(steps_per_action):
                # print(observables["gripper_to_cube_pos"].obs)
                # print(env._eef_xpos)
                obs, reward, done, info = env.step(total_action[:,i])
                # env.render()
                frame = obs["agentview" + "_image"]
                writer.append_data(frame)
                print("Saving frame #{}".format(j))
        env.close()
        writer.close()

    else:
        # create an environment to visualize on-screen
        env = make(
            "Lift",
            robots="IIWA",                          # load robot
            gripper_types="Robotiq140Gripper",      # load gripper
            controller_configs=controller_config,   # each arm is controlled using OSC
            env_configuration="single-arm-opposed", # single or two-arm selection
            has_renderer=True,                      # on-screen rendering
            render_camera="frontview",              # visualize the "frontview" camera 'frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'
            has_offscreen_renderer=True,            # no off-screen rendering
            ignore_done=True,
            control_freq=20,                        # 20 hz control for applied actions
            use_object_obs=True,                    # no observations needed
            use_camera_obs=True,                    # no observations needed
        )
        
        # reset the environment
        obs = env.reset()
        # env.viewer.set_camera(camera_id=0)
        observables = env._setup_observables()
        
        # z
        # |
        # |___ y  front view (x points outside of screen)
        differ = env.model.mujoco_arena.table_offset - env._eef_xpos
        action_approach_hor = np.array([differ[0], differ[1], 0, 0, 0, -0.12, -0.2])
        action_approach_ver = np.array([0, 0, differ[2]+0.01, 0, 0, 0, 0])
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
        env.close()