"""RSL-RL智能体检查点播放脚本 / Script to play a checkpoint of an RL agent from RSL-RL."""

"""首先启动Isaac Sim仿真器 / Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# 添加argparse参数 / Add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch
import numpy as np
import time

from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg,DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg, export_mlp_as_onnx, export_policy_as_jit

# # [新增 1] 初始化键盘控制器
# from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg
# keyboard_cfg = Se3KeyboardCfg()
# # keyboard_cfg.vx_scale = 1.0      # 前后速度缩放
# # keyboard_cfg.vy_scale = 1.0      # 左右速度缩放
# # keyboard_cfg.yaw_scale = 1.5     # 旋转速度更敏感
# keyboard = Se3Keyboard(cfg=keyboard_cfg)
# print("\n" + "=" * 50)
# print("键盘控制已激活 / Keyboard Control Active")
# print("W / S : 前进 / 后退 (Linear Velocity X)")
# print("A / D : 左移 / 右移 (Linear Velocity Y)")
# print("Q / E : 左转 / 右转 (Angular Velocity Z)")
# print("K     : 复位键盘输入 (Reset Input)")
# print("=" * 50 + "\n")

class ManualController:
    """Manual controller for robot using WASD keys."""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.linear_velocity = torch.zeros(3, device=device)
        self.angular_velocity = torch.zeros(3, device=device)
        self.max_linear_vel = 1.0
        self.max_angular_vel = 1.0
        
    def get_velocity_command(self):
        """Return SE2 command [vx, vy, wz] in robot base frame."""
        return torch.tensor(
            [self.linear_velocity[0].item(), self.linear_velocity[1].item(), self.angular_velocity[2].item()],
            device=self.device,
            dtype=torch.float32,
        )
    
    def update_from_keys(self, keys_pressed):
        """Update velocity based on pressed keys."""
        # Reset velocities
        self.linear_velocity.zero_()
        self.angular_velocity.zero_()
        
        # WASD control
        if 'w' in keys_pressed:
            self.linear_velocity[0] = self.max_linear_vel  # Forward
        if 's' in keys_pressed:
            self.linear_velocity[0] = -self.max_linear_vel  # Backward
        if 'a' in keys_pressed:
            self.linear_velocity[1] = self.max_linear_vel  # Left
        if 'd' in keys_pressed:
            self.linear_velocity[1] = -self.max_linear_vel  # Right
            
        # QE for rotation
        if 'q' in keys_pressed:
            self.angular_velocity[2] = self.max_angular_vel  # Turn left
        if 'e' in keys_pressed:
            self.angular_velocity[2] = -self.max_angular_vel  # Turn right

        if 'k' in keys_pressed:
            self.linear_velocity.zero_()
            self.angular_velocity.zero_()

class CameraController:
    """Camera controller to follow the robot (no external deps)."""

    def __init__(self, env, camera_distance=5.0, camera_height=2.0):
        self.env = env
        self.camera_distance = camera_distance
        self.camera_height = camera_height

    def update_camera_view(self):
        """Update camera to follow the robot."""
        # Get robot position and yaw heading from quaternion
        robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
        q = self.env.unwrapped.scene["robot"].data.root_quat_w[0].cpu().numpy()  # [w, x, y, z]
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        # Compute forward vector in world from yaw (approx):
        # forward in local = [1, 0, 0]
        # world forward xz using quaternion (ignore roll/pitch for camera)
        # yaw from quaternion
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        # camera behind the robot along -forward
        camera_eye = robot_pos + (-forward * self.camera_distance)
        camera_eye[2] += self.camera_height
        # look at the robot center slightly above
        camera_target = robot_pos.copy()
        camera_target[2] += 0.5
        self.env.unwrapped.sim.set_camera_view(eye=camera_eye, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

def main():
    """使用RSL-RL智能体进行测试 / Play with RSL-RL agent."""
    # 解析配置 / Parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.seed = agent_cfg.seed

    # 指定日志实验目录 / Specify directory for logging experiments
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path
    log_dir = os.path.dirname(resume_path)

    # 创建isaac环境 / Create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    # load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

     # 导出策略到onnx / Export policy to onnx
    if EXPORT_POLICY:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.actor_critic, export_model_dir
        )
        print("Exported policy as jit script to: ", export_model_dir)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor, 
            export_model_dir, 
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            export_model_dir,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )

    # Initialize manual controller and camera controller
    manual_controller = ManualController(device=env.unwrapped.device)
    camera_controller = CameraController(env)

    # Get keyboard input handler
    try:
        import keyboard
        print("[INFO] Keyboard input enabled. Use WASD/QE to control the robot (ESC to exit)")
    except ImportError:
        print("[WARNING] keyboard module not available. Install with: pip install keyboard")
        print("[INFO] Using default policy control instead.")
        keyboard = None
    except Exception as e:
        print(f"[ERROR] Keyboard initialization failed: {e}")
        print("[INFO] Using default policy control instead.")
        keyboard = None

    timestep = 0
    
    # Initialize push force tracking variables
    push_start_time = None
    push_force_vector = None
    total_impulse = 0.0
    is_pushing = False
    
    # Initialize velocity tracking for MSE calculation
    velocity_errors_squared = []

    # reset environment
    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict["observations"].get("commands") 
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # Check for exit
        if keyboard and keyboard.is_pressed('esc'):
            print("[INFO] ESC pressed, exiting...")
            break
            
        # Get current key presses
        keys_pressed = []
        push_robot = False
        if keyboard:
            if keyboard.is_pressed('w'):
                keys_pressed.append('w')
            if keyboard.is_pressed('s'):
                keys_pressed.append('s')
            if keyboard.is_pressed('a'):
                keys_pressed.append('a')
            if keyboard.is_pressed('d'):
                keys_pressed.append('d')
            if keyboard.is_pressed('q'):
                keys_pressed.append('q')
            if keyboard.is_pressed('e'):
                keys_pressed.append('e')
            if keyboard.is_pressed('p'):
                push_robot = True
        
        # Update manual controller
        if keyboard:
            manual_controller.update_from_keys(keys_pressed)
            # Override the command in the environment (expects [vx, vy, wz])
            velocity_cmd = manual_controller.get_velocity_command()

            cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
            # broadcast to all envs
            cmd_term.vel_command_b[:] = velocity_cmd.unsqueeze(0).repeat(env.num_envs, 1)
            # ensure angular velocity mode (not heading) and not standing
            # if hasattr(cmd_term, "is_heading_env"):
            #     cmd_term.is_heading_env[:] = False
            # if hasattr(cmd_term, "is_standing_env"):
            #     cmd_term.is_standing_env[:] = False
            
            # [关键修改] 将键盘指令直接写入策略网络的输入张量
            # [Key Fix] Overwrite the commands tensor fed to the policy
            commands[:, :3] = velocity_cmd.unsqueeze(0)

        # Apply external force when 'p' is pressed
        if push_robot:
            if not is_pushing:
                # Start of push - initialize tracking
                is_pushing = True
                push_start_time = time.time()
                total_impulse = 0.0
                # Generate random direction (unit vector)
                random_direction = torch.randn(3, device=env.unwrapped.device)
                random_direction = random_direction / torch.norm(random_direction)  # normalize to unit vector
                # Scale to 200N
                force_magnitude = 200.0
                push_force_vector = random_direction * force_magnitude
                print(f"[INFO] Started applying 200N force: direction={push_force_vector.cpu().numpy()}")
            
            # Continue applying force (only if push_force_vector is initialized)
            if push_force_vector is not None:
                robot = env.unwrapped.scene["robot"]
                forces = torch.zeros(env.num_envs, robot.num_bodies, 3, device=env.unwrapped.device)
                forces[:, 0, :] = push_force_vector  # Apply to base link
                torques = torch.zeros_like(forces)
                robot.set_external_force_and_torque(forces, torques)
                
                # Accumulate impulse (Impulse = Force × dt)
                dt = env.unwrapped.physics_dt  # Physics timestep
                impulse_this_step = torch.norm(push_force_vector).item() * dt
                total_impulse += impulse_this_step
            
        else:
            if is_pushing:
                # End of push - print results
                push_duration = time.time() - push_start_time if push_start_time is not None else 0.0
                print(f"[INFO] Push ended. Duration: {push_duration:.3f}s, Total Impulse: {total_impulse:.2f} N·s")
                is_pushing = False
                push_force_vector = None

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            # env stepping
            obs, _, _, infos = env.step(actions)
            obs_history = infos["observations"].get("obsHistory")
            obs_history = obs_history.flatten(start_dim=1)
            commands = infos["observations"].get("commands")
            
        # Get robot actual velocity and desired velocity
        robot = env.unwrapped.scene["robot"]
        
        # Check if base frame velocities are available (first iteration only)
        if timestep == 0:
            print("\n=== Checking available velocity attributes ===")
            print(f"Has root_lin_vel_w: {hasattr(robot.data, 'root_lin_vel_w')}")
            print(f"Has root_ang_vel_w: {hasattr(robot.data, 'root_ang_vel_w')}")
            print(f"Has root_lin_vel_b: {hasattr(robot.data, 'root_lin_vel_b')}")
            print(f"Has root_ang_vel_b: {hasattr(robot.data, 'root_ang_vel_b')}")
            if hasattr(robot.data, 'root_lin_vel_b'):
                print("✓ Base-frame velocities are available directly!")
            else:
                print("✗ Base-frame velocities NOT available, using manual transformation")
            print("=" * 50 + "\n")
        
        # Try to use base-frame velocities if available, otherwise transform
        if hasattr(robot.data, 'root_lin_vel_b') and hasattr(robot.data, 'root_ang_vel_b'):
            # Direct access to base-frame velocities
            actual_lin_vel = robot.data.root_lin_vel_b[0, :2]  # [vx_b, vy_b] in base frame
            actual_ang_vel = robot.data.root_ang_vel_b[0, 2]   # wz in base frame
        else:
            # Manual transformation from world to base frame
            # Get velocity in world frame
            actual_lin_vel_w = robot.data.root_lin_vel_w[0, :2]  # [vx_w, vy_w] in world frame
            actual_ang_vel_w = robot.data.root_ang_vel_w[0, 2]   # wz in world frame (same in both frames for yaw)
            
            # Get robot orientation (quaternion [w, x, y, z])
            quat_w = robot.data.root_quat_w[0]  # quaternion in world frame
            
            # Convert linear velocity from world frame to base frame
            # Extract yaw angle from quaternion
            w, x, y, z = quat_w[0], quat_w[1], quat_w[2], quat_w[3]
            yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            
            # Rotation matrix from world to base frame (2D rotation around z-axis)
            cos_yaw = torch.cos(yaw)
            sin_yaw = torch.sin(yaw)
            
            # Transform velocity: v_base = R^T * v_world
            actual_lin_vel = torch.stack([
                cos_yaw * actual_lin_vel_w[0] + sin_yaw * actual_lin_vel_w[1],  # vx in base frame
                -sin_yaw * actual_lin_vel_w[0] + cos_yaw * actual_lin_vel_w[1]  # vy in base frame
            ])
            actual_ang_vel = actual_ang_vel_w  # Angular velocity around z is same in both frames
        
        # Desired velocity from commands (first env, already in base frame)

        desired_vel = commands[0, :3]  # [vx_des, vy_des, wz_des] in base frame
        desired_lin_vel = desired_vel[:2]
        desired_ang_vel = desired_vel[2]
        
        # Calculate velocity tracking error
        lin_vel_error = actual_lin_vel - desired_lin_vel
        ang_vel_error = actual_ang_vel - desired_ang_vel
        
        # Calculate MSE for this timestep
        lin_vel_mse = torch.sum(lin_vel_error ** 2).item()
        ang_vel_mse = (ang_vel_error ** 2).item()
        total_vel_mse = lin_vel_mse + ang_vel_mse
        
        velocity_errors_squared.append(total_vel_mse)
        
        # Print velocity tracking info every 50 steps
        if timestep % 100 == 0:
            avg_mse = np.mean(velocity_errors_squared[-50:]) if len(velocity_errors_squared) >= 50 else np.mean(velocity_errors_squared)
            print(f"\n[Step {timestep}] Velocity Tracking:")
            print(f"  Desired: vx={desired_vel[0].item():.3f}, vy={desired_vel[1].item():.3f}, wz={desired_vel[2].item():.3f}")
            print(f"  Actual:  vx={actual_lin_vel[0].item():.3f}, vy={actual_lin_vel[1].item():.3f}, wz={actual_ang_vel.item():.3f}")
            print(f"  Error:   vx={lin_vel_error[0].item():.3f}, vy={lin_vel_error[1].item():.3f}, wz={ang_vel_error.item():.3f}")
            print(f"  MSE (last 50 steps): {avg_mse:.6f}")
        
        timestep += 1 

        # Update camera to follow robot
        camera_controller.update_camera_view()

    # Print final statistics
    if len(velocity_errors_squared) > 0:
        overall_mse = np.mean(velocity_errors_squared)
        print(f"\n{'='*60}")
        print(f"[FINAL STATISTICS]")
        print(f"  Total timesteps: {timestep}")
        print(f"  Overall Velocity Tracking MSE: {overall_mse:.6f}")
        print(f"  Overall Velocity Tracking RMSE: {np.sqrt(overall_mse):.6f}")
        print(f"{'='*60}\n")

    # close the simulator
    env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
