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
import matplotlib.pyplot as plt

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

"""
改造说明：
- 移除键盘控制，改为自动随机指令。
- 持续 60s 仿真，每 5s 采样一次新的随机速度指令：vx, vy, wz ∈ U(-1.2, 1.2)。
- 记录期望与实际速度（基座坐标系）并在结束后绘图保存。
"""

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

    # Initialize camera controller
    camera_controller = CameraController(env)

    timestep = 0
    
    # Initialize loop timing measurement
    loop_times = []
    last_loop_time = time.time()

    # reset environment
    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict["observations"].get("commands")

    # Force test scheduling (no velocity commands)
    duration_s = 90.0              # total duration 90s
    push_interval_s = 5.0          # apply a force every 5s
    push_duration_s = 0.5          # each force lasts 0.5s
    start_time = time.time()
    next_push_time = start_time + push_interval_s  # first push at 5s
    is_pushing = False
    push_end_time = None
    push_force_vector = None

    # Random generator for forces
    rng = np.random.default_rng()

    # Data logging for plotting
    times: list[float] = []
    vx_hist: list[float] = []
    vy_hist: list[float] = []
    wz_hist: list[float] = []
    fx_hist: list[float] = []
    fy_hist: list[float] = []
    fz_hist: list[float] = []
    fmag_hist: list[float] = []

    # Low-pass filter states (first-order) for velocities
    tau = 0.2  # seconds
    vx_f = 0.0
    vy_f = 0.0
    wz_f = 0.0

    # simulate environment
    while simulation_app.is_running():
        # Measure loop timing
        current_time = time.time()
        loop_dt = current_time - last_loop_time
        loop_times.append(loop_dt)
        last_loop_time = current_time

        # Stop condition: run for duration_s seconds
        if (current_time - start_time) >= duration_s:
            print(f"[INFO] Reached {duration_s:.1f}s, stopping simulation loop.")
            break

        # Start a new push every push_interval_s seconds
        if (not is_pushing) and (current_time >= next_push_time):
            # Random unit direction in 3D
            direction = rng.standard_normal(3)
            norm = np.linalg.norm(direction)
            if norm == 0:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = direction / norm
            magnitude = rng.uniform(0.0, 100.0)  # [0, 100] N
            push_force_vector = torch.tensor(direction * magnitude, device=env.unwrapped.device, dtype=torch.float32)
            is_pushing = True
            push_end_time = current_time + push_duration_s
            print(f"[PUSH] Applying force {magnitude:.1f}N, dir={direction}, duration={push_duration_s}s")

        # While pushing, apply force; else ensure forces cleared
        robot = env.unwrapped.scene["robot"]
        forces = torch.zeros(env.num_envs, robot.num_bodies, 3, device=env.unwrapped.device)
        torques = torch.zeros_like(forces)
        if is_pushing and (push_force_vector is not None):
            # Apply to base link (index 0 assumed); if different, change here
            forces[:, 0, :] = push_force_vector
        robot.set_external_force_and_torque(forces, torques)

        # End push window
        if is_pushing and (push_end_time is not None) and (current_time >= push_end_time):
            is_pushing = False
            push_force_vector = None
            next_push_time += push_interval_s


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
        
        # Record velocities (base frame) and applied force for plotting
        # Compute current time since start
        t_rel = current_time - start_time
        times.append(t_rel)

        # Get robot velocities in world frame
        robot = env.unwrapped.scene["robot"]
        lin_vel_b = robot.data.root_lin_vel_b[0].cpu().numpy()  # [vx, vy, vz] in base
        ang_vel_b = robot.data.root_ang_vel_b[0].cpu().numpy()  # [wx, wy, wz] in base
        quat_w = robot.data.root_quat_w[0].cpu().numpy()        # [w, x, y, z]
        wq, xq, yq, zq = float(quat_w[0]), float(quat_w[1]), float(quat_w[2]), float(quat_w[3])
        yaw = np.arctan2(2.0 * (wq * zq + xq * yq), 1.0 - 2.0 * (yq * yq + zq * zq))
        c, s = np.cos(yaw), np.sin(yaw)
        # Rotate world -> base (Rz(-yaw))
        vx_b = c * lin_vel_b[0] + s * lin_vel_b[1]
        vy_b = -s * lin_vel_b[0] + c * lin_vel_b[1]
        wz_b = ang_vel_b[2]  # rotation about z unaffected by Rz

        # Low-pass filtering with variable dt (exponential smoothing)
        alpha = loop_dt / (tau + loop_dt) if (tau + loop_dt) > 0 else 1.0
        vx_f = vx_f + alpha * (vx_b - vx_f)
        vy_f = vy_f + alpha * (vy_b - vy_f)
        wz_f = wz_f + alpha * (wz_b - wz_f)

        vx_hist.append(vx_f)
        vy_hist.append(vy_f)
        wz_hist.append(wz_f)

        # Applied force (same on all envs): take current push vector or zero
        if is_pushing and (push_force_vector is not None):
            f_vec = push_force_vector.detach().cpu().numpy()
        else:
            f_vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        fx_hist.append(float(f_vec[0]))
        fy_hist.append(float(f_vec[1]))
        fz_hist.append(float(f_vec[2]))
        fmag_hist.append(float(np.linalg.norm(f_vec)))
            
        # Print loop timing occasionally
        if timestep % 20 == 0 and len(loop_times) > 0:
            avg_loop_time = np.mean(loop_times[-100:]) if len(loop_times) >= 100 else np.mean(loop_times)
            avg_loop_freq = 1.0 / avg_loop_time if avg_loop_time > 0 else 0
            print(f"[Step {timestep}] Loop timing: {avg_loop_time*1000:.2f}ms ({avg_loop_freq:.1f} Hz)")
        
        timestep += 1 

        # Update camera to follow robot
        camera_controller.update_camera_view()

    # Print final statistics
    avg_loop_time = np.mean(loop_times[1:]) if len(loop_times) > 1 else 0  # Skip first measurement
    avg_loop_freq = 1.0 / avg_loop_time if avg_loop_time > 0 else 0
    print(f"\n{'='*60}")
    print(f"[FINAL STATISTICS]")
    print(f"  Total timesteps: {timestep}")
    print(f"  Average loop time: {avg_loop_time*1000:.2f}ms ({avg_loop_freq:.1f} Hz)")
    print(f"  Physics dt: {env.unwrapped.physics_dt*1000:.2f}ms ({1.0/env.unwrapped.physics_dt:.0f} Hz)")
    print(f"  Control decimation: {env.unwrapped.cfg.decimation}")
    print(f"  Control dt: {env.unwrapped.step_dt*1000:.2f}ms ({1.0/env.unwrapped.step_dt:.1f} Hz)")
    print(f"{'='*60}\n")

    # Plot results: filtered base-frame velocities and applied force
    try:
        os.makedirs(os.path.join(log_dir, "plots"), exist_ok=True)
        out_dir = os.path.join(log_dir, "plots")

        # Velocities figure
        fig1, axs1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axs1[0].plot(times, vx_hist, label="vx (filtered)")
        axs1[0].set_ylabel("vx [m/s]")
        axs1[0].legend(loc="upper right")
        axs1[1].plot(times, vy_hist, label="vy (filtered)", color="tab:orange")
        axs1[1].set_ylabel("vy [m/s]")
        axs1[1].legend(loc="upper right")
        axs1[2].plot(times, wz_hist, label="wz (filtered)", color="tab:green")
        axs1[2].set_ylabel("wz [rad/s]")
        axs1[2].set_xlabel("time [s]")
        axs1[2].legend(loc="upper right")
        fig1.suptitle("Filtered Base-frame Velocities (forcetest)")
        fig1.tight_layout()
        vel_path = os.path.join(out_dir, "forcetest_velocities.png")
        fig1.savefig(vel_path, dpi=150)
        plt.close(fig1)

        # Forces figure
        fig2, axs2 = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        axs2[0].plot(times, fx_hist, label="Fx")
        axs2[0].set_ylabel("Fx [N]")
        axs2[0].legend(loc="upper right")
        axs2[1].plot(times, fy_hist, label="Fy", color="tab:orange")
        axs2[1].set_ylabel("Fy [N]")
        axs2[1].legend(loc="upper right")
        axs2[2].plot(times, fz_hist, label="Fz", color="tab:green")
        axs2[2].set_ylabel("Fz [N]")
        axs2[2].legend(loc="upper right")
        axs2[3].plot(times, fmag_hist, label="|F|", color="tab:red")
        axs2[3].set_ylabel("|F| [N]")
        axs2[3].set_xlabel("time [s]")
        axs2[3].legend(loc="upper right")
        fig2.suptitle("Applied External Force (forcetest)")
        fig2.tight_layout()
        force_path = os.path.join(out_dir, "forcetest_forces.png")
        fig2.savefig(force_path, dpi=150)
        plt.close(fig2)

        print(f"[PLOT] Saved velocity plot to: {vel_path}")
        print(f"[PLOT] Saved force plot to: {force_path}")
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
