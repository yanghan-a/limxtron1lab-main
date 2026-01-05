import math

from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.pointfoot_cfg import POINTFOOT_CFG
from bipedal_locomotion.tasks.locomotion.cfg.PF.limx_base_env_cfg import PFEnvCfg
from bipedal_locomotion.tasks.locomotion.cfg.PF.terrains_cfg import (
    BLIND_ROUGH_TERRAINS_CFG,
    BLIND_ROUGH_TERRAINS_PLAY_CFG,
    STAIRS_TERRAINS_CFG,
    STAIRS_TERRAINS_PLAY_CFG,
    BLIND_HARD_ROUGH_TERRAINS_CFG,
    BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG,
)

from isaaclab.sensors import RayCasterCfg, patterns
from bipedal_locomotion.tasks.locomotion import mdp
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg


######################
# 双足机器人基础环境 / Pointfoot Base Environment
######################


@configclass
class PFBaseEnvCfg(PFEnvCfg):
    """双足机器人基础环境配置 - 所有变体的共同基础 / Base environment configuration for pointfoot robot - common foundation for all variants"""
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = POINTFOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }
        # 调整基座质量随机化参数 / Adjust base mass randomization parameters
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        # 设置基座接触终止条件 / Set base contact termination condition
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"
        
        # 更新视口相机设置 / Update viewport camera settings
        self.viewer.origin_type = "env"  # 相机跟随环境 / Camera follows environment


@configclass
class PFBaseEnvCfg_PLAY(PFBaseEnvCfg):
    """双足机器人基础测试环境配置 - 用于策略评估 / Base play environment configuration - for policy evaluation"""
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None


############################
# 双足机器人盲视平地环境 / Pointfoot Blind Flat Environment
############################


@configclass
class PFBlindFlatEnvCfg(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum.terrain_levels = None


@configclass
class PFBlindFlatEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum.terrain_levels = None


#############################
# 双足机器人盲视粗糙环境 / Pointfoot Blind Rough Environment
#############################


@configclass
class PFBlindRoughEnvCfg(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.scene.terrain.terrain_type = "generator"
        # self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG
        self.scene.terrain.terrain_generator = BLIND_HARD_ROUGH_TERRAINS_CFG


@configclass
class PFBlindRoughEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        # self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_PLAY_CFG
        self.scene.terrain.terrain_generator = BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG



##############################
# 双足机器人盲视楼梯环境 / Pointfoot Blind Stairs Environment
##############################


@configclass
class PFBlindStairEnvCfg(PFBaseEnvCfg):
    """盲视楼梯环境配置 - 专门训练爬楼梯能力 / Blind stairs environment configuration - specialized for stair climbing training"""
    
    def __post_init__(self):
        """后初始化 - 配置楼梯训练环境 / Post-initialization - configure stairs training environment"""
        super().__post_init__()
        
        # 移除视觉组件 / Remove vision components
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        # 调整速度命令范围以适应楼梯环境 / Adjust velocity command ranges for stairs environment
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)      # 前进速度：0.5-1.0 m/s / Forward velocity: 0.5-1.0 m/s
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)     # 横向速度：0（仅直行）/ Lateral velocity: 0 (straight only)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)  # 转向：±30度 / Turning: ±30 degrees

        # 调整奖励权重以适应楼梯爬升 / Adjust reward weights for stair climbing
        self.rewards.rew_lin_vel_xy.weight = 2.0          # 增加线速度跟踪奖励 / Increase linear velocity tracking reward
        self.rewards.rew_ang_vel_z.weight = 1.5           # 增加角速度跟踪奖励 / Increase angular velocity tracking reward
        self.rewards.pen_lin_vel_z.weight = 0.1          # 增加Z方向速度惩罚 / Increase Z velocity penalty
        self.rewards.pen_ang_vel_xy.weight = -0.05        # XY角速度惩罚 / XY angular velocity penalty
        self.rewards.pen_action_rate.weight = -0.01       # 动作变化率惩罚 / Action rate penalty
        self.rewards.pen_flat_orientation.weight = -5   # 姿态保持惩罚 / Orientation keeping penalty
        self.rewards.pen_undesired_contacts.weight = -1.0 # 不期望接触惩罚 / Undesired contact penalty

        # 设置楼梯地形 / Set up stairs terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG

@configclass
class PFBlindStairEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
    """盲视楼梯测试环境配置 / Blind stairs play environment configuration"""
    
    def __post_init__(self):
        """后初始化 - 配置楼梯测试环境 / Post-initialization - configure stairs testing environment"""
        super().__post_init__()
        
        # 移除视觉组件 / Remove vision components
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        # 设置测试专用的速度命令 / Set testing-specific velocity commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)    # 固定前进速度范围 / Fixed forward velocity range
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)   # 无横向移动 / No lateral movement
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)   # 无转向 / No turning

        # 固定重置姿态（无偏航角变化）/ Fixed reset pose (no yaw variation)
        self.events.reset_robot_base.params["pose_range"]["yaw"] = (-0.0, 0.0)

        # 设置测试楼梯地形 / Set up testing stairs terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        # 设置中等难度的楼梯测试环境 / Set medium difficulty stairs testing environment
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(difficulty_range=(0.5, 0.5))



##############################
# modifying the reward parameters双足机器人盲视楼梯环境 / Pointfoot Blind Stairs Environment
##############################
@configclass
class PFBlindStairEnvCfgMy(PFBaseEnvCfg):
    """盲视楼梯环境配置 - 专门训练爬楼梯能力 / Blind stairs environment configuration - specialized for stair climbing training"""
    
    def __post_init__(self):
        """后初始化 - 配置楼梯训练环境 / Post-initialization - configure stairs training environment"""
        super().__post_init__()
        
        # 移除视觉组件 / Remove vision components
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        # 调整速度命令范围以适应楼梯环境 / Adjust velocity command ranges for stairs environment
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)      # 前进速度：0.5-1.0 m/s / Forward velocity: 0.5-1.0 m/s
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)     # 横向速度：0（仅直行）/ Lateral velocity: 0 (straight only)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)  # 转向：±30度 / Turning: ±30 degrees

        self.commands.gait_command= mdp.UniformGaitCommandCfg(
            resampling_time_range=(5.0, 5.0),  # 命令重采样时间范围 (固定5秒) / Command resampling time range (fixed 5s)
            debug_vis=False,                    # 不显示调试可视化 / No debug visualization
            ranges=mdp.UniformGaitCommandCfg.Ranges(
                # frequencies=(1.5, 2.5),     # 步态频率范围 [Hz] / Gait frequency range [Hz]
                frequencies=(2.2, 3.2),     # 步态频率范围 [Hz] / Gait frequency range [Hz]
                offsets=(0.5, 0.5),         # 相位偏移范围 [0-1] / Phase offset range [0-1]
                durations=(0.5, 0.5),       # 接触持续时间范围 [0-1] / Contact duration range [0-1]
                swing_height=(0.2, 0.3)     # 摆动高度范围 [m] / Swing height range [m]
            ),
        )

        # 调整奖励权重以适应楼梯爬升 / Adjust reward weights for stair climbing
        
        self.rewards.rew_lin_vel_xy.weight = 1.5          # 增加线速度跟踪奖励 / Increase linear velocity tracking reward
        self.rewards.rew_ang_vel_z.weight = 0.75           # 增加角速度跟踪奖励 / Increase angular velocity tracking reward
       
        self.rewards.keep_balance.weight = 1.0          #keep balance

        self.rewards.pen_undesired_contacts.weight = -0.5 # 不期望接触惩罚 / Undesired contact penalty


        self.rewards.pen_lin_vel_z.weight = -0.5          # 增加Z方向速度惩罚 / Increase Z velocity penalty
        self.rewards.pen_ang_vel_xy.weight = -0.05        # XY角速度惩罚 / XY angular velocity penalty


        self.rewards.pen_action_rate.weight = -0.01       # 动作变化率惩罚 / Action rate penalty
        self.rewards.pen_action_smoothness.weight = -0.01       # 动作变化率惩罚 / Action rate penalty

        self.rewards.pen_flat_orientation.weight = -1.0   # 姿态保持惩罚 / Orientation keeping penalty

        self.rewards.pen_joint_vel_l2.weight = -5.0e-05
        self.rewards.pen_joint_accel.weight = -2.5e-07
        self.rewards.pen_joint_powers.weight = -2.5e-05

        self.rewards.pen_base_height.weight = -1.0

        self.rewards.pen_joint_torque.weight = -2.0e-05
        self.rewards.pen_joint_pos_limits.weight = -1.0

        self.rewards.test_gait_reward.weight = 1.0

        self.rewards.pen_feet_distance.weight = RewTerm(
            func=mdp.feet_distance,                     # 足部距离惩罚 / Foot distance penalty
            weight=-10,
            params={
                "min_feet_distance": 0.100,            # 最小足部距离 / Minimum foot distance
                "feet_links_name": ["foot_[RL]_Link"]  # 足部连杆名称 / Foot link names
            }
        )

        self.rewards.foot_landing_vel.weight = 0.0
        self.rewards.pen_feet_regulation.weight = 0.0

        
        # 设置楼梯地形 / Set up stairs terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG

@configclass
class PFBlindStairEnvCfg_PLAYMy(PFBaseEnvCfg_PLAY):
    """盲视楼梯测试环境配置 / Blind stairs play environment configuration"""
    
    def __post_init__(self):
        """后初始化 - 配置楼梯测试环境 / Post-initialization - configure stairs testing environment"""
        super().__post_init__()
        
        # 移除视觉组件 / Remove vision components
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        # 设置测试专用的速度命令 / Set testing-specific velocity commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)    # 固定前进速度范围 / Fixed forward velocity range
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)   # 无横向移动 / No lateral movement
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)   # 无转向 / No turning

        # 固定重置姿态（无偏航角变化）/ Fixed reset pose (no yaw variation)
        self.events.reset_robot_base.params["pose_range"]["yaw"] = (-0.0, 0.0)

        # 设置测试楼梯地形 / Set up testing stairs terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        # 设置中等难度的楼梯测试环境 / Set medium difficulty stairs testing environment
        # self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(difficulty_range=(0.5, 0.5))
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(difficulty_range=(0.5, 1.0))



#############################
# 带高度扫描的双足机器人楼梯环境 / Pointfoot Stairs Environment with Height Scanning
#############################

@configclass
class PFStairEnvCfgv1(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
                    noise=GaussianNoise(mean=0.0, std=0.01),
                    clip = (0.0, 10.0),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG


@configclass
class PFStairEnvCfgv1_PLAY(PFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(difficulty_range=(0.5, 0.5))