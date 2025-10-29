from isaaclab.utils import configclass

from whole_body_tracking.robots.t1 import T1_ACTION_SCALE, BOOSTER_T1_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class T1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = T1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "Trunk"
        self.commands.motion.body_names = [
            "Trunk",
            "Hip_Roll_Left",
            "Shank_Left",
            "left_foot_link",
            "Hip_Roll_Right",
            "Shank_Right",
            "right_foot_link",
            "Waist",
            "AL2",
            "AL3",
            "left_hand_link",
            "AR2",
            "AR3",
            "right_hand_link",
        ]


@configclass
class T1FlatWoStateEstimationEnvCfg(T1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class T1FlatLowFreqEnvCfg(T1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
