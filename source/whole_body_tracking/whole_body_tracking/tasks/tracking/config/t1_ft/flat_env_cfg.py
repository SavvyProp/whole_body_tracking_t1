from isaaclab.utils import configclass

from whole_body_tracking.robots.t1 import T1_ACTION_SCALE, BOOSTER_T1_LOW_GAIN_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_ft_env_cfg import TrackingFTEnvCfg


@configclass
class T1FTEnvCfg(TrackingFTEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = BOOSTER_T1_LOW_GAIN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
