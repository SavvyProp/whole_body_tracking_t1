# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# Change damping to 4 * p / natural_freq

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

ARMATURE_MID = 0.01
ARMATURE_HIGH = 0.025
ARMATURE_LOW = 0.005

STIFFNESS_LOW = ARMATURE_LOW * NATURAL_FREQ**2
STIFFNESS_MID = ARMATURE_MID * NATURAL_FREQ**2
STIFFNESS_HIGH = ARMATURE_HIGH * NATURAL_FREQ**2

DAMPING_LOW = 2.0 * DAMPING_RATIO * ARMATURE_LOW * NATURAL_FREQ
DAMPING_MID = 2.0 * DAMPING_RATIO * ARMATURE_MID * NATURAL_FREQ
DAMPING_HIGH = 2.0 * DAMPING_RATIO * ARMATURE_HIGH * NATURAL_FREQ

BOOSTER_T1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/booster/T1_29dof/T1_29dof.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.72),
        joint_pos={
            # Head
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0,
            # Arm
            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.35,
            "Right_Shoulder_Roll": 1.35,
            ".*_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            # Waist
            "Waist": 0.0,
            # Leg
            ".*_Hip_Pitch": -0.20,
            ".*_Hip_Roll": 0.0,
            ".*_Hip_Yaw": 0.0,
            ".*_Knee_Pitch": 0.42,
            ".*_Ankle_Pitch": -0.23,
            ".*_Ankle_Roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
                "Waist",
            ],
            effort_limit_sim={
                ".*_Hip_Pitch": 45.0,
                ".*_Hip_Roll": 30.0,
                ".*_Hip_Yaw": 30.0,
                ".*_Knee_Pitch": 60.0,
                "Waist": 30.0,
            },
            velocity_limit_sim={
                ".*_Hip_Pitch": 12.5,
                ".*_Hip_Roll": 10.9,
                ".*_Hip_Yaw": 10.9,
                ".*_Knee_Pitch": 11.7,
                "Waist": 10.88,
            },
            armature = {
                ".*_Hip_Pitch": ARMATURE_HIGH,
                ".*_Hip_Roll": ARMATURE_MID,
                ".*_Hip_Yaw": ARMATURE_MID,
                ".*_Knee_Pitch": ARMATURE_HIGH,
                "Waist": ARMATURE_MID,
            },
            stiffness = {
                ".*_Hip_Pitch": STIFFNESS_HIGH,
                ".*_Hip_Roll": STIFFNESS_MID,
                ".*_Hip_Yaw": STIFFNESS_MID,
                ".*_Knee_Pitch": STIFFNESS_HIGH,
                "Waist": STIFFNESS_MID,
            },
            damping = {
                ".*_Hip_Pitch": DAMPING_HIGH,
                ".*_Hip_Roll": DAMPING_MID,
                ".*_Hip_Yaw": DAMPING_MID,
                ".*_Knee_Pitch": DAMPING_HIGH,
                "Waist": DAMPING_MID,
            }
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit_sim={".*_Ankle_Pitch": 24, ".*_Ankle_Roll": 15},
            velocity_limit_sim={".*_Ankle_Pitch": 18.8, ".*_Ankle_Roll": 12.4},
            stiffness=STIFFNESS_LOW,
            damping=DAMPING_LOW,
            armature=ARMATURE_LOW,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
                ".*_Wrist_Pitch",
                ".*_Wrist_Yaw",
                ".*_Hand_Roll",
            ],
            effort_limit_sim=18.0,
            velocity_limit_sim=18.8,
            stiffness=STIFFNESS_LOW,
            damping=DAMPING_LOW,
            armature=ARMATURE_LOW,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["AAHead_yaw", "Head_pitch"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=STIFFNESS_LOW,
            damping=DAMPING_LOW,
            armature=ARMATURE_LOW,
        )
    },
)

FT_FAC = 0.75

BOOSTER_T1_LOWGAIN_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/booster/T1_29dof/T1_29dof.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.72),
        joint_pos={
            # Head
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0,
            # Arm
            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.35,
            "Right_Shoulder_Roll": 1.35,
            ".*_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            # Waist
            "Waist": 0.0,
            # Leg
            ".*_Hip_Pitch": -0.20,
            ".*_Hip_Roll": 0.0,
            ".*_Hip_Yaw": 0.0,
            ".*_Knee_Pitch": 0.42,
            ".*_Ankle_Pitch": -0.23,
            ".*_Ankle_Roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
                "Waist",
            ],
            effort_limit_sim={
                ".*_Hip_Pitch": 45.0,
                ".*_Hip_Roll": 25.0,
                ".*_Hip_Yaw": 25.0,
                ".*_Knee_Pitch": 60.0,
                "Waist": 30.0,
            },
            velocity_limit_sim={
                ".*_Hip_Pitch": 12.5,
                ".*_Hip_Roll": 10.9,
                ".*_Hip_Yaw": 10.9,
                ".*_Knee_Pitch": 11.7,
                "Waist": 10.88,
            },
            armature = {
                ".*_Hip_Pitch": ARMATURE_HIGH,
                ".*_Hip_Roll": ARMATURE_MID,
                ".*_Hip_Yaw": ARMATURE_MID,
                ".*_Knee_Pitch": ARMATURE_HIGH,
                "Waist": ARMATURE_MID,
            },
            stiffness = {
                ".*_Hip_Pitch": STIFFNESS_HIGH,
                ".*_Hip_Roll": STIFFNESS_MID,
                ".*_Hip_Yaw": STIFFNESS_MID,
                ".*_Knee_Pitch": STIFFNESS_HIGH,
                "Waist": STIFFNESS_MID,
            },
            damping = {
                ".*_Hip_Pitch": DAMPING_HIGH,# * FT_FAC * 2.0,
                ".*_Hip_Roll": DAMPING_MID,# * FT_FAC * 2.0,
                ".*_Hip_Yaw": DAMPING_MID,# * FT_FAC * 2.0,
                ".*_Knee_Pitch": DAMPING_HIGH,# * FT_FAC * 2.0,
                "Waist": DAMPING_MID,# * FT_FAC,
            }
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit_sim={".*_Ankle_Pitch": 24, ".*_Ankle_Roll": 15},
            velocity_limit_sim={".*_Ankle_Pitch": 18.8, ".*_Ankle_Roll": 12.4},
            stiffness=STIFFNESS_LOW,
            damping=DAMPING_LOW,# * FT_FAC * 2.0,
            armature=ARMATURE_LOW,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
                ".*_Wrist_Pitch",
                ".*_Wrist_Yaw",
                ".*_Hand_Roll",
            ],
            effort_limit_sim=18.0,
            velocity_limit_sim=18.8,
            stiffness=STIFFNESS_LOW,
            damping=DAMPING_LOW, #* FT_FAC * 2.0,
            armature=ARMATURE_LOW,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["AAHead_yaw", "Head_pitch"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=STIFFNESS_LOW * FT_FAC,
            damping=DAMPING_LOW, # * FT_FAC * 2.0,
            armature=ARMATURE_LOW,
        )
    },
)
"""Configuration for the Booster T1 Humanoid robot."""

T1_ACTION_SCALE = {}
for a in BOOSTER_T1_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            T1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

T1_LG_ACTION_SCALE = {}
for a in BOOSTER_T1_LOWGAIN_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    d = a.damping
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    if not isinstance(d, dict):
        d = {n: d for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            print(n, e[n], s[n], d[n])
            T1_LG_ACTION_SCALE[n] = 0.25 * FT_FAC * e[n] / s[n]

print("T1_ACTION_SCALE:", T1_ACTION_SCALE)
print("T1_LG_ACTION_SCALE:", T1_LG_ACTION_SCALE)

"""Configuration for the Booster T1 Humanoid robot."""

