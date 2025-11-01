from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

def contact_state(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    threshold = 10.0
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    contact_mask = (torch.linalg.norm(net_forces_w, dim=-1) > threshold)  # (N, |body_ids|)
    pred_contact_mask = torch.sigmoid(env.ft_rew_info["components"]["w"])
    lse = torch.sum(torch.square(contact_mask.float() - pred_contact_mask), dim=-1)
    return lse

def centroid_velocity(env: ManagerBasedRLEnv):
    command_name = "motion"
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, ["Trunk"])
    linvel = command.body_lin_vel_w[:, body_indexes][:, 0, :]
    des_linvel = env.ft_rew_info["components"]["des_com_vel"][:, :3]
    lse = torch.sum(torch.square(linvel - des_linvel), dim=-1)

    linvel_rew = torch.exp(-lse / 0.25)
    return linvel_rew

def ft_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    pos_component = env.action_manager.action[:, :23]
    remaining_component = env.action_manager.action[:, 23:]
    pos_prev = env.action_manager.prev_action[:, :23]
    remaining_prev = env.action_manager.prev_action[:, 23:]
    pos_l2_err = torch.sum(torch.square(pos_component - pos_prev), dim=1)
    remaining_l2_err = torch.sum(torch.square(remaining_component - remaining_prev), dim=1)
    return pos_l2_err + remaining_l2_err * 0.10

def ft_torque_select(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Encourage the torque selection values to be close to 0 or 1."""
    torque_select = env.ft_rew_info["components"]["torque_select"]
    l2_err = torch.sum( torque_select, dim= -1)
    return l2_err / 23

def ft_force_correctness(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    pred_force = env.ft_rew_info["debug"]["f"]
    num_eef = pred_force.shape[1] // 6
    pred_force = pred_force.reshape(pred_force.shape[0], num_eef, 6)
    pred_lin_force = pred_force[:, :, :3]
    force_err = torch.sum(torch.square(net_forces_w - pred_lin_force), dim=-1)  # (N, EEF_NUM)
    sigma = 150.0
    exp_err = torch.exp(-torch.sum(force_err, dim = -1) / (sigma ** 2))
    return exp_err

def ft_tau_ref(env: ManagerBasedRLEnv) -> torch.Tensor:
    applied_torque = env.scene["robot"].data.applied_torque
    tau_ref = env.ft_rew_info["components"]["torque"]
    # Reward for minimizing error between applied torque and tau_ref
    frc_err = torch.sum(torch.square(applied_torque - tau_ref), dim=-1)
    sigma = 50
    exp_err = torch.exp(-frc_err / (sigma ** 2))
    return exp_err