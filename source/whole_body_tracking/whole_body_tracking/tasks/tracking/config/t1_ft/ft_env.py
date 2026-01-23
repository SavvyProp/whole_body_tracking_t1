from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ActionManager
import torch
from whole_body_tracking.utils import ft
from isaaclab.managers import ActionManager, EventManager, ObservationManager, RecorderManager
from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from isaaclab.ui.widgets import ManagerLiveVisualizer
import time
from whole_body_tracking.utils.ft import EEF_BODIES
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_angle_axis, quat_mul
import re

# Implementation of FT environment. Idea is to implement the pure FT
# function and save a ft_info dict as part of the env class
# No observation change, contact rewards, centroid velocity rewards

# Implementation note: build a custom ActionManager with overriden action size
# ActionManager should have the big action size

def robot_dict(robot):
    #cor_nle = robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()[:, 6:]
    grav_nle = robot.root_physx_view.get_gravity_compensation_forces()[:, 6:]
    nle = grav_nle #cor_nle + grav_nle
    return {
        "jacs": robot.root_physx_view.get_jacobians(),
        "base_quat": robot.data.root_link_quat_w,
        "com_pos": robot.data.root_com_pos_w,
        "body_pos_w": robot.data.body_pos_w,
        "body_vel_w": robot.data.body_vel_w[..., :3],
        "nle": nle,
    }

def model_based_controller_dict(robot, r_dict, action, lcc_rand, physics_dt = 0.005):
    com_vel = robot.data.root_lin_vel_w  # (N, 3)
    base_angvel = robot.data.root_com_ang_vel_w
    com_pos = r_dict["com_pos"]
    body_pos_w = r_dict["body_pos_w"]
    base_quat = r_dict["base_quat"]
    pos, ff_torque, info = ft.step(com_pos, com_vel, r_dict["jacs"],
                             body_pos_w, base_quat,
                             base_angvel, action, r_dict["nle"], lcc_rand)
    info["com_vel"] = com_vel
    info["com_angvel"] = base_angvel
    # Update values in r_dict
    r_dict["com_pos"] = com_pos + com_vel * physics_dt
    r_dict["body_pos_w"] = body_pos_w + r_dict["body_vel_w"] * physics_dt
    rot_mag = torch.linalg.norm(base_angvel, dim=-1, keepdim=False) + 1e-8
    axis = base_angvel / rot_mag[..., None]
    small_quat = quat_from_angle_axis(rot_mag * physics_dt, axis)
    r_dict["base_quat"] = quat_mul(small_quat, base_quat)
    return pos, ff_torque, info

def make_ft_rew_dict(robot, contact_mask, info, linacc, angacc, r_dict):
    ft_rew_dict = {
        "applied_torque": robot.data.applied_torque,
        "contact_mask": contact_mask,
        "grf": info["f"],
        "ff_tau": info["candidate_tau"],
        "w": info["w"],
        "des_com_vel": info["com_vel"],
        "des_com_angvel": info["com_angvel"],
        "com_acc": info["com_acc"],
        "com_angacc": info["com_angacc"],
        "lin_acc": linacc,
        "ang_acc": angacc,
        "nle": r_dict["nle"],
    }
    return ft_rew_dict

class FTActionManager(ActionManager):
    @property
    def total_action_dim(self) -> int:
        """Total action dimension."""
        return ft.CTRL_NUM * 2 + 7 + ft.EEF_NUM
    
    def process_action(self, action: torch.Tensor):
        if self.total_action_dim != action.shape[1]:
            raise ValueError(f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}.")
        # store the input actions
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)

    def update_torques(self, pos, torque, kp):
        #idx = 0
        for term_name, term in self._terms.items():
            pos_offset = torque / kp
            new_pos = pos + pos_offset
            if term_name == "joint_pos":
                term_actions = new_pos
            #else:  # torque
            #    term_actions = torque
            term.process_actions(term_actions)
            #idx += term.action_dim

def get_pd_gains_in_dof_order(robot, num_envs, device=None):
    """Return (kp, kd) shaped (num_envs, num_dof) aligned with robot.data.joint_pos.

    This supports Isaac Lab actuator configs such as ImplicitActuatorCfg where joints are
    selected via regex expressions (typically `joint_names_expr`) and gains may be scalars
    or dicts keyed by the same regex expressions.
    """
    device = device or robot.device

    dof_names = list(robot.data.joint_names)
    num_dof = len(dof_names)

    kp = torch.zeros((num_envs, num_dof), device=device)
    kd = torch.zeros((num_envs, num_dof), device=device)

    def _matches(expr: str, name: str) -> bool:
        # Treat actuator expressions as regex. Use fullmatch when possible, otherwise match.
        try:
            return (re.fullmatch(expr, name) is not None) or (re.match(expr, name) is not None)
        except re.error:
            # If expr isn't a valid regex, fall back to exact match.
            return expr == name

    def _value_for_expr(val, expr: str):
        # val can be scalar-like or dict keyed by expr.
        if isinstance(val, dict):
            # Prefer exact key match; otherwise allow regex key to match the expr.
            if expr in val:
                return val[expr]
            for k, v in val.items():
                try:
                    if re.fullmatch(k, expr) or re.match(k, expr):
                        return v
                except re.error:
                    continue
            # Not found -> treat as 0.0
            return 0.0
        return val

    for _, actuator in robot.cfg.actuators.items():
        # ImplicitActuatorCfg typically exposes joint_names_expr
        if hasattr(actuator, "joint_names_expr"):
            exprs = list(actuator.joint_names_expr)
            for expr in exprs:
                kp_val = _value_for_expr(getattr(actuator, "stiffness", 0.0), expr)
                kd_val = _value_for_expr(getattr(actuator, "damping", 0.0), expr)

                # Convert to python floats if tensors are provided
                if torch.is_tensor(kp_val):
                    kp_val = kp_val.item()
                if torch.is_tensor(kd_val):
                    kd_val = kd_val.item()

                idxs = [i for i, n in enumerate(dof_names) if _matches(expr, n)]
                if len(idxs) == 0:
                    continue

                kp[:, idxs] = float(kp_val)
                kd[:, idxs] = float(kd_val)

            continue

        # Fallback path for explicit joint lists (if present in other actuator types)
        if hasattr(actuator, "joint_names"):
            joint_names = list(actuator.joint_names)
            kps = torch.as_tensor(getattr(actuator, "stiffness", 0.0), device=device).flatten()
            kds = torch.as_tensor(getattr(actuator, "damping", 0.0), device=device).flatten()

            if kps.numel() == 1:
                kps = kps.repeat(len(joint_names))
            if kds.numel() == 1:
                kds = kds.repeat(len(joint_names))

            if kps.numel() != len(joint_names) or kds.numel() != len(joint_names):
                raise ValueError(
                    f"Actuator gains size mismatch: {len(joint_names)=}, {kps.numel()=}, {kds.numel()=}"
                )

            name_to_idx = {n: i for i, n in enumerate(dof_names)}
            for j, jname in enumerate(joint_names):
                if jname not in name_to_idx:
                    continue
                dof_i = name_to_idx[jname]
                kp[:, dof_i] = kps[j]
                kd[:, dof_i] = kds[j]
            continue

        raise AttributeError(
            f"Unsupported actuator cfg type: {type(actuator).__name__}. "
            "Expected joint_names_expr or joint_names."
        )

    return kp, kd

class FTEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: TrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.ft_rew_info = None  # placeholder for ft reward info
        self.sensor_cfg = SceneEntityCfg(
                "contact_forces",
                body_names=EEF_BODIES
            )
        self.lcc_bias = {
            "com_vel": torch.zeros((self.num_envs, 3), device=self.device),
            "com_angvel": torch.zeros((self.num_envs, 3), device=self.device),
            "mass_fac": torch.ones((self.num_envs,), device = self.device),
            "i_fac": torch.ones((self.num_envs, 3, 3), device = self.device),
            "jac_fac": torch.ones((self.num_envs, 24, 29 + 6), device = self.device),
            "pos": torch.zeros((self.num_envs, 5, 3), device = self.device)
        }
        self.sensor_cfg.resolve(self.scene)
        self.kp, self.kd = get_pd_gains_in_dof_order(self.scene["robot"], 
                                                     self.num_envs,
                                                     device=self.device)
        

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        print("[INFO] Event Manager: ", self.event_manager)
        # -- recorder manager
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        # -- action manager
        # -- observation manager
        self.action_manager = FTActionManager(self.cfg.actions, self)

        self.observation_manager = ObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)


        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")


    def step(self, action: torch.Tensor):
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        # perform physics stepping
        r_dict = robot_dict(self.scene["robot"])
        prev_vel = self.scene["robot"].data.root_lin_vel_w.clone()
        prev_angvel = self.scene["robot"].data.root_com_ang_vel_w.clone()
        lin_acc = torch.zeros_like(prev_vel, device=prev_vel.device)
        ang_acc = torch.zeros_like(prev_angvel, device=prev_angvel.device)
        for i in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            
            #pos, torque, info = model_based_controller(self.scene["robot"], self.action_manager._action)
            
            #r_dict["com_vel"] = self.scene["robot"].data.root_lin_vel_w
            #r_dict["base_angvel"] = self.scene["robot"].data.root_com_ang_vel_w
            pos, torque, info = model_based_controller_dict(self.scene["robot"],
                                                            r_dict, 
                                                            self.action_manager._action,
                                                            self.lcc_bias,
                                                            physics_dt = self.physics_dt)
            self.action_manager.update_torques(pos, torque, self.kp)
            self.action_manager.apply_action()

            # Calculate acceleration
            lin_acc = (info["com_vel"] - prev_vel) / self.physics_dt
            ang_acc = (info["com_angvel"] - prev_angvel) / self.physics_dt
            prev_vel = info["com_vel"]
            prev_angvel = info["com_angvel"]

            #print(f"[DEBUG] FT controller time: {time.perf_counter() - st:.6f} sec")
            # set actions into simulator
            #st = time.perf_counter()
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

            #print(f"[DEBUG] Physics step time: {time.perf_counter() - st:.6f} sec")

        with torch.no_grad():
            #action_ = action.clone()
            contact_sensor = self.scene.sensors[self.sensor_cfg.name]
            net_forces_w = contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids]
            contact_mask = (torch.linalg.norm(net_forces_w, dim=-1) > 10.0)  # (N, |body_ids|)
            self.ft_rew_info = make_ft_rew_dict(self.scene["robot"], 
                                                contact_mask,
                                                info, lin_acc, ang_acc, r_dict)
        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras