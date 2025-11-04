from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ActionManager
import torch
from whole_body_tracking.utils import ft
from isaaclab.managers import ActionManager, EventManager, ObservationManager, RecorderManager
from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from isaaclab.ui.widgets import ManagerLiveVisualizer
import time
# Implementation of FT environment. Idea is to implement the pure FT
# function and save a ft_info dict as part of the env class
# No observation change, contact rewards, centroid velocity rewards

# Implementation note: build a custom ActionManager with overriden action size
# ActionManager should have the big action size


def model_based_controller(robot, action):
    body_pos_w = robot.data.body_pos_w

    jacs = robot.root_physx_view.get_jacobians()

    cor_nle = robot.root_physx_view.get_coriolis_and_centrifugal_forces()
    grav_nle = robot.root_physx_view.get_generalized_gravity_forces()
    nle = cor_nle + grav_nle
    
    # Base position (world): pos (3) + quat (4)
    
    base_quat = robot.data.root_link_quat_w  # (N, 3)

    joint_vel = robot.data.joint_vel  # (N, num_joints)

    base_angvel = robot.data.root_com_ang_vel_b

    com_pos = robot.data.root_link_pos_w  # (N, 3)
    com_vel = robot.data.root_com_lin_vel_w  # (N, 3)
    
    pos, ff_torque = ft.jit_step(com_pos, com_vel, jacs, body_pos_w, 
                             base_quat, base_angvel, joint_vel, action)
    #ff_torque = action[:, 23:46] * 0.05
    #ff_torque += nle

    return pos, ff_torque

def make_ft_rew_dict(robot, action):
    body_pos_w = robot.data.body_pos_w

    jacs = robot.root_physx_view.get_jacobians()
    # Base position (world): pos (3) + quat (4)
    
    base_quat = robot.data.root_link_quat_w  # (N, 3)

    joint_vel = robot.data.joint_vel  # (N, num_joints)

    base_angvel = robot.data.root_com_ang_vel_w

    com_pos = robot.data.root_link_pos_w  # (N, 3)
    com_vel = robot.data.root_com_lin_vel_w  # (N, 3)

    ft_rew_dict = ft.ft_rew_info(com_pos, com_vel, jacs, body_pos_w, 
                             base_quat, base_angvel, joint_vel, action)
    
    return ft_rew_dict

class FTActionManager(ActionManager):
    @property
    def total_action_dim(self) -> int:
        """Total action dimension."""
        return 53 + ft.EEF_NUM
    
    def process_action(self, action: torch.Tensor):
        if self.total_action_dim != action.shape[1]:
            raise ValueError(f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}.")
        # store the input actions
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)

    def update_torques(self, pos, torque):
        idx = 0
        for term_name, term in self._terms.items():
            if term_name == "joint_pos":
                term_actions = pos
            else:  # torque
                term_actions = torque
            term.process_actions(term_actions)
            idx += term.action_dim

class FTEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: TrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.ft_rew_info = None  # placeholder for ft reward info

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

        with torch.no_grad():
            action_ = action.clone()
            self.ft_rew_info = make_ft_rew_dict(self.scene["robot"], self.action_manager._action)

        # perform physics stepping
        for i in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            if i % 1 == 0:
                pos, torque = model_based_controller(self.scene["robot"], self.action_manager._action)
                self.action_manager.update_torques(pos, torque)
            self.action_manager.apply_action()
            # set actions into simulator
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