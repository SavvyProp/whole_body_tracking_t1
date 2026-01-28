import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Tracking-FTFT-T1-v0",
    entry_point="whole_body_tracking.tasks.tracking.config.t1_ftft.ftft_env:FTFEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T1FTFTEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-FTFT-T1-Eval-v0",
    entry_point="whole_body_tracking.tasks.tracking.config.t1_ftft.ftft_env:FTFEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T1FTFTEnvEvalCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T1FlatPPORunnerCfg",
    },
)