from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .config import (
    KLASK_PARAMS,
    ActionsCfg,
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    SceneCfg,
    TerminationsCfg,
)


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    decimation = KLASK_PARAMS["decimation"]
    sim = SimulationCfg(
        physx=PhysxCfg(bounce_threshold_velocity=0.0),
        render_interval=KLASK_PARAMS["decimation"],
        dt=KLASK_PARAMS["physics_dt"],
    )
    viewer = ViewerCfg(eye=(0.0, 0.0, 1.0), lookat=(0.0, 0.0, 0.0))
    episode_length_s = KLASK_PARAMS["episode_length_s"]
    scene = SceneCfg(num_envs=KLASK_PARAMS["scene"]["num_envs"], env_spacing=1.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    curriculum = CurriculumCfg()
    terminations = TerminationsCfg()
