import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from ..mdp.rewards import ball_in_goal, in_goal
from .parameters import KLASK_PARAMS


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    goal_scored = TerminationTermCfg(
        func=ball_in_goal,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["scene"]["opponent_goal"],
            "max_ball_vel": KLASK_PARAMS["scene"]["max_ball_vel_goal"],
        },
    )

    goal_conceded = TerminationTermCfg(
        func=ball_in_goal,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["scene"]["player_goal"],
            "max_ball_vel": KLASK_PARAMS["scene"]["max_ball_vel_goal"],
        },
    )

    player_in_goal = TerminationTermCfg(
        func=in_goal,
        params={
            "asset_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "goal": KLASK_PARAMS["scene"]["player_goal"],
        },
    )

    opponent_in_goal = TerminationTermCfg(
        func=in_goal,
        params={
            "asset_cfg": SceneEntityCfg("klask", body_names=["Peg_2"]),
            "goal": KLASK_PARAMS["scene"]["opponent_goal"],
        },
    )
