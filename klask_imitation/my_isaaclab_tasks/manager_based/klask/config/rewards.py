import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from ..mdp.rewards import (
    ball_in_goal,
    ball_in_own_half,
    ball_speed,
    ball_stationary,
    collision_player_ball,
    distance_ball_goal,
    distance_player_ball_own_half,
    distance_to_wall,
    in_goal,
    peg_in_defense_line_with_rebounds,
    shot_over_middle,
)
from .parameters import KLASK_PARAMS


@configclass
class RewardsCfg:
    time_punishment = RewardTermCfg(func=mdp.is_alive, weight=0.0)

    time_out_punishment = RewardTermCfg(func=mdp.time_out, weight=0.0)

    shot_over_middle_line = RewardTermCfg(
        func=shot_over_middle,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=0.1,
    )

    player_in_goal = RewardTermCfg(
        func=in_goal,
        params={
            "asset_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "goal": KLASK_PARAMS["scene"]["player_goal"],
        },
        weight=-10.0,
    )

    opponent_in_goal = RewardTermCfg(
        func=in_goal,
        params={
            "asset_cfg": SceneEntityCfg("klask", body_names=["Peg_2"]),
            "goal": KLASK_PARAMS["scene"]["opponent_goal"],
        },
        weight=-10.0,
    )

    goal_scored = RewardTermCfg(
        func=ball_in_goal,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["scene"]["opponent_goal"],
            "max_ball_vel": KLASK_PARAMS["scene"]["max_ball_vel_goal"],
        },
        weight=10.0 #/ (KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"]),
    )

    goal_conceded = RewardTermCfg(
        func=ball_in_goal,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["scene"]["player_goal"],
            "max_ball_vel": KLASK_PARAMS["scene"]["max_ball_vel_goal"],
        },
        weight=-10.0,
    )

    distance_player_ball_own_half = RewardTermCfg(
        func=distance_player_ball_own_half,
        params={
            "player_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=0.0,
    )

    distance_ball_opponent_goal = RewardTermCfg(
        func=distance_ball_goal,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["scene"]["opponent_goal"],
        },
        weight=1.0,
    )

    distance_ball_own_goal = RewardTermCfg(
        func=distance_ball_goal,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["scene"]["player_goal"],
        },
        weight=0,
    )

    ball_speed = RewardTermCfg(
        func=ball_speed,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=0.1,
    )

    ball_stationary = RewardTermCfg(
        func=ball_stationary,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=-0.0,
    )

    collision_player_ball = RewardTermCfg(
        func=collision_player_ball,
        params={
            "player_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=0.5,
    )

    ball_in_own_half = RewardTermCfg(
        func=ball_in_own_half, params={"ball_cfg": SceneEntityCfg("ball")}, weight=-0
    )

    close_to_boundaries = RewardTermCfg(
        func=distance_to_wall,
        params={
            "player_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "edge": KLASK_PARAMS["edge"],
        },
        weight=-0.00,
    )
    player_strategically_positioned = RewardTermCfg(
        func=peg_in_defense_line_with_rebounds,
        params={
            "player_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "opponent_cfg": SceneEntityCfg("klask", body_names=["Peg_2"]),
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=0,
    )
