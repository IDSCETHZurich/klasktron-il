import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from ..mdp.observations import (
    KalmanFilter,
    angle_ball_goal,
    angle_ball_opp,
    body_xy_pos_w,
    distance_ball_to_player,
    distance_to_goal,
)
from .parameters import KLASK_PARAMS


@configclass
class ObservationsCfg:
    # TODO: noise corruption
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        peg_1_pos = ObservationTermCfg(
            func=body_xy_pos_w,
            params={"asset_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"])},
        )

        peg_1_x_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["slider_to_peg_1"]
                )
            },
        )

        peg_1_y_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["ground_to_slider_1"]
                )
            },
        )

        peg_2_pos = ObservationTermCfg(
            func=body_xy_pos_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask",
                    body_names=["Peg_2"],
                )
            },
        )

        peg_2_x_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["slider_to_peg_2"]
                )
            },
        )

        peg_2_y_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["ground_to_slider_2"]
                )
            },
        )

        ball_state = ObservationTermCfg(
            func=KalmanFilter,
            params={
                "process_variance_pos": 0.0008285,
                "process_variance_vel": 0.0008285,
                "measurement_variance_pos": 0.0008285,
                "x_limits": (-0.1532725, 0.1532725),
                "y_limits": (-0.2195525, 0.2195525),
                "distance": 0.029,
                "peg_1": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                "peg_2": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                "ball": SceneEntityCfg(name="ball"),
            },
        )

        if KLASK_PARAMS["observations"].get("additional_observations", 0):
            angle_pegball_pegoppgoal = ObservationTermCfg(
                func=angle_ball_goal,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                    "goal": KLASK_PARAMS["scene"]["opponent_goal"],
                },
            )

            angle_oppball_oppgoal = ObservationTermCfg(
                func=angle_ball_goal,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                    "goal": KLASK_PARAMS["scene"]["player_goal"],
                },
            )

            angle_pegball_pegopp = ObservationTermCfg(
                func=angle_ball_opp,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_1_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                    "player_2_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                },
            )

            angle_oppball_opppeg = ObservationTermCfg(
                func=angle_ball_opp,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_1_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                    "player_2_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                },
            )

            distance_ball_goal = ObservationTermCfg(
                func=distance_to_goal,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "goal": KLASK_PARAMS["scene"]["player_goal"],
                },
            )
            distance_ball_oppgoal = ObservationTermCfg(
                func=distance_to_goal,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "goal": KLASK_PARAMS["scene"]["opponent_goal"],
                },
            )

            distance_ball_player = ObservationTermCfg(
                func=distance_ball_to_player,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                },
            )

            distance_ball_opp = ObservationTermCfg(
                func=distance_ball_to_player,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                },
            )
        if KLASK_PARAMS["observations"]["action_history"] > 0:
            action_history_x = ObservationTermCfg(
                func=mdp.last_action,
                params={"action_name": "player_x"},
                history_length=KLASK_PARAMS["observations"]["action_history"],
            )

            action_history_y = ObservationTermCfg(
                func=mdp.last_action,
                params={"action_name": "player_y"},
                history_length=KLASK_PARAMS["observations"]["action_history"],
            )

    @configclass
    class OpponentCfg(ObservationGroupCfg):
        peg_2_pos = ObservationTermCfg(
            func=body_xy_pos_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask",
                    body_names=["Peg_2"],
                )
            },
        )

        peg_2_x_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["slider_to_peg_2"]
                )
            },
        )

        peg_2_y_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["ground_to_slider_2"]
                )
            },
        )

        peg_1_pos = ObservationTermCfg(
            func=body_xy_pos_w,
            params={"asset_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"])},
        )

        peg_1_x_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["slider_to_peg_1"]
                )
            },
        )

        peg_1_y_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["ground_to_slider_1"]
                )
            },
        )

        ball_state = ObservationTermCfg(
            func=KalmanFilter,
            params={
                "process_variance_pos": 0.0008285,
                "process_variance_vel": 0.0008285,
                "measurement_variance_pos": 0.0008285,
                "x_limits": (-0.1532725, 0.1532725),
                "y_limits": (-0.2195525, 0.2195525),
                "distance": 0.029,
                "peg_1": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                "peg_2": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                "ball": SceneEntityCfg(name="ball"),
            },
        )

        if KLASK_PARAMS["observations"].get("additional_observations", 0):
            angle_oppball_oppgoal = ObservationTermCfg(
                func=angle_ball_goal,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                    "goal": KLASK_PARAMS["scene"]["player_goal"],
                },
            )

            angle_pegball_pegoppgoal = ObservationTermCfg(
                func=angle_ball_goal,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                    "goal": KLASK_PARAMS["scene"]["opponent_goal"],
                },
            )

            angle_oppball_opppeg = ObservationTermCfg(
                func=angle_ball_opp,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_1_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                    "player_2_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                },
            )

            angle_pegball_pegopp = ObservationTermCfg(
                func=angle_ball_opp,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_1_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                    "player_2_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                },
            )

            distance_ball_oppgoal = ObservationTermCfg(
                func=distance_to_goal,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "goal": KLASK_PARAMS["scene"]["opponent_goal"],
                },
            )
            distance_ball_goal = ObservationTermCfg(
                func=distance_to_goal,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "goal": KLASK_PARAMS["scene"]["player_goal"],
                },
            )

            distance_ball_opp = ObservationTermCfg(
                func=distance_ball_to_player,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                },
            )

            distance_ball_player = ObservationTermCfg(
                func=distance_ball_to_player,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                },
            )
        if KLASK_PARAMS["observations"]["action_history"] > 0:
            action_history_x = ObservationTermCfg(
                func=mdp.last_action,
                params={"action_name": "opponent_x"},
                history_length=KLASK_PARAMS["observations"]["action_history"],
            )

            action_history_y = ObservationTermCfg(
                func=mdp.last_action,
                params={"action_name": "opponent_y"},
                history_length=KLASK_PARAMS["observations"]["action_history"],
            )

    policy: PolicyCfg = PolicyCfg()
    opponent: OpponentCfg = OpponentCfg()
