import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from ..mdp.events import reset_joints_by_offset
from .parameters import KLASK_PARAMS


@configclass
class EventCfg:
    # on reset
    reset_peg_1_x = EventTermCfg(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["slider_to_peg_1"]),
            "position_range": KLASK_PARAMS["events"]["reset_peg_1_x"]["position_range"],
            "velocity_range": KLASK_PARAMS["events"]["reset_peg_1_x"]["velocity_range"],
        },
    )

    reset_peg_1_y = EventTermCfg(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["ground_to_slider_1"]),
            "position_range": KLASK_PARAMS["events"]["reset_peg_1_y"]["position_range"],
            "velocity_range": KLASK_PARAMS["events"]["reset_peg_1_y"]["velocity_range"],
        },
    )

    reset_peg_2_x = EventTermCfg(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["slider_to_peg_2"]),
            "position_range": KLASK_PARAMS["events"]["reset_peg_2_x"]["position_range"],
            "velocity_range": KLASK_PARAMS["events"]["reset_peg_2_x"]["velocity_range"],
        },
    )

    reset_peg_2_y = EventTermCfg(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["ground_to_slider_2"]),
            "position_range": KLASK_PARAMS["events"]["reset_peg_2_y"]["position_range"],
            "velocity_range": KLASK_PARAMS["events"]["reset_peg_2_y"]["velocity_range"],
        },
    )

    reset_ball = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {
                "x": KLASK_PARAMS["events"]["reset_ball"]["pose_range"]["x"],
                "y": KLASK_PARAMS["events"]["reset_ball"]["pose_range"]["y"],
                "z": KLASK_PARAMS["events"]["reset_ball"]["pose_range"]["z"],
            },
            "velocity_range": {
                "x": KLASK_PARAMS["events"]["reset_ball"]["velocity_range"]["x"],
                "y": KLASK_PARAMS["events"]["reset_ball"]["velocity_range"]["y"],
                "z": KLASK_PARAMS["events"]["reset_ball"]["velocity_range"]["z"],
            },
        },
    )

    # Domain Randomization:

    randomize_ball_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "mass_distribution_params": KLASK_PARAMS["events"]["randomize_ball_mass"][
                "mass_distribution_params"
            ],
            "operation": "abs",
        },
    )

    randomize_ball_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "static_friction_range": KLASK_PARAMS["events"]["randomize_ball_material"][
                "static_friction_range"
            ],
            "dynamic_friction_range": KLASK_PARAMS["events"]["randomize_ball_material"][
                "dynamic_friction_range"
            ],
            "restitution_range": KLASK_PARAMS["events"]["randomize_ball_material"][
                "restitution_range"
            ],
            "num_buckets": 100,
            "make_consistent": True,
        },
    )

    randomize_board_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask"),
            "static_friction_range": KLASK_PARAMS["events"]["randomize_board_material"][
                "static_friction_range"
            ],
            "dynamic_friction_range": KLASK_PARAMS["events"][
                "randomize_board_material"
            ]["dynamic_friction_range"],
            "restitution_range": KLASK_PARAMS["events"]["randomize_board_material"][
                "restitution_range"
            ],
            "num_buckets": 100,
            "make_consistent": True,
        },
    )
