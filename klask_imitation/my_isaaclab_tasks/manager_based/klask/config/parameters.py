KLASK_PARAMS = {
    "decimation": 20,  # system is running at 50Hz (night shift with 100Hz)
    "physics_dt": 0.001,
    "episode_length_s": 5.0,
    "action_history": 0,
    "edge": (-0.16, 0.16, -0.22, -0.02),
    "scene": {
        "num_envs": 4096,
        "ball": {
            "radius": 0.007,
        },
        "player_goal": (0.0, -0.176215, 0.01905),
        "opponent_goal": (0.0, 0.176215, 0.01905),
        "max_ball_vel_goal": 100.0,  # maximum speed the ball may have for a goal to be counted
    },
    "events": {
        "reset_peg_1_x": {
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
        "reset_peg_1_y": {
            "position_range": (-0.14, -0.03),
            "velocity_range": (0.0, 0.0),
        },
        "reset_peg_2_x": {
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
        "reset_peg_2_y": {
            "position_range": (0.03, 0.14),
            "velocity_range": (0.0, 0.0),
        },
        "reset_ball": {
            "pose_range": {
                "x": (-0.15, 0.15),
                "y": (-0.21, 0.21),
                "z": (0.032, 0.032),
            },
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.0, 0.0),
            },
        },
        "randomize_ball_mass": {
            "mass_distribution_params": (0.0018, 0.0022),
        },
        "randomize_ball_material": {
            "static_friction_range": (0.2, 0.5),
            "dynamic_friction_range": (0.3, 0.45),
            "restitution_range": (0.95, 0.99),
        },
        "randomize_board_material": {
            "static_friction_range": (0.2, 0.5),
            "dynamic_friction_range": (0.3, 0.45),
            "restitution_range": (0.95, 0.99),
        },
    },
    "observations": {
        "action_history": 0,
        "additional_observations": False,
    },
}

