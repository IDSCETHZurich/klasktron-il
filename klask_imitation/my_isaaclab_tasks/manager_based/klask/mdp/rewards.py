import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .utils import (
    body_xy_pos_w,
    difference_speed,
    distance_player_ball,
    root_lin_xy_vel_w,
    root_xy_pos_w,
    speed,
)


def in_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    goal: tuple[float, float, float],
    weight: float | None = None,
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Check if asset located in circle
    cx, cy, r = goal
    asset_pos_rel = (
        asset.data.body_pos_w[:, asset_cfg.body_ids, :].squeeze()
        - env.scene.env_origins
    )
    bodies_in_goal = (asset_pos_rel[:, 0] - cx) ** 2 + (
        asset_pos_rel[:, 1] - cy
    ) ** 2 <= r**2

    if weight is not None:
        bodies_in_goal *= weight

    return bodies_in_goal


def ball_in_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    goal: tuple[float, float, float],
    max_ball_vel: float = 2.0,
    weight: float | None = None,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Check if ball located inside goal
    cx, cy, r = goal
    ball_pos_rel = asset.data.root_pos_w - env.scene.env_origins
    ball_in_goal = (ball_pos_rel[:, 0] - cx) ** 2 + (
        ball_pos_rel[:, 1] - cy
    ) ** 2 <= r**2

    # Check if ball is slower than max_vel_ball
    ball_slow = (
        asset.data.root_lin_vel_w[:, 0] ** 2 + asset.data.root_lin_vel_w[:, 1] ** 2
        <= max_ball_vel**2
    )

    if weight is None:
        ball_in_goal = ball_in_goal * ball_slow
    else:
        ball_in_goal = weight * ball_in_goal * ball_slow

    return ball_in_goal


def peg_in_defense_line_with_rebounds(
    env: ManagerBasedRLEnv,
    player_cfg: SceneEntityCfg,
    opponent_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    weight: float | None = None,
) -> torch.Tensor:
    """Rewards the player for blocking direct and rebound shot lines from opponent to ball.
    Includes reflections off the four walls.
    """
    # Positions
    ball_pos = root_xy_pos_w(env, ball_cfg)  # (N, 2)
    opponent_pos = body_xy_pos_w(env, opponent_cfg)  # (N, 2)
    player_pos = body_xy_pos_w(env, player_cfg)  # (N, 2)

    # Only consider when ball in opp half
    ball_in_opp_half = ball_pos[:, 1] > 0.0
    ball_is_close = distance_player_ball(env, player_cfg, ball_cfg) < 0.08

    # Reflect ball across 4 edges
    mirror_balls = [ball_pos]  # start with original ball

    mirror_opponents = [opponent_pos]
    # Reflect across x walls
    mirror_opponents.append(
        torch.stack([-0.32 - opponent_pos[:, 0], opponent_pos[:, 1]], dim=1)
    )  # left wall
    mirror_opponents.append(
        torch.stack([0.32 - opponent_pos[:, 0], opponent_pos[:, 1]], dim=1)
    )  # right wall

    mirror_balls.append(
        torch.stack([-0.32 - ball_pos[:, 0], ball_pos[:, 1]], dim=1)
    )  # left wall
    mirror_balls.append(
        torch.stack([0.32 - ball_pos[:, 0], ball_pos[:, 1]], dim=1)
    )  # right wall
    # Reflect across y walls ( this is unlikely )
    # mirror_balls.append(torch.stack([ball_pos[:, 0], -0.44 - ball_pos[:, 1]], dim=1))  # bottom wall
    # mirror_balls.append(torch.stack([ball_pos[:, 0], 0.44 - ball_pos[:, 1]], dim=1))   # top wall

    all_rewards = []

    for mirror_ball, mirror_opp in zip(mirror_balls, mirror_opponents):
        vec_ob = mirror_ball - mirror_opp
        vec_op = player_pos - mirror_opp

        # Dot product (batch)
        dot = torch.sum(vec_ob * vec_op, dim=1)
        norm_ob = torch.norm(vec_ob, dim=1)
        norm_op = torch.norm(vec_op, dim=1)

        # Angle (in radians)
        cos_theta = dot / (norm_ob * norm_op + 1e-6)
        angle = torch.acos(
            torch.clamp(cos_theta, -1.0, 1.0)
        )  # Clamp for numerical stability

        # Smaller angle = better blocking → higher reward
        reward = torch.exp(-1.0 * angle)  # Adjust 5.0 as needed
        all_rewards.append(reward)
    # Take maximum reward across all paths (best alignment)
    final_reward = torch.stack(all_rewards, dim=1).max(dim=1).values

    # Apply ball-in-own-half condition
    final_reward = final_reward * ball_is_close.float() * ball_in_opp_half.float()

    # Apply weight if needed
    if weight is not None:
        final_reward *= weight

    return final_reward


def shot_over_middle(
    env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg, weight: float | None = None
) -> torch.Tensor:
    ball_pos = root_xy_pos_w(env, ball_cfg)  # shape: (num_envs, 2)
    ball_vel = root_lin_xy_vel_w(env, ball_cfg)  # shape: (num_envs, 2)

    # Detect near center line and moving forward in +y direction
    is_near_center = (ball_pos[:, 1] >= 0.002) & (ball_pos[:, 1] <= 0.005)
    is_moving_forward = ball_vel[:, 1] > 0.0
    if weight == None:
        return (
            torch.abs(ball_vel[:, 1]) ** 4 * is_near_center * is_moving_forward
        ).float()
    return (
        weight
        * (torch.abs(ball_vel[:, 1]) ** 4 * is_near_center * is_moving_forward).float()
    )


def ball_speed(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    vel = root_lin_xy_vel_w(env, ball_cfg)
    return speed(vel)


def distance_ball_goal(
    env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg, goal: tuple[float, float, float]
) -> torch.Tensor:
    cx, cy, r = goal
    ball_pos = root_xy_pos_w(env, ball_cfg)
    return torch.exp(
        -5 * torch.sqrt((ball_pos[:, 0] - cx) ** 2 + (ball_pos[:, 1] - cy) ** 2)
    )  # factor 5 because distances are really small


def distance_player_ball_own_half(
    env: ManagerBasedRLEnv, player_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg
) -> torch.Tensor:
    ball_pos = root_xy_pos_w(env, ball_cfg)
    ball_in_own_half = ball_pos[:, 1] < 0.0
    return ball_in_own_half * (
        torch.exp(-5 * distance_player_ball(env, player_cfg, ball_cfg))
    )  # factor 5 because distances are really small


def ball_stationary(
    env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg, eps=5e-3
) -> torch.Tensor:
    return ball_speed(env, ball_cfg) < eps


def collision_player_ball(
    env: ManagerBasedRLEnv,
    player_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    eps=0.017,
) -> torch.Tensor:
    return (distance_player_ball(env, player_cfg, ball_cfg) < eps) * torch.exp(
        -0.1 / difference_speed(env, player_cfg, ball_cfg)
    )


def ball_in_own_half(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg):
    ball_pos = root_xy_pos_w(env, ball_cfg)
    return 1.0 * (ball_pos[:, 1] < 0.0)

'''
def distance_to_wall(
    env: ManagerBasedRLEnv, player_cfg: SceneEntityCfg, edge: tuple[float]
) -> torch.Tensor:
    player_pos = body_xy_pos_w(env, player_cfg)
    device = player_pos.device
    cost = torch.zeros(player_pos.shape[0], device=device)

    x_edge = player_pos[:, 0] < 0.03 + torch.tensor(edge[0])
    distance = player_pos[:, 0] - torch.tensor(edge[0])
    cost += x_edge * torch.exp(-5 * distance)

    x_edge = torch.tensor(edge[1]) - player_pos[:, 0] < 0.03
    distance = torch.tensor(edge[1]) - player_pos[:, 0]
    cost += x_edge * torch.exp(-5 * distance)

    y_edge = player_pos[:, 1] - torch.tensor(edge[2]) < 0.03
    distance = player_pos[:, 1] - torch.tensor(edge[2])
    cost += y_edge * torch.exp(-5 * distance)

    y_edge = torch.tensor(edge[3]) - player_pos[:, 1] < 0.03
    distance = torch.tensor(edge[3]) - player_pos[:, 1]
    cost += y_edge * torch.exp(-5 * distance)

    return 1.0 * (cost)
'''
def distance_to_wall(env, player_cfg, edge):
    player_pos = body_xy_pos_w(env, player_cfg)

    x = player_pos[:,0]
    y = player_pos[:,1]

    # Minimum distance to each boundary
    dist_left   = torch.clamp(x - edge[0], min=0)
    dist_right  = torch.clamp(edge[1] - x, min=0)
    dist_bottom = torch.clamp(y - edge[2], min=0)
    dist_top    = torch.clamp(edge[3] - y, min=0)

    # Distance to closest wall
    dist = torch.stack([dist_left, dist_right, dist_bottom, dist_top], dim=1).min(dim=1).values

    # Reward = negative when close, near zero when safe
    reward = torch.exp(-10 * dist)

    return reward
