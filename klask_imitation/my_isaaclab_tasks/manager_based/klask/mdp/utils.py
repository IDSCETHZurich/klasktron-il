import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _zeros_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros((int(env.num_envs), 2), dtype=torch.float32, device=env.device)


def _asset_has_data(asset: object) -> bool:
    return hasattr(asset, "_data") and getattr(asset, "_data") is not None


def root_xy_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Asset root position in the environment frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    if not _asset_has_data(asset):
        return _zeros_xy(env)

    pos = asset.data.root_pos_w[:, :2]
    if hasattr(env.scene, "env_origins"):
        pos = pos - env.scene.env_origins[:, :2]
    return pos


def root_lin_xy_vel_w(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Asset root linear velocity in the environment frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    if not _asset_has_data(asset):
        return _zeros_xy(env)
    return asset.data.root_lin_vel_w[:, :2]

def body_xy_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Return (num_envs, 2) xy-position of the specified body/bodies.
    During manager construction, articulation data may not be initialized yet.
    In that phase we return zeros so observation-shape inference can proceed.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if not _asset_has_data(asset):
        return _zeros_xy(env)

    body_ids = asset_cfg.body_ids
    if body_ids is None:
        pos = asset.data.body_pos_w[:, :1, :2]
    else:
        pos = asset.data.body_pos_w[:, body_ids, :2]
    if pos.ndim == 3:
        pos = pos[:, 0, :]

    if hasattr(env.scene, "env_origins"):
        pos = pos - env.scene.env_origins[:, :2]
    return pos


def body_lin_xy_vel_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Return (num_envs, 2) xy-velocity of the specified body/bodies.
    If body_ids has multiple entries, the first one is used.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if not _asset_has_data(asset):
        return _zeros_xy(env)

    body_ids = asset_cfg.body_ids
    data = asset.data
    if hasattr(data, "body_lin_vel_w"):
        full_vel = data.body_lin_vel_w
    elif hasattr(data, "body_vel_w"):
        full_vel = data.body_vel_w
    else:
        return _zeros_xy(env)

    if body_ids is None:
        vel = full_vel[:, :1, :2]
    else:
        vel = full_vel[:, body_ids, :2]
    if vel.ndim == 3:
        vel = vel[:, 0, :]

    return vel


def distance_player_ball(
    env: ManagerBasedRLEnv, player_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg
) -> torch.Tensor:
    return torch.sqrt(
        torch.sum(
            (root_xy_pos_w(env, ball_cfg) - body_xy_pos_w(env, player_cfg)) ** 2, dim=1
        )
    )


def speed(vel: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2)


def difference_speed(
    env: ManagerBasedRLEnv,
    player_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Speed difference between player body and ball, shape (num_envs,).

    Uses:
      - body_lin_xy_vel_w for the player (Articulation body)
      - root_lin_xy_vel_w for the ball (RigidObject root)

    Adds a small epsilon to avoid division by zero in rewards.
    """
    # (N, 2)
    vel_player = body_lin_xy_vel_w(env, player_cfg)
    vel_ball = root_lin_xy_vel_w(env, ball_cfg)

    diff = vel_player - vel_ball  # (N, 2)
    speed_diff = torch.norm(diff, dim=1)  # (N,)

    return speed_diff + eps
