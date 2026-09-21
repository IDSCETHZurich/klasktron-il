"""Observation/-dimension helpers for driving RL-Games checkpoints.

The frozen PPO expert shipped in ``klask_imitation/assets/ppo_expert`` consumes a
20-dimensional observation: the 12-D benchmark state in the environment's policy
layout (``peg1_pos, peg1_vel, peg2_pos, peg2_vel, ball_pos, ball_vel``) plus eight
engineered geometric channels -- four angles then four distances. Those eight are
reconstructed here from the 12-D state.

This module is the single source of truth for that reconstruction so that the
evaluation scripts and ``scripts/tests/play.py`` cannot drift apart. Feeding the
expert a different set of eight channels (for example an action history of the
same width) lines up numerically but is semantically wrong, and the expert will
act on noise rather than failing loudly.
"""

from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch

KLASK_BASE_OBS_DIM = 12
KLASK_EXPERT_OBS_DIM = 20


def _goals() -> tuple:
    """Return (player_goal, opponent_goal).

    Imported lazily: the ``my_isaaclab_tasks`` package pulls in Isaac Lab, which
    is only importable once the simulation app has been launched. Keeping this
    out of module scope lets the checkpoint-inspection helpers be imported (and
    unit-tested) without Isaac Sim.
    """
    from my_isaaclab_tasks.manager_based.klask.config.parameters import KLASK_PARAMS

    return (
        KLASK_PARAMS["scene"]["player_goal"],  # (cx, cy, r)
        KLASK_PARAMS["scene"]["opponent_goal"],
    )


def _load_rl_games_checkpoint_payload(checkpoint_path: str) -> Dict[str, Any]:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected RL-Games checkpoint payload: {checkpoint_path}")
    return payload


def _infer_action_dim_from_rl_games_checkpoint(checkpoint_path: str) -> int:
    payload = _load_rl_games_checkpoint_payload(checkpoint_path)
    model = payload.get("model", payload)
    if not isinstance(model, dict):
        raise ValueError(f"Checkpoint missing model state dict: {checkpoint_path}")
    for key in ("a2c_network.sigma", "a2c_network.mu.bias"):
        tensor = model.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 1 and tensor.shape[0] > 0:
            return int(tensor.shape[0])
    mu_w = model.get("a2c_network.mu.weight")
    if isinstance(mu_w, torch.Tensor) and mu_w.ndim >= 2 and mu_w.shape[0] > 0:
        return int(mu_w.shape[0])
    raise ValueError(f"Could not infer RL-Games action dim from checkpoint: {checkpoint_path}")


def _infer_obs_dim_from_rl_games_checkpoint(checkpoint_path: str) -> int:
    payload = _load_rl_games_checkpoint_payload(checkpoint_path)
    model = payload.get("model", payload)
    if not isinstance(model, dict):
        raise ValueError(f"Checkpoint missing model state dict: {checkpoint_path}")
    rms = model.get("running_mean_std.running_mean")
    if isinstance(rms, torch.Tensor) and rms.ndim == 1 and rms.shape[0] > 0:
        return int(rms.shape[0])
    for key, tensor in model.items():
        if key.startswith("a2c_network.actor_mlp.") and key.endswith(".weight"):
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 2 and tensor.shape[1] > 0:
                return int(tensor.shape[1])
            break
    raise ValueError(f"Could not infer RL-Games obs dim from checkpoint: {checkpoint_path}")


def _compute_local_extra_obs_features(obs_arr: np.ndarray) -> np.ndarray:
    """Reconstruct engineered feature channels from local 12D Klask observations."""
    if obs_arr.ndim != 2 or obs_arr.shape[1] != 12:
        raise ValueError(f"Expected (N, 12) local observation array, got shape {obs_arr.shape}.")

    self_pos = obs_arr[:, 0:2]
    other_pos = obs_arr[:, 4:6]
    ball_pos = obs_arr[:, 8:10]
    goal_player, goal_opponent = _goals()
    goal_self = np.asarray(goal_player[:2], dtype=np.float32)[None, :]
    goal_other = np.asarray(goal_opponent[:2], dtype=np.float32)[None, :]

    def angle_player_ball_goal(player_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
        vec_goal = goal_pos - player_pos
        vec_ball = ball_pos - player_pos
        dot = np.sum(vec_goal * vec_ball, axis=1)
        norm_goal = np.linalg.norm(vec_goal, axis=1)
        norm_ball = np.linalg.norm(vec_ball, axis=1)
        cos_theta = dot / (norm_goal * norm_ball + 1e-8)
        return np.arccos(np.clip(cos_theta, -1.0, 1.0)).astype(np.float32)[:, None]

    def angle_player_ball_other(player_a: np.ndarray, player_b: np.ndarray) -> np.ndarray:
        vec_a_to_ball = ball_pos - player_a
        vec_a_to_b = player_b - player_a
        dot = np.sum(vec_a_to_ball * vec_a_to_b, axis=1)
        norm_a_to_ball = np.linalg.norm(vec_a_to_ball, axis=1)
        norm_a_to_b = np.linalg.norm(vec_a_to_b, axis=1)
        cos_theta = dot / (norm_a_to_ball * norm_a_to_b + 1e-8)
        return np.arccos(np.clip(cos_theta, -1.0, 1.0)).astype(np.float32)[:, None]

    def distance_ball_goal(goal_pos: np.ndarray) -> np.ndarray:
        return np.linalg.norm(ball_pos - goal_pos, axis=1).astype(np.float32)[:, None]

    def distance_ball_player(player_pos: np.ndarray) -> np.ndarray:
        return np.linalg.norm(ball_pos - player_pos, axis=1).astype(np.float32)[:, None]

    return np.concatenate(
        [
            angle_player_ball_goal(self_pos, goal_self),
            angle_player_ball_goal(other_pos, goal_other),
            angle_player_ball_other(self_pos, other_pos),
            angle_player_ball_other(other_pos, self_pos),
            distance_ball_goal(goal_other),
            distance_ball_goal(goal_self),
            distance_ball_player(self_pos),
            distance_ball_player(other_pos),
        ],
        axis=1,
    ).astype(np.float32)


def _augment_local_obs_to_expected_dim(obs_arr: np.ndarray, expected_dim: int) -> np.ndarray:
    current_dim = int(obs_arr.shape[1])
    if expected_dim == current_dim:
        return obs_arr
    if current_dim == 12 and expected_dim == 20:
        return np.concatenate([obs_arr.astype(np.float32), _compute_local_extra_obs_features(obs_arr)], axis=1)
    if expected_dim < current_dim:
        return obs_arr[:, :expected_dim].astype(np.float32)
    raise ValueError(
        f"Unsupported observation adaptation from current_dim={current_dim} to expected_dim={expected_dim}."
    )


def _build_env_info_for_dims(
    *,
    observation_space: gym.Space,
    action_space: gym.Space,
    action_dim: int,
    obs_dim: int,
) -> Dict[str, Any]:
    if not isinstance(observation_space, gym.spaces.Box):
        raise TypeError(f"Expected Box observation space, got {type(observation_space)!r}.")
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"Expected Box action space, got {type(action_space)!r}.")
    if action_dim < 1 or action_dim > int(action_space.shape[0]):
        raise ValueError(
            f"Invalid action_dim={action_dim}; action space shape={getattr(action_space, 'shape', None)}."
        )
    base_obs_dim = int(observation_space.shape[0])
    if obs_dim < 1:
        raise ValueError(
            f"Invalid obs_dim={obs_dim}; observation space shape={getattr(observation_space, 'shape', None)}."
        )

    player_action_space: gym.Space = action_space
    if action_dim != int(action_space.shape[0]):
        player_action_space = gym.spaces.Box(
            low=np.asarray(action_space.low[:action_dim], dtype=action_space.dtype),
            high=np.asarray(action_space.high[:action_dim], dtype=action_space.dtype),
            dtype=action_space.dtype,
        )

    player_obs_space: gym.Space = observation_space
    if obs_dim < base_obs_dim:
        player_obs_space = gym.spaces.Box(
            low=np.asarray(observation_space.low[:obs_dim], dtype=observation_space.dtype),
            high=np.asarray(observation_space.high[:obs_dim], dtype=observation_space.dtype),
            dtype=observation_space.dtype,
        )
    elif obs_dim > base_obs_dim:
        extra = int(obs_dim - base_obs_dim)
        low = np.concatenate(
            [
                np.asarray(observation_space.low, dtype=np.float32),
                np.full((extra,), -1.0e6, dtype=np.float32),
            ],
            axis=0,
        )
        high = np.concatenate(
            [
                np.asarray(observation_space.high, dtype=np.float32),
                np.full((extra,), 1.0e6, dtype=np.float32),
            ],
            axis=0,
        )
        player_obs_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    return {
        "observation_space": player_obs_space,
        "action_space": player_action_space,
        "agents": 1,
        "value_size": 1,
    }
