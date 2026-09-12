"""
Evaluate GAIL results:
- Performance metrics (goal scored/conceded, reward, episode length)
- Reward-net alignment with goal scoring
- Behavior heatmaps (agent vs demos)
- Optional discriminator overfitting analysis and marginalized reward maps

Outputs:
- summary.json
- episode_metrics.csv
- heatmap_agent.npy
- heatmap_demo.npy (if demos enabled)
- heatmap_agent_vs_demo.png
- velocity_histograms.png (if demos enabled; velocity direction angles)
- action_bias_vs_pos.png (if demos enabled; mean action_y vs peg_y for peg1 and peg2)
- discriminator_metrics.json (if --run-discriminator-analysis)
- reward_map_<target>.npy/.png (if --run-discriminator-analysis)
- reward_map_<target>_metadata.json (if --run-discriminator-analysis)
- visited_reward_map_<target>.npy/.png (if --run-discriminator-analysis)
- visited_reward_map_<target>_metadata.json (if --run-discriminator-analysis)
- discriminator_logits_hist.png (if --run-discriminator-analysis)
- peg1_ball_proximity_profile.png (if --run-discriminator-analysis)
- analysis_summary.json (if --run-discriminator-analysis)
"""

import sys
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import os
import json
import time
import pickle
import argparse
import copy
import datetime
import inspect
import random
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

import gymnasium as gym

from imitation.data.types import Trajectory, TrajectoryWithRew, maybe_wrap_in_dictobs
from imitation.rewards.reward_nets import BasicShapedRewardNet, RewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO

try:
    from my_isaaclab_tasks.manager_based.klask.imitation_reward_nets import (
        KlaskRelativeShapedRewardNet,
    )
except ModuleNotFoundError:
    _LOCAL_REWARD_NETS_PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "my_isaaclab_tasks",
            "manager_based",
            "klask",
            "imitation_reward_nets.py",
        )
    )
    _reward_net_spec = importlib.util.spec_from_file_location(
        "klask_imitation_reward_nets_local",
        _LOCAL_REWARD_NETS_PATH,
    )
    if _reward_net_spec is None or _reward_net_spec.loader is None:
        raise
    _reward_net_module = importlib.util.module_from_spec(_reward_net_spec)
    _reward_net_spec.loader.exec_module(_reward_net_module)
    KlaskRelativeShapedRewardNet = _reward_net_module.KlaskRelativeShapedRewardNet
from my_isaaclab_tasks.manager_based.klask.env_cfg import EnvCfg
from my_isaaclab_tasks.manager_based.klask.env_wrapper import KlaskCollisionAvoidanceWrapper
from my_isaaclab_tasks.manager_based.klask.actuator_model import ActuatorModelWrapper
from my_isaaclab_tasks.manager_based.klask.config.parameters import KLASK_PARAMS
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from discriminator_analysis_utils import (
    compute_peg_ball_proximity_profile,
    plot_peg_ball_proximity_profile,
)

# =============================================================================
# Constants
# =============================================================================

LONG_FACTOR = 0.0008285
SHORT_FACTOR = 1.0 / 1150.0
BOARD_LONG, BOARD_SHORT = 0.44, 0.32  # (long, short)

FIELD_X = (-BOARD_SHORT / 2, BOARD_SHORT / 2)
FIELD_Y = (-BOARD_LONG / 2, BOARD_LONG / 2)

GOAL_PLAYER = KLASK_PARAMS["scene"]["player_goal"]  # (cx, cy, r)
GOAL_OPPONENT = KLASK_PARAMS["scene"]["opponent_goal"]
MAX_BALL_VEL_GOAL = KLASK_PARAMS["scene"]["max_ball_vel_goal"]
DEFAULT_BC_OPPONENT_POLICY_PATH = (
    str(
        Path(__file__).resolve().parents[3]
        / "trained_models"
        / "data_wall_experiments"
        / "bc_runs"
        / "klask_bc_20260303_161251_HUMAN_DEMO_V2"
        / "ppo_policy_bc_final.zip"
    )
)
POLICY_FORMAT_SB3_PPO = "sb3_ppo"
POLICY_FORMAT_RL_GAMES = "rl_games"
DEMO_SOURCE_HUMAN_RAW = "human_raw"
DEMO_SOURCE_SIM_NATIVE = "sim_native"
REWARD_NET_TYPE_BASIC_SHAPED = "basic_shaped"
REWARD_NET_TYPE_KLASK_RELATIVE_SHAPED = "klask_relative_shaped"
PEG_BALL_PROXIMITY_DISTANCE_SAMPLES = 25
PEG_BALL_PROXIMITY_ANGLE_SAMPLES = 24


def _default_trained_models_dir() -> Path:
    return Path(
        os.environ.get(
            "KLASKTRON_TRAINED_MODELS_DIR",
            Path(__file__).resolve().parents[3] / "trained_models",
        )
    )


def _default_data_dir() -> Path:
    return Path(
        os.environ.get(
            "KLASKTRON_DATA_DIR",
            Path(__file__).resolve().parents[3] / "data",
        )
    )


def _default_artifacts_dir() -> Path:
    return Path(
        os.environ.get(
            "KLASKTRON_ARTIFACTS_DIR",
            Path(__file__).resolve().parents[3] / "artifacts",
        )
    )

# Demo observation layout after transform_observations:
# [bp_x, bp_y, bv_x, bv_y, lp_x, lp_y, rp_x, rp_y, lv_x, lv_y, rv_x, rv_y]
DEMO_LAYOUT = {
    "ball_pos": slice(0, 2),
    "ball_vel": slice(2, 4),
    "peg1_pos": slice(4, 6),  # lower peg in demos
    "peg2_pos": slice(6, 8),  # upper peg in demos
    "peg1_vel": slice(8, 10),
    "peg2_vel": slice(10, 12),
}

# Heuristics for splitting long demo trajectories into pseudo-episodes.
# Demos can contain many game rounds in one file; these thresholds help recover
# terminal boundaries from position jumps + reset-like states.

# =============================================================================
# Helpers
# =============================================================================

def unwrap_obs(obs):
    """Handle (obs, info) tuple from reset/step."""
    if isinstance(obs, tuple):
        return obs[0]
    return obs


def vec_step(venv, action):
    out = venv.step(action)
    if len(out) == 4:
        obs, reward, done, info = out
    else:
        obs, reward, terminated, truncated, info = out
        done = np.logical_or(terminated, truncated)
    return obs, reward, done, info


def obs_to_array(obs) -> np.ndarray:
    """Return the array-like observation used for metrics/heatmaps."""
    if isinstance(obs, dict):
        if "policy" in obs:
            return np.asarray(obs["policy"])
        return np.asarray(next(iter(obs.values())))
    return np.asarray(obs)


def ensure_2d_batch(arr: Any, dtype=np.float32) -> np.ndarray:
    """Normalize array-like inputs to shape (N, D)."""
    out = np.asarray(arr, dtype=dtype)
    if out.ndim == 1:
        out = out[None, :]
    return out


def to_torch_obs(obs, device: torch.device):
    """Convert obs (np array or dict) to torch tensors on device."""
    if isinstance(obs, dict):
        return {k: torch.as_tensor(v, device=device, dtype=torch.float32) for k, v in obs.items()}
    return torch.as_tensor(obs, device=device, dtype=torch.float32)


def build_policy_obs_slices(venv: Sb3VecEnvWrapper) -> Dict[str, slice]:
    """
    Build index slices for each observation term in the 'policy' group
    using the ObservationManager metadata (order is authoritative).
    """
    env = venv.unwrapped
    obs_man = env.observation_manager
    group = "policy"
    if group not in obs_man.active_terms:
        raise KeyError(f"Observation group '{group}' not found in observation manager")

    names = obs_man.active_terms[group]
    dims = obs_man.group_obs_term_dim[group]
    idx = 0
    slices: Dict[str, slice] = {}
    for name, dim in zip(names, dims):
        length = int(np.prod(dim))
        slices[name] = slice(idx, idx + length)
        idx += length
    return slices


def build_policy_layout(venv: Sb3VecEnvWrapper) -> Dict[str, slice]:
    """
    Derive slices from the policy observation layout.
    Expects terms: peg_1_pos, peg_2_pos, peg_1_x_vel, peg_1_y_vel, peg_2_x_vel, peg_2_y_vel, ball_state.
    """
    slices = build_policy_obs_slices(venv)
    required = (
        "peg_1_pos",
        "peg_2_pos",
        "peg_1_x_vel",
        "peg_1_y_vel",
        "peg_2_x_vel",
        "peg_2_y_vel",
        "ball_state",
    )
    missing = [k for k in required if k not in slices]
    if missing:
        raise KeyError(f"Missing expected terms in policy obs: {missing}")

    ball_state = slices["ball_state"]
    ball_pos = slice(ball_state.start, ball_state.start + 2)
    ball_vel = slice(ball_state.start + 2, ball_state.start + 4)

    return {
        "ball_pos": ball_pos,
        "ball_vel": ball_vel,
        "peg1_pos": slices["peg_1_pos"],
        "peg2_pos": slices["peg_2_pos"],
        "peg1_vel_x": slices["peg_1_x_vel"],
        "peg1_vel_y": slices["peg_1_y_vel"],
        "peg2_vel_x": slices["peg_2_x_vel"],
        "peg2_vel_y": slices["peg_2_y_vel"],
    }


def build_opponent_obs_from_policy_obs(
    obs: Any,
    policy_layout: Dict[str, slice],
    *,
    mirror_x: bool,
    mirror_y: bool,
) -> np.ndarray:
    """
    Build opponent-perspective observation from policy observation.
    - Always swaps peg1/peg2 position and velocity fields.
    - Optionally mirrors x/y components for ball and peg state.
    """
    src = ensure_2d_batch(obs_to_array(obs), dtype=np.float32)
    if src.shape[1] < 12:
        raise ValueError(f"Expected at least 12 observation dims, got {src.shape[1]}")

    dst = np.array(src, copy=True)

    # Swap peg position slices.
    p1 = policy_layout["peg1_pos"]
    p2 = policy_layout["peg2_pos"]
    p1_pos = src[:, p1].copy()
    p2_pos = src[:, p2].copy()
    dst[:, p1] = p2_pos
    dst[:, p2] = p1_pos

    # Swap peg velocity terms.
    p1_vx = policy_layout["peg1_vel_x"].start
    p1_vy = policy_layout["peg1_vel_y"].start
    p2_vx = policy_layout["peg2_vel_x"].start
    p2_vy = policy_layout["peg2_vel_y"].start
    dst[:, p1_vx] = src[:, p2_vx]
    dst[:, p1_vy] = src[:, p2_vy]
    dst[:, p2_vx] = src[:, p1_vx]
    dst[:, p2_vy] = src[:, p1_vy]

    if mirror_x:
        # Mirror x for pegs (positions + x velocities).
        dst[:, p1.start + 0] *= -1.0
        dst[:, p2.start + 0] *= -1.0
        dst[:, p1_vx] *= -1.0
        dst[:, p2_vx] *= -1.0

        # Mirror ball x position and x velocity.
        ball_pos_x = policy_layout["ball_pos"].start + 0
        ball_vel_x = policy_layout["ball_vel"].start + 0
        dst[:, ball_pos_x] *= -1.0
        dst[:, ball_vel_x] *= -1.0

    if mirror_y:
        # Mirror y for pegs (positions + y velocities).
        dst[:, p1.start + 1] *= -1.0
        dst[:, p2.start + 1] *= -1.0
        dst[:, p1_vy] *= -1.0
        dst[:, p2_vy] *= -1.0

        # Mirror ball y position and y velocity.
        ball_pos_y = policy_layout["ball_pos"].start + 1
        ball_vel_y = policy_layout["ball_vel"].start + 1
        dst[:, ball_pos_y] *= -1.0
        dst[:, ball_vel_y] *= -1.0

    return dst


def unmirror_opponent_action(
    action_opp: Any,
    *,
    mirror_x: bool,
    mirror_y: bool,
) -> np.ndarray:
    """
    Convert opponent-local action [ax, ay] into environment coordinates.
    Mirrored axes are flipped back.
    """
    out = ensure_2d_batch(action_opp, dtype=np.float32)
    if out.shape[1] < 2:
        raise ValueError(f"Opponent action must have at least 2 dims, got {out.shape}")
    opp_env = out[:, :2].copy()
    if mirror_x:
        opp_env[:, 0] *= -1.0
    if mirror_y:
        opp_env[:, 1] *= -1.0
    return opp_env


def build_self_play_action_from_peg1_policy(
    *,
    obs: Any,
    policy: PPO,
    policy_layout: Dict[str, slice],
    mirror_x: bool,
    mirror_y: bool,
    deterministic: bool = False,
) -> np.ndarray:
    """
    Build full 4D env action from a 2D Peg1 policy by reusing the same policy
    for Peg2 with remapped (and optionally mirrored) observations.
    """
    obs_policy = ensure_2d_batch(obs_to_array(obs), dtype=np.float32)

    act_peg1, _ = policy.predict(obs_policy, deterministic=deterministic)
    act_peg1 = ensure_2d_batch(act_peg1, dtype=np.float32)
    if act_peg1.shape[1] < 2:
        raise ValueError(f"Peg1 policy action must have at least 2 dims, got {act_peg1.shape}")
    act_peg1 = act_peg1[:, :2]

    obs_opp = build_opponent_obs_from_policy_obs(
        obs_policy,
        policy_layout,
        mirror_x=mirror_x,
        mirror_y=mirror_y,
    )
    act_opp_local, _ = policy.predict(obs_opp, deterministic=deterministic)
    act_opp = unmirror_opponent_action(
        act_opp_local,
        mirror_x=mirror_x,
        mirror_y=mirror_y,
    )

    return np.concatenate([act_peg1, act_opp], axis=1).astype(np.float32)


def build_training_style_opponent_obs(obs: Any) -> np.ndarray:
    """
    Remap observation to opponent perspective exactly like training
    OpponentVecEnvWrapper._to_opponent_obs:
      - swap first 8 dims as [4:8, 0:4]
    """
    obs_arr = ensure_2d_batch(obs_to_array(obs), dtype=np.float32)
    if obs_arr.shape[1] < 12:
        raise ValueError(f"Opponent obs remap expects >=12 dims, got {obs_arr.shape[1]}.")
    remapped = np.array(obs_arr, copy=True)
    remapped[:, :8] = obs_arr[:, [4, 5, 6, 7, 0, 1, 2, 3]]
    return remapped


def resolve_policy_format_value(
    raw_fmt: Optional[str],
    *,
    checkpoint: str,
    flag_name: str,
) -> str:
    raw = str(raw_fmt).strip().lower() if raw_fmt is not None else ""
    if raw in {"sb3", "sb3_ppo", "ppo"}:
        return POLICY_FORMAT_SB3_PPO
    if raw in {"rl_games", "rlgames", "rl"}:
        return POLICY_FORMAT_RL_GAMES
    if raw:
        raise ValueError(
            f"Unsupported {flag_name} {raw!r}. "
            f"Valid values: {POLICY_FORMAT_SB3_PPO}, {POLICY_FORMAT_RL_GAMES}."
        )
    return POLICY_FORMAT_SB3_PPO if str(checkpoint).lower().endswith(".zip") else POLICY_FORMAT_RL_GAMES


def _load_yaml_dict(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file does not contain a dictionary: {path}")
    return payload


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


def _adapt_obs_dim_tensor(obs: torch.Tensor, expected_dim: int) -> torch.Tensor:
    if obs.ndim != 2:
        raise ValueError(f"Expected 2D obs tensor, got shape {tuple(obs.shape)}")
    current_dim = int(obs.shape[1])
    if expected_dim == current_dim:
        return obs
    if expected_dim < current_dim:
        return obs[:, :expected_dim]
    raise ValueError(f"Checkpoint expects obs dim {expected_dim}, but env provides {current_dim}.")


def _compute_local_extra_obs_features(obs_arr: np.ndarray) -> np.ndarray:
    """Reconstruct engineered feature channels from local 12D Klask observations."""
    if obs_arr.ndim != 2 or obs_arr.shape[1] != 12:
        raise ValueError(f"Expected (N, 12) local observation array, got shape {obs_arr.shape}.")

    self_pos = obs_arr[:, 0:2]
    other_pos = obs_arr[:, 4:6]
    ball_pos = obs_arr[:, 8:10]
    goal_self = np.asarray(GOAL_PLAYER[:2], dtype=np.float32)[None, :]
    goal_other = np.asarray(GOAL_OPPONENT[:2], dtype=np.float32)[None, :]

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


class RlGamesPredictPolicyAdapter:
    """Expose an RL-Games player via an SB3-like `predict()` method."""

    def __init__(
        self,
        *,
        runner: Any,
        player: Any,
        device: str,
        obs_dim: int,
        action_dim: int,
    ):
        self.runner = runner
        self.player = player
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

    def predict(self, obs: Any, deterministic: bool = True):
        obs_arr = _augment_local_obs_to_expected_dim(
            ensure_2d_batch(obs_to_array(obs), dtype=np.float32),
            self.obs_dim,
        )
        obs_tensor = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            obs_t = self.player.obs_to_torch(obs_tensor)
            acts = self.player.get_action(obs_t, is_deterministic=bool(deterministic))
        if not torch.is_tensor(acts):
            acts = torch.as_tensor(acts, dtype=torch.float32, device=self.device)
        if acts.ndim == 1:
            acts = acts.unsqueeze(0)
        acts = acts[:, : max(1, self.action_dim)]
        acts_np = acts.detach().cpu().numpy().astype(np.float32)
        return np.clip(acts_np, -1.0, 1.0), None


def build_rl_games_predict_policy(
    *,
    checkpoint_path: str,
    config_path: str,
    device: str,
    num_envs: int,
    observation_space: gym.Space,
    action_space: gym.Space,
) -> RlGamesPredictPolicyAdapter:
    from rl_games.torch_runner import Runner

    cfg = copy.deepcopy(_load_yaml_dict(config_path))
    cfg.setdefault("params", {})
    cfg["params"].setdefault("config", {})
    cfg["params"]["config"]["device"] = device
    cfg["params"]["config"]["device_name"] = device
    cfg["params"]["config"]["num_actors"] = int(num_envs)

    action_dim = _infer_action_dim_from_rl_games_checkpoint(checkpoint_path)
    obs_dim = _infer_obs_dim_from_rl_games_checkpoint(checkpoint_path)
    env_info = _build_env_info_for_dims(
        observation_space=observation_space,
        action_space=action_space,
        action_dim=action_dim,
        obs_dim=obs_dim,
    )
    cfg["params"]["config"]["env_info"] = env_info
    cfg["params"]["config"]["vec_env"] = None

    runner = Runner()
    runner.load(cfg)
    player = runner.create_player()
    try:
        player.restore(checkpoint_path)
    except Exception as exc:
        print(
            f"[warn] RL-Games restore() failed ({exc}). "
            f"Falling back to direct torch.load(map_location={device})."
        )
        payload = _load_rl_games_checkpoint_payload(checkpoint_path)
        model_state = payload.get("model")
        if not isinstance(model_state, dict):
            raise RuntimeError(f"Checkpoint missing 'model' state: {checkpoint_path}") from exc
        player.model.load_state_dict(model_state)
        if getattr(player, "normalize_input", False) and "running_mean_std" in payload:
            player.model.running_mean_std.load_state_dict(payload["running_mean_std"])
    player.reset()
    player.device = torch.device(device)
    if hasattr(player, "model"):
        player.model.to(player.device)
    if hasattr(player, "action_space") and getattr(player.action_space, "shape", None) is not None:
        current_action_dim = int(player.action_space.shape[0])
        if current_action_dim != int(action_dim):
            low = np.asarray(player.action_space.low[:action_dim], dtype=np.float32)
            high = np.asarray(player.action_space.high[:action_dim], dtype=np.float32)
            player.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            if hasattr(player, "actions_num"):
                player.actions_num = int(action_dim)
    if hasattr(player, "actions_low"):
        player.actions_low = player.actions_low[:action_dim].to(player.device)
    if hasattr(player, "actions_high"):
        player.actions_high = player.actions_high[:action_dim].to(player.device)
    # In the SB3-style evaluation path we want normalized opponent actions and
    # let the wrapper perform final clipping/inversion.
    if hasattr(player, "clip_actions"):
        player.clip_actions = False
    # rl-games players require one explicit batch-size probe so that batched
    # observations are not treated as a single unbatched sample.
    dummy_obs = torch.zeros((max(1, int(num_envs)), int(obs_dim)), dtype=torch.float32, device=player.device)
    _ = player.get_batch_size(dummy_obs, 1)
    if getattr(player, "is_rnn", False) and hasattr(player, "init_rnn"):
        player.init_rnn()

    return RlGamesPredictPolicyAdapter(
        runner=runner,
        player=player,
        device=device,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )


def build_fixed_opponent_action_from_peg1_policy(
    *,
    obs: Any,
    peg1_policy: PPO,
    opponent_policy: PPO,
    deterministic_peg1: bool = False,
    deterministic_opponent: bool = True,
    invert_opponent_actions: bool = True,
) -> np.ndarray:
    """
    Build full 4D env action from a 2D Peg1 policy and a fixed opponent policy
    using the same remap/invert flow as training OpponentVecEnvWrapper.
    """
    obs_policy = ensure_2d_batch(obs_to_array(obs), dtype=np.float32)

    act_peg1, _ = peg1_policy.predict(obs_policy, deterministic=deterministic_peg1)
    act_peg1 = ensure_2d_batch(act_peg1, dtype=np.float32)
    if act_peg1.shape[1] < 2:
        raise ValueError(f"Peg1 policy action must have at least 2 dims, got {act_peg1.shape}")
    act_peg1 = act_peg1[:, :2]

    obs_opp = build_training_style_opponent_obs(obs_policy)
    act_opp, _ = opponent_policy.predict(
        obs_opp,
        deterministic=deterministic_opponent,
    )
    act_opp = ensure_2d_batch(act_opp, dtype=np.float32)
    if act_opp.shape[1] < 2:
        raise ValueError(f"Opponent policy action must have at least 2 dims, got {act_opp.shape}")
    act_opp = act_opp[:, :2]
    if invert_opponent_actions:
        act_opp = -act_opp
    act_opp = np.clip(act_opp, -1.0, 1.0)

    return np.concatenate([act_peg1, act_opp], axis=1).astype(np.float32)


def transform_observations(obs: np.ndarray, dt: float = 1 / 50, smooth_window: int = 15) -> np.ndarray:
    """
    Transform demo observations to simulation coordinates:
      - Map positions: swap axes (demo x->sim y, demo y->sim x), scale, center
      - (Optional) apply a small smoothing filter to positions (to reduce quantization artifacts)
      - Compute velocities via finite differences in sim coordinates
      - Return layout: [bp_x, bp_y, bv_x, bv_y,
                        lp_x, lp_y, rp_x, rp_y,
                        lv_x, lv_y, rv_x, rv_y]
    """
    o = np.asarray(obs, dtype=np.float32)
    if o.ndim == 1:
        o = o[None, :]
    if o.ndim != 2 or o.shape[1] != 12:
        return o

    # slices in incoming demo layout
    bp = o[:, 0:2]  # [bp_x, bp_y]
    lp = o[:, 4:6]  # [lp_x, lp_y]
    rp = o[:, 6:8]  # [rp_x, rp_y]

    # map positions: demo (x=long, y=short, top-left origin) -> sim (x=short, y=long, centered)
    def map_pos(pxy):
        x = pxy[:, 1] * SHORT_FACTOR - BOARD_SHORT / 2  # sim x from demo y
        y = pxy[:, 0] * LONG_FACTOR - BOARD_LONG / 2  # sim y from demo x
        return np.stack([x, y], axis=1)  # [x_sim, y_sim]

    bp_s = map_pos(bp)
    lp_s = map_pos(lp)
    rp_s = map_pos(rp)

    # Optional: smooth positions a little before differentiating.
    # This reduces axis-aligned spikes caused by quantized (pixel) positions.
    def smooth_pos(p: np.ndarray) -> np.ndarray:
        w = int(smooth_window)
        if w <= 1 or p.shape[0] < 3:
            return p
        if w == 3:
            kernel = np.asarray([0.25, 0.5, 0.25], dtype=np.float32)
        else:
            kernel = np.ones((w,), dtype=np.float32)
            kernel /= float(kernel.sum())
        pad = w // 2
        padded = np.pad(p, ((pad, pad), (0, 0)), mode="edge")
        out = np.empty_like(p)
        out[:, 0] = np.convolve(padded[:, 0], kernel, mode="valid")
        out[:, 1] = np.convolve(padded[:, 1], kernel, mode="valid")
        return out

    bp_s = smooth_pos(bp_s)
    lp_s = smooth_pos(lp_s)
    rp_s = smooth_pos(rp_s)

    # finite-difference velocities in sim coords
    def vel_from_pos(p):
        v = np.zeros_like(p)
        if p.shape[0] > 1:
            v[:-1] = (p[1:] - p[:-1]) / dt
            v[-1] = v[-2]
        return v  # [vx, vy]

    bv_s = vel_from_pos(bp_s)
    lv_s = vel_from_pos(lp_s)
    rv_s = vel_from_pos(rp_s)

    # Assemble in the same order as expected elsewhere:
    # [bp_x, bp_y, bv_x, bv_y, lp_x, lp_y, rp_x, rp_y, lv_x, lv_y, rv_x, rv_y]
    out = np.concatenate([bp_s, bv_s, lp_s, rp_s, lv_s, rv_s], axis=1)
    return out


def actions_from_transformed_obs(obs: np.ndarray) -> np.ndarray:
    """
    Extract actions (peg velocities) from transformed observations.
    Assumes transformed obs layout: [bp_x, bp_y, bv_x, bv_y,
                                     lp_x, lp_y, rp_x, rp_y,
                                     lv_x, lv_y, rv_x, rv_y]
    Returns actions of shape (T, 4): [lv_x, lv_y, rv_x, rv_y]
    using the first T rows (drop the terminal obs).
    """
    o = np.asarray(obs, dtype=np.float32)
    if o.ndim != 2 or o.shape[1] != 12:
        return np.zeros((0, 4), dtype=np.float32)
    acts = o[:-1, [8, 9, 10, 11]]  # take velocities, drop last step to get T
    return acts


def collect_pkl_paths(base_dir: str, pattern: str = "**/*.pkl") -> List[str]:
    from pathlib import Path

    base = Path(base_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Base directory {base_dir!r} not found")
    paths = sorted(str(p) for p in base.rglob("*.pkl"))
    if not paths:
        print(f"[warn] No .pkl files under {base_dir}")
    return paths


def strip_sim_native_action_history(
    trajectories: List[Trajectory],
    *,
    keep_obs_dim: int = 12,
) -> Tuple[List[Trajectory], int, List[int]]:
    """For sim-native data, keep only first keep_obs_dim observation channels."""
    out: List[Trajectory] = []
    stripped = 0
    seen_dims: set[int] = set()
    for tr in trajectories:
        if isinstance(tr.obs, dict):
            out.append(tr)
            continue
        obs = np.asarray(tr.obs, dtype=np.float32)
        if obs.ndim != 2:
            out.append(tr)
            continue
        seen_dims.add(int(obs.shape[1]))
        if obs.shape[1] > int(keep_obs_dim):
            obs = obs[:, : int(keep_obs_dim)]
            stripped += 1
        out.append(
            Trajectory(
                obs=obs,
                acts=np.asarray(tr.acts, dtype=np.float32),
                infos=tr.infos,
                terminal=tr.terminal,
            )
        )
    return out, stripped, sorted(seen_dims)


def zero_mask_observation_channels(
    trajectories: List[Trajectory],
    *,
    start: int = 8,
    end: int = 12,
) -> List[Trajectory]:
    """Set observation channels [start:end] to zero for all non-dict trajectories."""
    out: List[Trajectory] = []
    for tr in trajectories:
        if isinstance(tr.obs, dict):
            out.append(tr)
            continue
        obs = np.asarray(tr.obs, dtype=np.float32).copy()
        if obs.ndim == 2 and obs.shape[1] >= int(end):
            obs[:, int(start) : int(end)] = 0.0
        out.append(
            Trajectory(
                obs=obs,
                acts=np.asarray(tr.acts, dtype=np.float32),
                infos=tr.infos,
                terminal=tr.terminal,
            )
        )
    return out


def load_trajectories(
    paths: List[str],
    action_space: gym.Space,
    min_len: int = 2,
    *,
    source_format: str = DEMO_SOURCE_HUMAN_RAW,
    strip_sim_action_history: bool = True,
    masked_velocities: bool = False,
) -> List[Trajectory]:
    """
    Loads a list of pickles. Each file is expected to contain [TrajectoryWithRew(...)]
    as produced by your ROS converter (one trajectory per file).
    Returns list[Trajectory] with rewards dropped.
    """
    out: List[Trajectory] = []

    for p in paths:
        with open(p, "rb") as fh:
            payload = pickle.load(fh)

        # Normalise to iterable
        items: Iterable[Union[TrajectoryWithRew, Trajectory, dict]]
        if isinstance(payload, (list, tuple)):
            items = payload
        else:
            items = [payload]

        for it in items:
            if isinstance(it, TrajectoryWithRew):
                raw_obs = it.obs
            elif isinstance(it, Trajectory):
                raw_obs = it.obs
            elif isinstance(it, dict):
                raw_obs = it["obs"]
            else:
                raise TypeError(f"Unrecognized item in {p}: {type(it)}")

            obs = (
                maybe_wrap_in_dictobs(raw_obs)
                if isinstance(raw_obs, dict)
                else np.asarray(raw_obs)
            )
            if isinstance(obs, dict):
                # leave dict obs untouched
                acts = np.asarray(
                    it.acts
                    if isinstance(it, (Trajectory, TrajectoryWithRew))
                    else it["acts"]
                )
            else:
                if source_format == DEMO_SOURCE_HUMAN_RAW:
                    obs = transform_observations(obs)
                    acts = actions_from_transformed_obs(obs)
                elif source_format == DEMO_SOURCE_SIM_NATIVE:
                    raw_acts = (
                        it.acts if isinstance(it, (Trajectory, TrajectoryWithRew)) else it.get("acts")
                    )
                    if raw_acts is None:
                        raise ValueError(f"sim_native payload requires explicit acts in {p}")
                    acts = np.asarray(raw_acts)
                    if isinstance(action_space, gym.spaces.Box):
                        expected_dim = int(action_space.shape[-1])
                        if acts.ndim != 2 or acts.shape[1] != expected_dim:
                            raise ValueError(
                                f"sim_native action dimension mismatch in {p}: "
                                f"got {acts.shape}, expected (*,{expected_dim})."
                            )
                else:
                    raise ValueError(
                        f"Unsupported source_format={source_format!r}. "
                        f"Expected {DEMO_SOURCE_HUMAN_RAW!r} or {DEMO_SOURCE_SIM_NATIVE!r}."
                    )

                if obs.ndim != 2:
                    raise ValueError(f"Expected 2D observation array in {p}, got shape={obs.shape}.")
                if obs.shape[0] != acts.shape[0] + 1:
                    raise ValueError(
                        f"Malformed trajectory in {p}: obs_len={obs.shape[0]} acts_len={acts.shape[0]} "
                        "(expected obs_len == acts_len + 1)."
                    )

            tr = Trajectory(
                obs=obs,
                acts=acts,
                infos=it.infos
                if isinstance(it, (Trajectory, TrajectoryWithRew))
                else it.get("infos", None),
                terminal=getattr(it, "terminal", True)
                if isinstance(it, (Trajectory, TrajectoryWithRew))
                else bool(it.get("terminal", True)),
            )

            if len(tr.acts) < min_len:
                continue
            out.append(tr)

    if source_format == DEMO_SOURCE_SIM_NATIVE and bool(strip_sim_action_history):
        out, stripped_count, seen_dims = strip_sim_native_action_history(out, keep_obs_dim=12)
        print(
            "[dataset] strip_sim_action_history=True "
            f"seen_obs_dims={seen_dims} stripped_traj={stripped_count}"
        )
    if bool(masked_velocities):
        out = zero_mask_observation_channels(out, start=8, end=12)
        print("[dataset] zero-masked observation channels [8:12].")

    n_steps = sum(len(tr.acts) for tr in out)
    print(f"Loaded {len(out)} trajectories ({n_steps} steps) from {len(paths)} files")
    return out


def slice_peg1_actions(
    trajectories: List[Trajectory],
    *,
    action_start: int = 0,
    action_dim: int = 2,
) -> List[Trajectory]:
    """Convert demonstrations to peg1-only actions by slicing the leading action dims."""
    out: List[Trajectory] = []
    start = int(action_start)
    stop = start + int(action_dim)
    for tr in trajectories:
        acts = np.asarray(tr.acts, dtype=np.float32)
        if acts.ndim != 2:
            raise ValueError(f"Expected action array rank 2, got shape={acts.shape}.")
        if acts.shape[1] < stop:
            raise ValueError(
                f"Cannot slice peg1 actions [{start}:{stop}] from action shape={acts.shape}."
            )
        out.append(
            Trajectory(
                obs=tr.obs,
                acts=acts[:, start:stop].copy(),
                infos=tr.infos,
                terminal=tr.terminal,
            )
        )
    return out


def make_env(
    num_envs: int,
    seed: Optional[int] = None,
    *,
    use_collision_avoidance: bool = False,
    collision_max_vel: float = 0.2,
    actuator_apply_player: bool = True,
    actuator_apply_opponent: bool = True,
) -> Sb3VecEnvWrapper:
    cfg = EnvCfg()
    cfg.scene.num_envs = num_envs
    if seed is not None:
        cfg.seed = seed

    base_env = gym.make("Isaac-Klask-v0", cfg=cfg)
    actuator_kwargs: Dict[str, Any] = {}
    unsupported_side_controls: List[str] = []
    try:
        wrapper_sig = inspect.signature(ActuatorModelWrapper.__init__)
    except (TypeError, ValueError):
        wrapper_sig = None
    if wrapper_sig is not None:
        if "apply_to_player" in wrapper_sig.parameters:
            actuator_kwargs["apply_to_player"] = bool(actuator_apply_player)
        elif not bool(actuator_apply_player):
            unsupported_side_controls.append("player")
        if "apply_to_opponent" in wrapper_sig.parameters:
            actuator_kwargs["apply_to_opponent"] = bool(actuator_apply_opponent)
        elif not bool(actuator_apply_opponent):
            unsupported_side_controls.append("opponent")
    elif not (bool(actuator_apply_player) and bool(actuator_apply_opponent)):
        unsupported_side_controls.extend(["player", "opponent"])

    if unsupported_side_controls:
        print(
            "[warn] Installed ActuatorModelWrapper does not support selective side bypass "
            f"for: {', '.join(unsupported_side_controls)}. Falling back to legacy wrapper behavior."
        )

    env = ActuatorModelWrapper(base_env, **actuator_kwargs)
    if use_collision_avoidance:
        env = KlaskCollisionAvoidanceWrapper(env, max_vel=collision_max_vel)
    venv = Sb3VecEnvWrapper(env)
    setattr(venv, "_base_env", base_env)
    return venv


def close_env(venv: Optional[Sb3VecEnvWrapper]) -> None:
    if venv is None:
        return
    try:
        venv.close()
    except Exception as e:
        print(f"Warning: Error closing wrapped environment: {e}")
    base_env = getattr(venv, "_base_env", None)
    if base_env is not None:
        try:
            base_env.close()
        except Exception as e:
            print(f"Warning: Error closing base environment: {e}")
        setattr(venv, "_base_env", None)
    try:
        from isaaclab.sim import SimulationContext
        sim = SimulationContext.instance()
        if sim is not None:
            sim.close()
            SimulationContext._instance = None
    except Exception as e:
        print(f"Warning: Error closing simulation context: {e}")

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_checkpoint_run_config(checkpoint: Optional[str]) -> Dict[str, Any]:
    if not checkpoint:
        return {}
    run_dir = os.path.dirname(os.path.abspath(checkpoint))
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r") as fh:
            payload = json.load(fh)
    except Exception as exc:
        print(f"[warn] failed to read checkpoint config {config_path}: {exc}")
        return {}
    if not isinstance(payload, dict):
        return {}
    inner = payload.get("config")
    if isinstance(inner, dict):
        return inner
    return payload


def is_human_demo_bc_checkpoint(checkpoint: Optional[str]) -> bool:
    if not checkpoint:
        return False
    if os.path.basename(checkpoint) != "ppo_policy_bc_final.zip":
        return False
    cfg = _load_checkpoint_run_config(checkpoint)
    if str(cfg.get("demo_source_format", "")).strip().lower() == DEMO_SOURCE_HUMAN_RAW:
        return True
    run_name = os.path.basename(os.path.dirname(os.path.abspath(checkpoint))).upper()
    return "HUMAN_DEMO" in run_name


@dataclass
class RunningCorr:
    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        if x.size == 0:
            return
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch in RunningCorr: {x.shape} vs {y.shape}")
        self.n += x.size
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_x2 += float((x * x).sum())
        self.sum_y2 += float((y * y).sum())
        self.sum_xy += float((x * y).sum())

    def corr(self) -> float:
        if self.n < 2:
            return float("nan")
        num = self.n * self.sum_xy - self.sum_x * self.sum_y
        den_x = self.n * self.sum_x2 - self.sum_x * self.sum_x
        den_y = self.n * self.sum_y2 - self.sum_y * self.sum_y
        if den_x <= 0 or den_y <= 0:
            return float("nan")
        return num / np.sqrt(den_x * den_y)


def goal_indicator_from_obs(
    obs: np.ndarray,
    goal: Tuple[float, float, float],
    *,
    ball_pos_slice: slice,
    ball_vel_slice: slice,
) -> np.ndarray:
    """Return boolean array indicating ball in goal with speed threshold."""
    arr = obs_to_array(obs)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    cx, cy, r = goal
    ball_pos = arr[:, ball_pos_slice]
    ball_vel = arr[:, ball_vel_slice]
    in_circle = (ball_pos[:, 0] - cx) ** 2 + (ball_pos[:, 1] - cy) ** 2 <= r ** 2
    speed_sq = ball_vel[:, 0] ** 2 + ball_vel[:, 1] ** 2
    slow_enough = speed_sq <= MAX_BALL_VEL_GOAL ** 2
    return np.logical_and(in_circle, slow_enough)


def peg_in_goal_indicator_from_obs(
    obs: Any,
    goal: tuple[float, float, float],
    *,
    peg_pos_slice: slice,
) -> np.ndarray:
    arr = obs_to_array(obs)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    cx, cy, r = goal
    peg_pos = arr[:, peg_pos_slice]
    return (peg_pos[:, 0] - cx) ** 2 + (peg_pos[:, 1] - cy) ** 2 <= r ** 2


def select_positions(obs: np.ndarray, target: str, layout: Dict[str, slice]) -> np.ndarray:
    """Select positions from obs array using the provided layout."""
    arr = obs_to_array(obs)
    if arr.ndim == 1:
        arr = arr[None, :]
    if target == "ball":
        sl = layout["ball_pos"]
    elif target == "peg1":
        sl = layout["peg1_pos"]
    elif target == "peg2":
        sl = layout["peg2_pos"]
    else:
        raise ValueError(f"Unknown target {target!r}")
    return arr[:, sl]


def build_native_layout_from_policy_layout(policy_layout: Dict[str, slice]) -> Dict[str, slice]:
    """Build 12D-compatible layout keys from policy observation slices."""
    return {
        "ball_pos": policy_layout["ball_pos"],
        "ball_vel": policy_layout["ball_vel"],
        "peg1_pos": policy_layout["peg1_pos"],
        "peg2_pos": policy_layout["peg2_pos"],
        "peg1_vel": slice(policy_layout["peg1_vel_x"].start, policy_layout["peg1_vel_y"].stop),
        "peg2_vel": slice(policy_layout["peg2_vel_x"].start, policy_layout["peg2_vel_y"].stop),
    }


def layout_fits_obs_dim(layout: Dict[str, slice], obs_dim: int) -> bool:
    """Return True if all layout slices are inside observation width."""
    if obs_dim <= 0:
        return False
    for sl in layout.values():
        if not isinstance(sl, slice):
            return False
        if sl.start is None or sl.stop is None:
            return False
        if int(sl.start) < 0 or int(sl.stop) > int(obs_dim):
            return False
    return True


def resolve_transition_layout(
    *,
    source_format: str,
    obs_dim: int,
    policy_layout: Dict[str, slice],
) -> Dict[str, slice]:
    """
    Resolve transition layout for demo/analysis tensors.
    - human_raw uses transformed DEMO_LAYOUT
    - sim_native uses policy-layout indexing so peg labels match occupancy plots.
    """
    if source_format == DEMO_SOURCE_SIM_NATIVE:
        native_layout = build_native_layout_from_policy_layout(policy_layout)
        if layout_fits_obs_dim(native_layout, obs_dim):
            return native_layout
        if layout_fits_obs_dim(DEMO_LAYOUT, obs_dim):
            print(
                "[warn] sim_native layout did not fit obs dim; "
                "falling back to DEMO_LAYOUT for compatibility."
            )
            return DEMO_LAYOUT
        raise ValueError(
            "Could not resolve a valid sim_native transition layout for obs_dim="
            f"{obs_dim}. native_layout={native_layout}"
        )

    if not layout_fits_obs_dim(DEMO_LAYOUT, obs_dim):
        raise ValueError(
            "human_raw transition layout mismatch: expected DEMO_LAYOUT to fit obs_dim="
            f"{obs_dim}."
        )
    return DEMO_LAYOUT


def _to_bool_vector(x: Any, num_envs: int) -> np.ndarray:
    """Convert tensor/array-like values to a boolean vector of shape (num_envs,)."""
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.size == 0:
        return np.zeros(num_envs, dtype=bool)
    flat = arr.reshape(-1).astype(bool)
    if flat.size != num_envs:
        raise ValueError(
            f"Termination term size mismatch: expected {num_envs}, got {flat.size}."
        )
    return flat


def termination_flags_from_manager(
    venv: Sb3VecEnvWrapper,
    num_envs: int,
    term_names: List[str],
) -> Dict[str, np.ndarray]:
    """Read per-env boolean flags for all requested termination terms."""
    base_env = venv.unwrapped
    term_mgr = getattr(base_env, "termination_manager", None)
    if term_mgr is None:
        raise RuntimeError("Underlying env has no `termination_manager`; cannot read termination flags.")
    term_flags: Dict[str, np.ndarray] = {}
    for name in term_names:
        term_flags[name] = _to_bool_vector(term_mgr.get_term(name), num_envs)
    return term_flags


def _demo_terminal_flags(
    obs: np.ndarray,
    T: int,
    *,
    layout: Dict[str, slice],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return per-step terminal-like flags for demo transitions:
      - goal_scored
      - goal_conceded
      - terminal_like (goal or peg-in-goal)
    """
    next_obs = obs[1 : T + 1]

    goal_scored = goal_indicator_from_obs(
        next_obs,
        GOAL_OPPONENT,
        ball_pos_slice=layout["ball_pos"],
        ball_vel_slice=layout["ball_vel"],
    )
    goal_conceded = goal_indicator_from_obs(
        next_obs,
        GOAL_PLAYER,
        ball_pos_slice=layout["ball_pos"],
        ball_vel_slice=layout["ball_vel"],
    )
    peg1_in_goal = peg_in_goal_indicator_from_obs(
        next_obs,
        GOAL_PLAYER,
        peg_pos_slice=layout["peg1_pos"],
    )
    peg2_in_goal = peg_in_goal_indicator_from_obs(
        next_obs,
        GOAL_OPPONENT,
        peg_pos_slice=layout["peg2_pos"],
    )

    # Segment only on event onsets (rising edges), not on every frame where
    # an in-goal condition remains true for multiple steps.
    terminal_state = goal_scored | goal_conceded | peg1_in_goal | peg2_in_goal
    terminal_like = terminal_state.copy()
    if terminal_like.size > 1:
        terminal_like[1:] = terminal_state[1:] & ~terminal_state[:-1]
    return goal_scored, goal_conceded, terminal_like


def _split_demo_segments(terminal_like: np.ndarray) -> List[Tuple[int, int]]:
    """
    Split transition indices [0..T-1] into pseudo-episodes using terminal_like flags.
    Returns list of (start, end) with end exclusive, over transition indices.
    """
    T = int(terminal_like.shape[0])
    segments: List[Tuple[int, int]] = []
    start = 0
    for t in range(T):
        if terminal_like[t]:
            end = t + 1
            if end > start:
                segments.append((start, end))
            start = end
    if start < T:
        segments.append((start, T))
    return segments


def accumulate_heatmap(
    heat: np.ndarray,
    positions: np.ndarray,
    xrange: Tuple[float, float],
    yrange: Tuple[float, float],
) -> None:
    xmin, xmax = xrange
    ymin, ymax = yrange

    x = positions[:, 0]
    y = positions[:, 1]

    xbins = heat.shape[1]
    ybins = heat.shape[0]

    x_idx = ((x - xmin) / (xmax - xmin) * xbins).astype(np.int64)
    y_idx = ((y - ymin) / (ymax - ymin) * ybins).astype(np.int64)

    mask = (x_idx >= 0) & (x_idx < xbins) & (y_idx >= 0) & (y_idx < ybins)
    if not np.any(mask):
        return
    np.add.at(heat, (y_idx[mask], x_idx[mask]), 1)


def load_reward_net(
    path: str,
    *,
    observation_space: gym.Space,
    action_space: gym.Space,
    device: torch.device,
    reward_net_type: str = REWARD_NET_TYPE_BASIC_SHAPED,
    reward_hid_sizes: Optional[List[int]] = None,
    potential_hid_sizes: Optional[List[int]] = None,
) -> RewardNet:
    reward_hid_sizes = reward_hid_sizes or [32]
    potential_hid_sizes = potential_hid_sizes or [32, 32]
    kwargs = {
        "observation_space": observation_space,
        "action_space": action_space,
        "normalize_input_layer": RunningNorm,
        "reward_hid_sizes": tuple(int(x) for x in reward_hid_sizes),
        "potential_hid_sizes": tuple(int(x) for x in potential_hid_sizes),
    }
    if reward_net_type == REWARD_NET_TYPE_BASIC_SHAPED:
        reward_net = BasicShapedRewardNet(**kwargs).to(device)
    elif reward_net_type == REWARD_NET_TYPE_KLASK_RELATIVE_SHAPED:
        reward_net = KlaskRelativeShapedRewardNet(**kwargs).to(device)
    else:
        raise ValueError(f"Unsupported reward_net_type={reward_net_type!r}")
    state = torch.load(path, map_location=device)
    reward_net.load_state_dict(state)
    reward_net.eval()
    return reward_net


def evaluate_agent(
    venv: Sb3VecEnvWrapper,
    policy: PPO,
    opponent_policy: Optional[PPO],
    reward_net: Optional[BasicShapedRewardNet],
    n_episodes: int,
    bins: int,
    heatmap_targets: List[str],
    device: torch.device,
    policy_layout: Dict[str, slice],
    *,
    peg1_only_self_play: bool = False,
    peg1_only_fixed_opponent: bool = False,
    mirror_opponent_x: bool = False,
    mirror_opponent_y: bool = False,
    deterministic_peg1: bool = False,
    deterministic_self_play: bool = False,
    deterministic_opponent: bool = True,
    invert_opponent_actions: bool = True,
    plot_directly_velocity_requested: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Run evaluation and return:
      - summary metrics
      - per-episode metrics
      - agent heatmaps (one per target)
      - velocity samples (speed magnitudes) from the policy actions for peg1/peg2
      - action vs position samples for peg1/peg2 (policy)
    """

    heat_agent = {target: np.zeros((bins, bins), dtype=np.float64) for target in heatmap_targets}
    vel_samples: Dict[str, List[np.ndarray]] = {"peg1_angle_deg": [], "peg2_angle_deg": []}
    action_pos_samples: Dict[str, List[np.ndarray]] = {
        "peg1_y": [],
        "peg1_action_y": [],
        "peg2_y": [],
        "peg2_action_y": [],
    }

    obs = unwrap_obs(venv.reset())
    num_envs = venv.num_envs
    step_dt = float(getattr(venv.unwrapped, "step_dt", KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"]))
    term_mgr = getattr(venv.unwrapped, "termination_manager", None)
    if term_mgr is None:
        raise RuntimeError("Underlying env has no `termination_manager`; cannot read termination flags.")
    termination_term_names: List[str] = list(getattr(term_mgr, "active_terms", []))
    if len(termination_term_names) == 0:
        raise RuntimeError("Termination manager has no active terms.")
    term_csv_keys: Dict[str, str] = {name: f"term_{name}" for name in termination_term_names}

    ep_reward = np.zeros(num_envs, dtype=np.float64)
    ep_length = np.zeros(num_envs, dtype=np.int64)
    ep_goals = np.zeros(num_envs, dtype=np.int64)
    ep_conceded = np.zeros(num_envs, dtype=np.int64)
    ep_reward_net = np.zeros(num_envs, dtype=np.float64)

    episode_metrics: List[Dict[str, Any]] = []

    step_corr = RunningCorr()
    step_reward_sum_goal = 0.0
    step_reward_count_goal = 0
    step_reward_sum_non_goal = 0.0
    step_reward_count_non_goal = 0

    # Diagnostics: action -> motion correlation and boundary occupancy
    corr_a1_dy1 = RunningCorr()
    corr_a2_dy2 = RunningCorr()
    corr_a1_dy2 = RunningCorr()
    corr_a2_dy1 = RunningCorr()
    corr_a1_dy1_interior = RunningCorr()
    corr_a2_dy2_interior = RunningCorr()
    corr_a1_dy2_interior = RunningCorr()
    corr_a2_dy1_interior = RunningCorr()
    boundary_count_peg1 = 0
    boundary_count_peg2 = 0
    boundary_total = 0
    interior_count_peg1 = 0
    interior_count_peg2 = 0
    peg1_y_min = KLASK_PARAMS["edge"][2]
    peg1_y_max = KLASK_PARAMS["edge"][3]
    peg2_y_min = -KLASK_PARAMS["edge"][3]
    peg2_y_max = -KLASK_PARAMS["edge"][2]
    peg_x_min = KLASK_PARAMS["edge"][0]
    peg_x_max = KLASK_PARAMS["edge"][1]
    boundary_eps = 1e-3

    total_env_reward = 0.0
    total_env_reward_sq = 0.0
    total_steps = 0
    total_goal_steps = 0
    total_conceded_steps = 0

    episodes_completed = 0

    while episodes_completed < n_episodes:
        if peg1_only_fixed_opponent:
            if opponent_policy is None:
                raise ValueError("peg1_only_fixed_opponent=True requires a loaded opponent policy.")
            action = build_fixed_opponent_action_from_peg1_policy(
                obs=obs,
                peg1_policy=policy,
                opponent_policy=opponent_policy,
                deterministic_peg1=deterministic_peg1,
                deterministic_opponent=deterministic_opponent,
                invert_opponent_actions=invert_opponent_actions,
            )
        elif peg1_only_self_play:
            action = build_self_play_action_from_peg1_policy(
                obs=obs,
                policy=policy,
                policy_layout=policy_layout,
                mirror_x=mirror_opponent_x,
                mirror_y=mirror_opponent_y,
                deterministic=deterministic_self_play,
            )
        else:
            action, _ = policy.predict(obs, deterministic=False)
            action = ensure_2d_batch(action, dtype=np.float32)

        action_arr = ensure_2d_batch(action, dtype=np.float32)
        if action_arr.shape[1] < 4:
            raise ValueError(
                f"Evaluation expects 4D env actions, got {action_arr.shape}. "
                "Use --peg1-only-self-play for 2D Peg1-only policies."
            )

        # Policy action is the desired peg velocity: [dx1, dy1, dx2, dy2].
        # Store direction (angle in degrees) for distribution plots.
        # Note: for a ~zero vector, the direction is undefined; we mark those as NaN.
        if action_arr.ndim == 2 and action_arr.shape[1] >= 4:
            eps = 1e-6
            vx1 = action_arr[:, 0]
            vy1 = action_arr[:, 1]
            v1 = np.hypot(vx1, vy1)
            a1 = (np.degrees(np.arctan2(vy1, vx1)) + 360.0) % 360.0
            a1[v1 <= eps] = np.nan
            a1[(vx1 == 0.0) | (vy1 == 0.0)] = np.nan

            vx2 = action_arr[:, 2]
            vy2 = action_arr[:, 3]
            v2 = np.hypot(vx2, vy2)
            a2 = (np.degrees(np.arctan2(vy2, vx2)) + 360.0) % 360.0
            a2[v2 <= eps] = np.nan
            a2[(vx2 == 0.0) | (vy2 == 0.0)] = np.nan

            vel_samples["peg1_angle_deg"].append(a1)
            vel_samples["peg2_angle_deg"].append(a2)
        next_obs, reward, done, info = vec_step(venv, action_arr)
        next_obs = unwrap_obs(next_obs)

        # Reward net inference (optional)
        if reward_net is not None:
            with torch.no_grad():
                obs_t = to_torch_obs(obs, device)
                next_obs_t = to_torch_obs(next_obs, device)
                reward_action_dim = int(getattr(policy.action_space, "shape", (action_arr.shape[1],))[0])
                reward_action_arr = action_arr[:, :reward_action_dim]
                act_t = torch.as_tensor(reward_action_arr, device=device, dtype=torch.float32)
                done_t = torch.as_tensor(done.astype(np.float32), device=device)
                r_hat = reward_net(obs_t, act_t, next_obs_t, done_t)
                r_hat = r_hat.detach().cpu().numpy().reshape(-1)
        else:
            r_hat = np.zeros_like(reward, dtype=np.float32)

        # Goal events come directly from per-env termination terms on the underlying env.
        term_flags = termination_flags_from_manager(
            venv=venv,
            num_envs=num_envs,
            term_names=termination_term_names,
        )
        goal_scored_term = term_flags.get("goal_scored", np.zeros(num_envs, dtype=bool))
        goal_conceded_term = term_flags.get("goal_conceded", np.zeros(num_envs, dtype=bool))
        goal_scored_event = np.logical_and(goal_scored_term, done)
        goal_conceded_event = np.logical_and(goal_conceded_term, done)
        goal_scored_step = goal_scored_event
        goal_conceded_step = goal_conceded_event

        both_goal_flags = np.logical_and(goal_scored_event, goal_conceded_event)
        if np.any(both_goal_flags):
            idx = np.where(both_goal_flags)[0].tolist()
            print(f"[warn] Both goal_scored and goal_conceded were True on done env indices: {idx}")

        # Heatmap accumulation (agent occupancy) for each target
        for target in heatmap_targets:
            positions = select_positions(obs, target, policy_layout)
            accumulate_heatmap(heat_agent[target], positions, FIELD_X, FIELD_Y)

        # Diagnostics: action->motion correlations and boundary occupancy
        obs_arr = obs_to_array(obs)
        next_arr = obs_to_array(next_obs)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[None, :]
        if next_arr.ndim == 1:
            next_arr = next_arr[None, :]

        peg1_x = obs_arr[:, policy_layout["peg1_pos"].start + 0]
        peg1_y = obs_arr[:, policy_layout["peg1_pos"].start + 1]
        peg2_x = obs_arr[:, policy_layout["peg2_pos"].start + 0]
        peg2_y = obs_arr[:, policy_layout["peg2_pos"].start + 1]
        peg1_y_next = next_arr[:, policy_layout["peg1_pos"].start + 1]
        peg2_y_next = next_arr[:, policy_layout["peg2_pos"].start + 1]

        valid = ~done
        if np.any(valid):
            dy1 = peg1_y_next[valid] - peg1_y[valid]
            dy2 = peg2_y_next[valid] - peg2_y[valid]
            corr_a1_dy1.update(action_arr[valid, 1], dy1)
            corr_a2_dy2.update(action_arr[valid, 3], dy2)
            corr_a1_dy2.update(action_arr[valid, 1], dy2)
            corr_a2_dy1.update(action_arr[valid, 3], dy1)

            boundary_total += int(np.sum(valid))
            boundary_count_peg1 += int(np.sum(peg1_y[valid] <= peg1_y_min + boundary_eps))
            boundary_count_peg2 += int(np.sum(peg2_y[valid] <= peg2_y_min + boundary_eps))

            # Interior-only diagnostics: exclude points near x/y bounds.
            interior_peg1 = (
                valid
                & (peg1_x > peg_x_min + boundary_eps)
                & (peg1_x < peg_x_max - boundary_eps)
                & (peg1_y > peg1_y_min + boundary_eps)
                & (peg1_y < peg1_y_max - boundary_eps)
            )
            interior_peg2 = (
                valid
                & (peg2_x > peg_x_min + boundary_eps)
                & (peg2_x < peg_x_max - boundary_eps)
                & (peg2_y > peg2_y_min + boundary_eps)
                & (peg2_y < peg2_y_max - boundary_eps)
            )
            interior_count_peg1 += int(np.sum(interior_peg1))
            interior_count_peg2 += int(np.sum(interior_peg2))
            interior_common = interior_peg1 & interior_peg2

            if np.any(interior_peg1):
                dy1_int = peg1_y_next[interior_peg1] - peg1_y[interior_peg1]
                corr_a1_dy1_interior.update(action_arr[interior_peg1, 1], dy1_int)

            if np.any(interior_peg2):
                dy2_int = peg2_y_next[interior_peg2] - peg2_y[interior_peg2]
                corr_a2_dy2_interior.update(action_arr[interior_peg2, 3], dy2_int)

            if np.any(interior_common):
                dy1_common = peg1_y_next[interior_common] - peg1_y[interior_common]
                dy2_common = peg2_y_next[interior_common] - peg2_y[interior_common]
                corr_a1_dy2_interior.update(action_arr[interior_common, 1], dy2_common)
                corr_a2_dy1_interior.update(action_arr[interior_common, 3], dy1_common)

            # Action-vs-position samples (policy).
            # Optionally store observed env velocity instead of requested action.
            action_pos_samples["peg1_y"].append(peg1_y[valid])
            action_pos_samples["peg2_y"].append(peg2_y[valid])
            if plot_directly_velocity_requested:
                action_pos_samples["peg1_action_y"].append(action_arr[valid, 1])
                action_pos_samples["peg2_action_y"].append(action_arr[valid, 3])
            else:
                action_pos_samples["peg1_action_y"].append(
                    ((peg1_y_next[valid] - peg1_y[valid]) / step_dt).astype(np.float32)
                )
                action_pos_samples["peg2_action_y"].append(
                    ((peg2_y_next[valid] - peg2_y[valid]) / step_dt).astype(np.float32)
                )

        # Step-level aggregation
        total_env_reward += float(np.sum(reward))
        total_env_reward_sq += float(np.sum(reward ** 2))
        total_steps += reward.size
        total_goal_steps += int(np.sum(goal_scored_step))
        total_conceded_steps += int(np.sum(goal_conceded_step))

        if reward_net is not None:
            step_corr.update(r_hat, goal_scored_step.astype(np.float32))

            # Reward-net mean for goal vs non-goal steps
            if np.any(goal_scored_step):
                step_reward_sum_goal += float(r_hat[goal_scored_step].sum())
                step_reward_count_goal += int(goal_scored_step.sum())
            if np.any(~goal_scored_step):
                step_reward_sum_non_goal += float(r_hat[~goal_scored_step].sum())
                step_reward_count_non_goal += int((~goal_scored_step).sum())

        # Per-episode aggregation (vectorized)
        ep_reward += reward
        ep_length += 1
        ep_goals += goal_scored_event.astype(np.int64)
        ep_conceded += goal_conceded_event.astype(np.int64)
        if reward_net is not None:
            ep_reward_net += r_hat

        if np.any(done):
            done_idx = np.where(done)[0]
            for i in done_idx:
                row = {
                    "episode": episodes_completed,
                    "reward": float(ep_reward[i]),
                    "length": int(ep_length[i]),
                    "goals_scored": int(ep_goals[i]),
                    "goals_conceded": int(ep_conceded[i]),
                    "reward_net_sum": float(ep_reward_net[i]),
                }
                for term_name in termination_term_names:
                    row[term_csv_keys[term_name]] = int(term_flags[term_name][i])
                episode_metrics.append(
                    row
                )
                episodes_completed += 1
                if episodes_completed >= n_episodes:
                    break

            ep_reward[done_idx] = 0.0
            ep_length[done_idx] = 0
            ep_goals[done_idx] = 0
            ep_conceded[done_idx] = 0
            ep_reward_net[done_idx] = 0.0

        obs = next_obs

    # Summary metrics
    # Drop very short episodes from summary statistics, as these usually
    # correspond to initialization/reset artifacts instead of policy behavior.
    summary_min_episode_length = 3
    summary_episode_metrics = [
        m for m in episode_metrics if int(m["length"]) > summary_min_episode_length
    ]
    num_episodes_total = int(len(episode_metrics))
    num_episodes_used = int(len(summary_episode_metrics))
    num_episodes_discarded = int(num_episodes_total - num_episodes_used)

    rewards = np.array([m["reward"] for m in summary_episode_metrics], dtype=np.float64)
    lengths = np.array([m["length"] for m in summary_episode_metrics], dtype=np.float64)
    goals_scored = np.array([m["goals_scored"] for m in summary_episode_metrics], dtype=np.float64)
    goals_conceded = np.array([m["goals_conceded"] for m in summary_episode_metrics], dtype=np.float64)
    reward_net_sums = np.array([m["reward_net_sum"] for m in summary_episode_metrics], dtype=np.float64)
    term_episode_rates = {
        f"term_rate_{name}": float(
            np.mean([m[term_csv_keys[name]] for m in summary_episode_metrics], dtype=np.float64)
        )
        if summary_episode_metrics
        else float("nan")
        for name in termination_term_names
    }

    env_reward_mean = float(rewards.mean()) if rewards.size else float("nan")
    env_reward_std = float(rewards.std()) if rewards.size else float("nan")
    mean_length = float(lengths.mean()) if lengths.size else float("nan")

    goals_per_episode = float(goals_scored.mean()) if goals_scored.size else float("nan")
    conceded_per_episode = float(goals_conceded.mean()) if goals_conceded.size else float("nan")
    goal_episode_rate = float((goals_scored > 0).mean()) if goals_scored.size else float("nan")

    # Episode-level correlation: reward_net_sum vs goals_scored
    ep_corr = RunningCorr()
    if reward_net_sums.size:
        ep_corr.update(reward_net_sums, goals_scored)

    summary = {
        "episodes": num_episodes_used,
        "episodes_total": num_episodes_total,
        "episodes_used_for_summary": num_episodes_used,
        "episodes_discarded_short": num_episodes_discarded,
        "summary_discard_length_lte": int(summary_min_episode_length),
        "termination_terms": list(termination_term_names),
        "mean_env_reward": env_reward_mean,
        "std_env_reward": env_reward_std,
        "mean_episode_length": mean_length,
        "goals_scored_per_episode": goals_per_episode,
        "goals_conceded_per_episode": conceded_per_episode,
        "goal_episode_rate": goal_episode_rate,
        "goal_steps_rate": float(total_goal_steps / total_steps) if total_steps > 0 else float("nan"),
        "conceded_steps_rate": float(total_conceded_steps / total_steps) if total_steps > 0 else float("nan"),
        "reward_net_corr_step_vs_goal": step_corr.corr() if reward_net is not None else float("nan"),
        "reward_net_corr_episode_sum_vs_goals": ep_corr.corr() if reward_net is not None else float("nan"),
        "reward_net_mean_on_goal_steps": float(step_reward_sum_goal / step_reward_count_goal)
        if step_reward_count_goal > 0 and reward_net is not None
        else float("nan"),
        "reward_net_mean_on_non_goal_steps": float(step_reward_sum_non_goal / step_reward_count_non_goal)
        if step_reward_count_non_goal > 0 and reward_net is not None
        else float("nan"),
        "diag_peg1_y_min": peg1_y_min,
        "diag_peg1_y_max": peg1_y_max,
        "diag_peg2_y_min": peg2_y_min,
        "diag_peg2_y_max": peg2_y_max,
        "diag_peg1_y_min_frac": float(boundary_count_peg1 / boundary_total)
        if boundary_total > 0
        else float("nan"),
        "diag_peg2_y_min_frac": float(boundary_count_peg2 / boundary_total)
        if boundary_total > 0
        else float("nan"),
        "diag_peg1_interior_frac": float(interior_count_peg1 / boundary_total)
        if boundary_total > 0
        else float("nan"),
        "diag_peg2_interior_frac": float(interior_count_peg2 / boundary_total)
        if boundary_total > 0
        else float("nan"),
        "diag_corr_action1_dy1": corr_a1_dy1.corr(),
        "diag_corr_action2_dy2": corr_a2_dy2.corr(),
        "diag_corr_action1_dy2": corr_a1_dy2.corr(),
        "diag_corr_action2_dy1": corr_a2_dy1.corr(),
        "diag_corr_action1_dy1_interior": corr_a1_dy1_interior.corr(),
        "diag_corr_action2_dy2_interior": corr_a2_dy2_interior.corr(),
        "diag_corr_action1_dy2_interior": corr_a1_dy2_interior.corr(),
        "diag_corr_action2_dy1_interior": corr_a2_dy1_interior.corr(),
    }
    summary.update(term_episode_rates)

    policy_vel = {
        "peg1_angle_deg": np.concatenate(vel_samples["peg1_angle_deg"]).astype(np.float32)
        if vel_samples["peg1_angle_deg"]
        else np.zeros((0,), dtype=np.float32),
        "peg2_angle_deg": np.concatenate(vel_samples["peg2_angle_deg"]).astype(np.float32)
        if vel_samples["peg2_angle_deg"]
        else np.zeros((0,), dtype=np.float32),
    }

    policy_action_pos = {
        "peg1_y": np.concatenate(action_pos_samples["peg1_y"]).astype(np.float32)
        if action_pos_samples["peg1_y"]
        else np.zeros((0,), dtype=np.float32),
        "peg1_action_y": np.concatenate(action_pos_samples["peg1_action_y"]).astype(np.float32)
        if action_pos_samples["peg1_action_y"]
        else np.zeros((0,), dtype=np.float32),
        "peg2_y": np.concatenate(action_pos_samples["peg2_y"]).astype(np.float32)
        if action_pos_samples["peg2_y"]
        else np.zeros((0,), dtype=np.float32),
        "peg2_action_y": np.concatenate(action_pos_samples["peg2_action_y"]).astype(np.float32)
        if action_pos_samples["peg2_action_y"]
        else np.zeros((0,), dtype=np.float32),
    }

    return summary, episode_metrics, heat_agent, policy_vel, policy_action_pos


def collect_demo_velocity_orientations(trajs: List[Trajectory]) -> Dict[str, np.ndarray]:
    """Collect demo peg velocity direction angles (0..360 deg) from Trajectory.acts ([dx1, dy1, dx2, dy2])."""
    peg1: List[np.ndarray] = []
    peg2: List[np.ndarray] = []
    for tr in trajs:
        acts = np.asarray(tr.acts)
        if acts.ndim != 2 or acts.shape[1] < 4:
            continue
        eps = 1e-6

        vx1 = acts[:, 0]
        vy1 = acts[:, 1]
        v1 = np.hypot(vx1, vy1)
        a1 = (np.degrees(np.arctan2(vy1, vx1)) + 360.0) % 360.0
        a1[v1 <= eps] = np.nan
        a1[(vx1 == 0.0) | (vy1 == 0.0)] = np.nan

        vx2 = acts[:, 2]
        vy2 = acts[:, 3]
        v2 = np.hypot(vx2, vy2)
        a2 = (np.degrees(np.arctan2(vy2, vx2)) + 360.0) % 360.0
        a2[v2 <= eps] = np.nan
        a2[(vx2 == 0.0) | (vy2 == 0.0)] = np.nan

        peg1.append(a1)
        peg2.append(a2)

    return {
        "peg1_angle_deg": np.concatenate(peg1).astype(np.float32) if peg1 else np.zeros((0,), dtype=np.float32),
        "peg2_angle_deg": np.concatenate(peg2).astype(np.float32) if peg2 else np.zeros((0,), dtype=np.float32),
    }


def collect_demo_action_vs_pos(
    trajs: List[Trajectory],
    *,
    layout: Dict[str, slice],
) -> Dict[str, np.ndarray]:
    """Collect demo peg1/peg2 y-position and y-action aligned to obs[:-1] (acts length)."""
    peg1_y: List[np.ndarray] = []
    peg1_action_y: List[np.ndarray] = []
    peg2_y: List[np.ndarray] = []
    peg2_action_y: List[np.ndarray] = []
    for tr in trajs:
        obs = np.asarray(tr.obs)
        acts = np.asarray(tr.acts)
        if obs.ndim != 2 or acts.ndim != 2 or acts.shape[1] < 4:
            continue
        T = min(len(acts), len(obs) - 1)
        if T <= 0:
            continue
        peg1_y.append(obs[:T, layout["peg1_pos"].start + 1])
        peg1_action_y.append(acts[:T, 1])
        peg2_y.append(obs[:T, layout["peg2_pos"].start + 1])
        peg2_action_y.append(acts[:T, 3])

    return {
        "peg1_y": np.concatenate(peg1_y).astype(np.float32) if peg1_y else np.zeros((0,), dtype=np.float32),
        "peg1_action_y": np.concatenate(peg1_action_y).astype(np.float32)
        if peg1_action_y
        else np.zeros((0,), dtype=np.float32),
        "peg2_y": np.concatenate(peg2_y).astype(np.float32) if peg2_y else np.zeros((0,), dtype=np.float32),
        "peg2_action_y": np.concatenate(peg2_action_y).astype(np.float32)
        if peg2_action_y
        else np.zeros((0,), dtype=np.float32),
    }


def plot_velocity_histograms(
    *,
    demo_vel: Dict[str, np.ndarray],
    policy_vel: Dict[str, np.ndarray],
    out_path: str,
    bins: int = 60,
    policy_label: str = "Policy",
) -> None:
    """
    Create a 2x2 figure with 4 histograms:
      - Demo peg1 velocity direction angle (deg)
      - Policy peg1 velocity direction angle (deg)
      - Demo peg2 velocity direction angle (deg)
      - Policy peg2 velocity direction angle (deg)
    """
    demo_peg1 = np.asarray(demo_vel.get("peg1_angle_deg", np.zeros((0,), dtype=np.float32))).ravel()
    demo_peg2 = np.asarray(demo_vel.get("peg2_angle_deg", np.zeros((0,), dtype=np.float32))).ravel()
    pol_peg1 = np.asarray(policy_vel.get("peg1_angle_deg", np.zeros((0,), dtype=np.float32))).ravel()
    pol_peg2 = np.asarray(policy_vel.get("peg2_angle_deg", np.zeros((0,), dtype=np.float32))).ravel()

    edges = np.linspace(0.0, 360.0, int(bins) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True, sharey=True)

    def _hist(ax, data: np.ndarray, title: str) -> None:
        finite = data[np.isfinite(data)]
        ax.hist(finite, bins=edges, density=True, color="tab:blue", alpha=0.85)
        ax.set_title(f"{title} (n={int(finite.size)})")
        ax.grid(True, alpha=0.2)

    _hist(axes[0, 0], demo_peg1, "Demo peg1 angle")
    _hist(axes[0, 1], pol_peg1, f"{policy_label} peg1 angle")
    _hist(axes[1, 0], demo_peg2, "Demo peg2 angle")
    _hist(axes[1, 1], pol_peg2, f"{policy_label} peg2 angle")

    for ax in axes[1, :]:
        ax.set_xlabel("angle (deg, 0..360)")
    for ax in axes[:, 0]:
        ax.set_ylabel("density")
    for ax in axes.ravel():
        ax.set_xlim(0.0, 360.0)
        ax.set_xticks([0, 90, 180, 270, 360])

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_action_bias_vs_pos(
    *,
    demo_data: Dict[str, np.ndarray],
    policy_data: Dict[str, np.ndarray],
    out_path: str,
    bins: int = 40,
    y_value_label: str = "action_y",
) -> None:
    """Plot mean y-value vs peg_y for peg1/peg2 for demos and policy in a 2x2 figure."""
    demo_y = np.asarray(demo_data.get("peg1_y", np.zeros((0,), dtype=np.float32))).ravel()
    demo_a = np.asarray(demo_data.get("peg1_action_y", np.zeros((0,), dtype=np.float32))).ravel()
    pol_y = np.asarray(policy_data.get("peg1_y", np.zeros((0,), dtype=np.float32))).ravel()
    pol_a = np.asarray(policy_data.get("peg1_action_y", np.zeros((0,), dtype=np.float32))).ravel()
    demo_y2 = np.asarray(demo_data.get("peg2_y", np.zeros((0,), dtype=np.float32))).ravel()
    demo_a2 = np.asarray(demo_data.get("peg2_action_y", np.zeros((0,), dtype=np.float32))).ravel()
    pol_y2 = np.asarray(policy_data.get("peg2_y", np.zeros((0,), dtype=np.float32))).ravel()
    pol_a2 = np.asarray(policy_data.get("peg2_action_y", np.zeros((0,), dtype=np.float32))).ravel()

    def binned_mean(
        x: np.ndarray,
        y: np.ndarray,
        x_min: float,
        x_max: float,
        n_bins: int,
        *,
        min_count: int = 50,
        trim_edge_bins: int = 1,
    ):
        edges = np.linspace(x_min, x_max, n_bins + 1)
        idx = np.digitize(x, edges) - 1
        centers = (edges[:-1] + edges[1:]) / 2.0
        mean = np.full((n_bins,), np.nan, dtype=np.float32)
        counts = np.zeros((n_bins,), dtype=np.int64)
        for b in range(n_bins):
            mask = idx == b
            counts[b] = int(np.sum(mask))
            if counts[b] > 0:
                mean[b] = float(np.mean(y[mask]))
        # Hide noisy bins: low support and the outermost edge bins.
        mean[counts < int(min_count)] = np.nan
        if trim_edge_bins > 0 and n_bins > 2 * trim_edge_bins:
            mean[:trim_edge_bins] = np.nan
            mean[-trim_edge_bins:] = np.nan
        return centers, mean

    peg1_y_min = KLASK_PARAMS["edge"][2]
    peg1_y_max = KLASK_PARAMS["edge"][3]
    peg2_y_min = -KLASK_PARAMS["edge"][3]
    peg2_y_max = -KLASK_PARAMS["edge"][2]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharey=True)

    c, m = binned_mean(demo_y, demo_a, peg1_y_min, peg1_y_max, bins)
    axes[0, 0].plot(c, m, color="tab:red")
    axes[0, 0].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[0, 0].set_title(f"Demo: mean {y_value_label} vs peg1_y")
    axes[0, 0].set_xlabel("peg1_y")
    axes[0, 0].set_ylabel(f"mean {y_value_label}")

    c, m = binned_mean(pol_y, pol_a, peg1_y_min, peg1_y_max, bins)
    axes[0, 1].plot(c, m, color="tab:blue")
    axes[0, 1].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[0, 1].set_title(f"Policy: mean {y_value_label} vs peg1_y")
    axes[0, 1].set_xlabel("peg1_y")

    c, m = binned_mean(demo_y2, demo_a2, peg2_y_min, peg2_y_max, bins)
    axes[1, 0].plot(c, m, color="tab:red")
    axes[1, 0].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[1, 0].set_title(f"Demo: mean {y_value_label} vs peg2_y")
    axes[1, 0].set_xlabel("peg2_y")
    axes[1, 0].set_ylabel(f"mean {y_value_label}")

    c, m = binned_mean(pol_y2, pol_a2, peg2_y_min, peg2_y_max, bins)
    axes[1, 1].plot(c, m, color="tab:blue")
    axes[1, 1].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[1, 1].set_title(f"Policy: mean {y_value_label} vs peg2_y")
    axes[1, 1].set_xlabel("peg2_y")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.2)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compute_demo_heatmap_from_trajs(
    trajs: List[Trajectory],
    bins: int,
    heatmap_targets: List[str],
    *,
    layout: Dict[str, slice],
) -> Dict[str, np.ndarray]:
    heat_demo = {target: np.zeros((bins, bins), dtype=np.float64) for target in heatmap_targets}

    for tr in trajs:
        if isinstance(tr.obs, dict):
            # If dict obs, skip: not compatible with simple heatmap here
            continue
        obs = np.asarray(tr.obs, dtype=np.float32)
        acts = np.asarray(tr.acts)
        if obs.ndim != 2 or acts.ndim != 2 or len(acts) == 0:
            continue
        # Keep demo occupancy transition-aligned to avoid overweighting terminal rows.
        T = min(len(acts), len(obs) - 1)
        if T <= 0:
            continue
        obs_t = obs[:T]
        for target in heatmap_targets:
            positions = select_positions(obs_t, target, layout)
            accumulate_heatmap(heat_demo[target], positions, FIELD_X, FIELD_Y)

    return heat_demo


def evaluate_reward_net_on_demos(
    trajs: List[Trajectory],
    reward_net: BasicShapedRewardNet,
    device: torch.device,
    *,
    layout: Dict[str, slice],
    batch_size: int = 8192,
) -> Dict[str, Any]:
    """
    Evaluate reward-net alignment with goal scoring on demo trajectories.
    Goal is detected from ball position (goal circle) in next_obs.
    """
    step_corr = RunningCorr()
    step_reward_sum_goal = 0.0
    step_reward_count_goal = 0
    step_reward_sum_non_goal = 0.0
    step_reward_count_non_goal = 0

    total_steps = 0
    total_goal_steps = 0
    total_trajs = 0
    goal_traj_count = 0
    total_demo_episodes = 0
    goal_episode_count = 0

    for tr in trajs:
        if isinstance(tr.obs, dict):
            continue
        obs = np.asarray(tr.obs, dtype=np.float32)
        acts = np.asarray(tr.acts, dtype=np.float32)
        if obs.ndim != 2 or acts.ndim != 2 or len(acts) == 0:
            continue

        T = min(len(acts), len(obs) - 1)
        if T <= 0:
            continue
        obs_t = obs[:T]
        next_obs = obs[1 : T + 1]
        acts = acts[:T]

        goal_scored, _, terminal_like = _demo_terminal_flags(obs, T, layout=layout)
        segments = _split_demo_segments(terminal_like)
        if not segments:
            continue

        done = np.zeros((T,), dtype=np.float32)
        for _, end in segments:
            done[end - 1] = 1.0

        total_steps += int(T)
        total_goal_steps += int(goal_scored.sum())
        total_trajs += 1
        if goal_scored.any():
            goal_traj_count += 1
        total_demo_episodes += len(segments)
        goal_episode_count += int(sum(np.any(goal_scored[s:e]) for (s, e) in segments))

        # Reward net inference in batches
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            obs_b = obs_t[start:end]
            next_obs_b = next_obs[start:end]
            acts_b = acts[start:end]
            done_b = done[start:end]
            goal_b = goal_scored[start:end]

            with torch.no_grad():
                obs_tb = torch.as_tensor(obs_b, device=device, dtype=torch.float32)
                next_obs_tb = torch.as_tensor(next_obs_b, device=device, dtype=torch.float32)
                acts_tb = torch.as_tensor(acts_b, device=device, dtype=torch.float32)
                done_tb = torch.as_tensor(done_b, device=device)
                r_hat = reward_net(obs_tb, acts_tb, next_obs_tb, done_tb)
                r_hat = r_hat.detach().cpu().numpy().reshape(-1)

            step_corr.update(r_hat, goal_b.astype(np.float32))

            if np.any(goal_b):
                step_reward_sum_goal += float(r_hat[goal_b].sum())
                step_reward_count_goal += int(goal_b.sum())
            if np.any(~goal_b):
                step_reward_sum_non_goal += float(r_hat[~goal_b].sum())
                step_reward_count_non_goal += int((~goal_b).sum())

    return {
        "demo_trajs_used": int(total_trajs),
        "demo_steps": int(total_steps),
        "demo_goal_steps": int(total_goal_steps),
        "demo_goal_steps_rate": float(total_goal_steps / total_steps) if total_steps > 0 else float("nan"),
        "demo_goal_traj_rate": float(goal_traj_count / total_trajs) if total_trajs > 0 else float("nan"),
        "demo_alignment_episodes": int(total_demo_episodes),
        "demo_alignment_goal_episode_rate": float(goal_episode_count / total_demo_episodes)
        if total_demo_episodes > 0
        else float("nan"),
        "demo_reward_net_corr_step_vs_goal": step_corr.corr(),
        "demo_reward_net_mean_on_goal_steps": float(step_reward_sum_goal / step_reward_count_goal)
        if step_reward_count_goal > 0
        else float("nan"),
        "demo_reward_net_mean_on_non_goal_steps": float(step_reward_sum_non_goal / step_reward_count_non_goal)
        if step_reward_count_non_goal > 0
        else float("nan"),
    }


def evaluate_demo_episode_metrics(
    trajs: List[Trajectory],
    *,
    layout: Dict[str, slice],
) -> Dict[str, Any]:
    """
    Evaluate goal/concede metrics on demo trajectories by detecting goal events from
    next observations (same detector used for policy rollouts).
    """
    episode_lengths: List[int] = []
    goals_scored_per_episode: List[int] = []
    goals_conceded_per_episode: List[int] = []

    total_steps = 0
    total_goal_steps = 0
    total_conceded_steps = 0

    for tr in trajs:
        if isinstance(tr.obs, dict):
            continue

        obs = np.asarray(tr.obs, dtype=np.float32)
        acts = np.asarray(tr.acts, dtype=np.float32)
        if obs.ndim != 2 or acts.ndim != 2 or len(acts) == 0:
            continue

        T = min(len(acts), len(obs) - 1)
        if T <= 0:
            continue

        goal_scored, goal_conceded, terminal_like = _demo_terminal_flags(obs, T, layout=layout)
        segments = _split_demo_segments(terminal_like)
        if not segments:
            continue

        for start, end in segments:
            episode_lengths.append(int(end - start))
            # One environment episode can terminate by one terminal event at most.
            goals_scored_per_episode.append(int(np.any(goal_scored[start:end])))
            goals_conceded_per_episode.append(int(np.any(goal_conceded[start:end])))

        total_steps += int(T)
        total_goal_steps += int(goal_scored.sum())
        total_conceded_steps += int(goal_conceded.sum())

    if not episode_lengths:
        return {
            "demo_episodes": 0,
            "demo_mean_episode_length": float("nan"),
            "demo_goals_scored_per_episode": float("nan"),
            "demo_goals_conceded_per_episode": float("nan"),
            "demo_goal_episode_rate": float("nan"),
            "demo_conceded_episode_rate": float("nan"),
            "demo_goal_steps_rate": float("nan"),
            "demo_conceded_steps_rate": float("nan"),
            "demo_goal_diff_per_episode": float("nan"),
        }

    lens = np.asarray(episode_lengths, dtype=np.float64)
    scored = np.asarray(goals_scored_per_episode, dtype=np.float64)
    conceded = np.asarray(goals_conceded_per_episode, dtype=np.float64)

    return {
        "demo_episodes": int(len(lens)),
        "demo_mean_episode_length": float(lens.mean()),
        "demo_goals_scored_per_episode": float(scored.mean()),
        "demo_goals_conceded_per_episode": float(conceded.mean()),
        "demo_goal_episode_rate": float((scored > 0).mean()),
        "demo_conceded_episode_rate": float((conceded > 0).mean()),
        "demo_goal_steps_rate": float(total_goal_steps / total_steps) if total_steps > 0 else float("nan"),
        "demo_conceded_steps_rate": float(total_conceded_steps / total_steps) if total_steps > 0 else float("nan"),
        "demo_goal_diff_per_episode": float((scored - conceded).mean()),
    }


def build_demo_segmentation_debug_rows(
    trajs: List[Trajectory],
    *,
    layout: Dict[str, slice],
) -> List[Dict[str, Any]]:
    """
    Build per-transition debug rows for demo episode segmentation.
    Each row corresponds to transition t: obs[t] -> obs[t+1].
    """
    rows: List[Dict[str, Any]] = []

    for traj_idx, tr in enumerate(trajs):
        if isinstance(tr.obs, dict):
            continue

        obs = np.asarray(tr.obs, dtype=np.float32)
        acts = np.asarray(tr.acts, dtype=np.float32)
        if obs.ndim != 2 or acts.ndim != 2 or len(acts) == 0:
            continue

        T = min(len(acts), len(obs) - 1)
        if T <= 0:
            continue

        next_obs = obs[1 : T + 1]
        goal_scored = goal_indicator_from_obs(
            next_obs,
            GOAL_OPPONENT,
            ball_pos_slice=layout["ball_pos"],
            ball_vel_slice=layout["ball_vel"],
        )
        goal_conceded = goal_indicator_from_obs(
            next_obs,
            GOAL_PLAYER,
            ball_pos_slice=layout["ball_pos"],
            ball_vel_slice=layout["ball_vel"],
        )
        peg1_in_goal = peg_in_goal_indicator_from_obs(
            next_obs,
            GOAL_PLAYER,
            peg_pos_slice=layout["peg1_pos"],
        )
        peg2_in_goal = peg_in_goal_indicator_from_obs(
            next_obs,
            GOAL_OPPONENT,
            peg_pos_slice=layout["peg2_pos"],
        )
        terminal_state = goal_scored | goal_conceded | peg1_in_goal | peg2_in_goal
        done_detected = terminal_state.copy()
        if done_detected.size > 1:
            done_detected[1:] = terminal_state[1:] & ~terminal_state[:-1]

        episode_idx = 0
        for t in range(T):
            rows.append(
                {
                    "traj_idx": int(traj_idx),
                    "step_idx": int(t),
                    "episode_idx": int(episode_idx),
                    "done_detected": int(done_detected[t]),
                    "goal_scored": int(goal_scored[t]),
                    "goal_conceded": int(goal_conceded[t]),
                    "peg1_in_goal": int(peg1_in_goal[t]),
                    "peg2_in_goal": int(peg2_in_goal[t]),
                    "terminal_state": int(terminal_state[t]),
                    "ball_x": float(next_obs[t, layout["ball_pos"].start + 0]),
                    "ball_y": float(next_obs[t, layout["ball_pos"].start + 1]),
                    "peg1_x": float(next_obs[t, layout["peg1_pos"].start + 0]),
                    "peg1_y": float(next_obs[t, layout["peg1_pos"].start + 1]),
                    "peg2_x": float(next_obs[t, layout["peg2_pos"].start + 0]),
                    "peg2_y": float(next_obs[t, layout["peg2_pos"].start + 1]),
                }
            )
            if done_detected[t]:
                episode_idx += 1

    return rows


def save_demo_segmentation_debug_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    import csv

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def trajectories_to_transition_batch(trajs: List[Trajectory]) -> Dict[str, np.ndarray]:
    """Flatten trajectories into transition-aligned obs/act/next_obs/done arrays."""
    obs_parts: List[np.ndarray] = []
    next_obs_parts: List[np.ndarray] = []
    act_parts: List[np.ndarray] = []
    done_parts: List[np.ndarray] = []

    for tr in trajs:
        if isinstance(tr.obs, dict):
            continue
        obs = np.asarray(tr.obs, dtype=np.float32)
        acts = np.asarray(tr.acts, dtype=np.float32)
        if obs.ndim != 2 or acts.ndim != 2:
            continue
        t = min(len(acts), len(obs) - 1)
        if t <= 0:
            continue
        done = np.zeros((t,), dtype=np.float32)
        if bool(getattr(tr, "terminal", True)):
            done[t - 1] = 1.0
        obs_parts.append(obs[:t])
        next_obs_parts.append(obs[1 : t + 1])
        act_parts.append(acts[:t])
        done_parts.append(done)

    if not obs_parts:
        return {
            "obs": np.zeros((0, 12), dtype=np.float32),
            "acts": np.zeros((0, 4), dtype=np.float32),
            "next_obs": np.zeros((0, 12), dtype=np.float32),
            "done": np.zeros((0,), dtype=np.float32),
        }
    return {
        "obs": np.concatenate(obs_parts, axis=0),
        "acts": np.concatenate(act_parts, axis=0),
        "next_obs": np.concatenate(next_obs_parts, axis=0),
        "done": np.concatenate(done_parts, axis=0),
    }


def subsample_transition_batch(
    batch: Dict[str, np.ndarray],
    *,
    max_samples: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Deterministically subsample a transition batch."""
    n = int(batch["obs"].shape[0])
    if max_samples <= 0 or n <= max_samples:
        return batch
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_samples, replace=False))
    return {
        "obs": batch["obs"][idx],
        "acts": batch["acts"][idx],
        "next_obs": batch["next_obs"][idx],
        "done": batch["done"][idx],
    }


def align_transition_batch_action_dim(
    batch: Dict[str, np.ndarray],
    *,
    action_dim: int,
    label: str,
) -> Dict[str, np.ndarray]:
    """Ensure transition batch actions match the reward-net action width."""
    acts = ensure_2d_batch(batch["acts"], dtype=np.float32)
    target_dim = int(action_dim)
    if target_dim <= 0:
        raise ValueError(f"{label}: invalid target action_dim={target_dim}.")
    if acts.shape[1] < target_dim:
        raise ValueError(
            f"{label}: action batch has width {acts.shape[1]}, cannot align to reward-net "
            f"action_dim={target_dim}."
        )
    if acts.shape[1] == target_dim:
        return batch
    aligned = dict(batch)
    aligned["acts"] = acts[:, :target_dim].copy()
    return aligned


def reward_logits_batched(
    *,
    reward_net: BasicShapedRewardNet,
    obs: np.ndarray,
    acts: np.ndarray,
    next_obs: np.ndarray,
    done: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    """Batch reward-net inference over transitions."""
    n = int(obs.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    out: List[np.ndarray] = []
    for start in range(0, n, max(1, int(batch_size))):
        end = min(start + max(1, int(batch_size)), n)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs[start:end], device=device, dtype=torch.float32)
            act_t = torch.as_tensor(acts[start:end], device=device, dtype=torch.float32)
            next_obs_t = torch.as_tensor(next_obs[start:end], device=device, dtype=torch.float32)
            done_t = torch.as_tensor(done[start:end], device=device, dtype=torch.float32)
            logits = reward_net(obs_t, act_t, next_obs_t, done_t).detach().cpu().numpy().reshape(-1)
        out.append(logits.astype(np.float32))
    return np.concatenate(out, axis=0)


def compute_balanced_binary_metrics(
    expert_logits: np.ndarray,
    generated_logits: np.ndarray,
) -> Dict[str, float]:
    """Compute balanced discriminator metrics using expert(+1) and generated(0) classes."""
    expert_logits = np.asarray(expert_logits, dtype=np.float32).reshape(-1)
    generated_logits = np.asarray(generated_logits, dtype=np.float32).reshape(-1)
    n_expert = int(expert_logits.size)
    n_generated = int(generated_logits.size)

    pred_expert_expert = expert_logits > 0.0
    pred_expert_generated = generated_logits > 0.0
    tp = int(np.sum(pred_expert_expert))
    fn = int(n_expert - tp)
    fp = int(np.sum(pred_expert_generated))
    tn = int(n_generated - fp)
    total = int(n_expert + n_generated)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "num_expert": float(n_expert),
        "num_generated": float(n_generated),
        "accuracy_total": float((tp + tn) / total) if total > 0 else float("nan"),
        "accuracy_expert": float(tp / n_expert) if n_expert > 0 else float("nan"),
        "accuracy_generated": float(tn / n_generated) if n_generated > 0 else float("nan"),
        "precision_expert": float(precision),
        "recall_expert": float(recall),
        "f1_expert": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "expert_logits_mean": float(np.mean(expert_logits)) if n_expert > 0 else float("nan"),
        "generated_logits_mean": float(np.mean(generated_logits)) if n_generated > 0 else float("nan"),
    }


def _draw_field_overlay(ax) -> None:
    rect = Rectangle(
        (FIELD_X[0], FIELD_Y[0]),
        FIELD_X[1] - FIELD_X[0],
        FIELD_Y[1] - FIELD_Y[0],
        fill=False,
        linewidth=1.5,
        color="white",
    )
    ax.add_patch(rect)
    for goal in [GOAL_PLAYER, GOAL_OPPONENT]:
        cx, cy, r = goal
        circ = Circle((cx, cy), r, fill=False, linewidth=1.2, color="white")
        ax.add_patch(circ)


def _build_reward_grid_and_valid_mask(
    *,
    obs_ctx: np.ndarray,
    target: str,
    target_slice: slice,
    bins: int,
    legal_region_only: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
    x_grid = np.linspace(FIELD_X[0], FIELD_X[1], int(bins), dtype=np.float32)
    y_grid = np.linspace(FIELD_Y[0], FIELD_Y[1], int(bins), dtype=np.float32)
    valid_mask = np.ones((int(bins), int(bins)), dtype=bool)
    legal_bounds = None

    if bool(legal_region_only) and target in ("peg1", "peg2"):
        peg_x_min = float(KLASK_PARAMS["edge"][0])
        peg_x_max = float(KLASK_PARAMS["edge"][1])
        side_a = (float(KLASK_PARAMS["edge"][2]), float(KLASK_PARAMS["edge"][3]))
        side_b = (-float(KLASK_PARAMS["edge"][3]), -float(KLASK_PARAMS["edge"][2]))
        target_y = obs_ctx[:, target_slice.start + 1]
        y_mean = float(np.mean(target_y)) if target_y.size > 0 else 0.0
        center_a = 0.5 * (side_a[0] + side_a[1])
        center_b = 0.5 * (side_b[0] + side_b[1])
        y_lo, y_hi = side_a if abs(y_mean - center_a) <= abs(y_mean - center_b) else side_b
        y_min = float(min(y_lo, y_hi))
        y_max = float(max(y_lo, y_hi))
        legal_bounds = {
            "x": [float(peg_x_min), float(peg_x_max)],
            "y": [float(y_min), float(y_max)],
            "target_context_mean_y": float(y_mean),
            "selected_side": "side_a" if abs(y_mean - center_a) <= abs(y_mean - center_b) else "side_b",
        }
        xx, yy = np.meshgrid(x_grid, y_grid)
        valid_mask = (
            (xx >= peg_x_min)
            & (xx <= peg_x_max)
            & (yy >= y_min)
            & (yy <= y_max)
        )

    return x_grid, y_grid, valid_mask, legal_bounds


def _target_slice_from_layout(*, target: str, layout: Dict[str, slice]) -> slice:
    if target == "ball":
        return layout["ball_pos"]
    if target == "peg1":
        return layout["peg1_pos"]
    if target == "peg2":
        return layout["peg2_pos"]
    raise ValueError(f"Unsupported reward-map target: {target!r}")


def compute_marginalized_reward_map(
    *,
    reward_net: BasicShapedRewardNet,
    context_batch: Dict[str, np.ndarray],
    target: str,
    layout: Dict[str, slice],
    bins: int,
    legal_region_only: bool,
    device: torch.device,
    batch_size: int = 8192,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Average discriminator rewards over context transitions while sweeping one target position."""
    target_slice = _target_slice_from_layout(target=target, layout=layout)

    obs_ctx = np.asarray(context_batch["obs"], dtype=np.float32)
    acts_ctx = np.asarray(context_batch["acts"], dtype=np.float32)
    next_obs_ctx = np.asarray(context_batch["next_obs"], dtype=np.float32)
    done_ctx = np.asarray(context_batch["done"], dtype=np.float32)
    if obs_ctx.shape[0] == 0:
        raise ValueError("No validation context transitions available for reward-map computation.")

    x_grid, y_grid, valid_mask, legal_bounds = _build_reward_grid_and_valid_mask(
        obs_ctx=obs_ctx,
        target=target,
        target_slice=target_slice,
        bins=int(bins),
        legal_region_only=bool(legal_region_only),
    )
    reward_map = np.full((int(bins), int(bins)), np.nan, dtype=np.float32)

    base_obs_target = obs_ctx[:, target_slice]
    base_next_target = next_obs_ctx[:, target_slice]
    for y_idx, y in enumerate(y_grid):
        for x_idx, x in enumerate(x_grid):
            if not bool(valid_mask[y_idx, x_idx]):
                continue
            target_xy = np.asarray([x, y], dtype=np.float32)
            delta = target_xy[None, :] - base_obs_target
            obs_mod = np.array(obs_ctx, copy=True)
            next_mod = np.array(next_obs_ctx, copy=True)
            obs_mod[:, target_slice] = target_xy
            next_mod[:, target_slice] = base_next_target + delta
            logits = reward_logits_batched(
                reward_net=reward_net,
                obs=obs_mod,
                acts=acts_ctx,
                next_obs=next_mod,
                done=done_ctx,
                device=device,
                batch_size=batch_size,
            )
            reward_map[y_idx, x_idx] = float(np.mean(logits))

    meta = {
        "target": str(target),
        "target_slice": [int(target_slice.start), int(target_slice.stop)],
        "bins": int(bins),
        "context_samples": int(obs_ctx.shape[0]),
        "legal_region_only": bool(legal_region_only),
        "legal_bounds": legal_bounds,
        "evaluated_cells": int(np.sum(valid_mask)),
        "skipped_cells": int(valid_mask.size - np.sum(valid_mask)),
        "valid_cell_fraction": float(np.mean(valid_mask.astype(np.float32))),
        "x_grid": [float(v) for v in x_grid.tolist()],
        "y_grid": [float(v) for v in y_grid.tolist()],
        "field_x": [float(FIELD_X[0]), float(FIELD_X[1])],
        "field_y": [float(FIELD_Y[0]), float(FIELD_Y[1])],
    }
    return reward_map, meta


def compute_visited_state_reward_map(
    *,
    transition_batch: Dict[str, np.ndarray],
    logits: np.ndarray,
    target: str,
    layout: Dict[str, slice],
    bins: int,
    legal_region_only: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Mean reward-net logit per visited board cell for a chosen target position."""
    obs = np.asarray(transition_batch["obs"], dtype=np.float32)
    if obs.shape[0] == 0:
        raise ValueError("No transitions available for visited-state reward-map computation.")
    logits_arr = np.asarray(logits, dtype=np.float32).reshape(-1)
    if logits_arr.shape[0] != obs.shape[0]:
        raise ValueError(
            f"Visited-state reward-map requires logits length={obs.shape[0]}, got {logits_arr.shape[0]}."
        )

    target_slice = _target_slice_from_layout(target=target, layout=layout)
    x_grid, y_grid, valid_mask, legal_bounds = _build_reward_grid_and_valid_mask(
        obs_ctx=obs,
        target=target,
        target_slice=target_slice,
        bins=int(bins),
        legal_region_only=bool(legal_region_only),
    )

    positions = select_positions(obs, target, layout)
    xbins = int(bins)
    ybins = int(bins)
    x_idx = ((positions[:, 0] - FIELD_X[0]) / (FIELD_X[1] - FIELD_X[0]) * xbins).astype(np.int64)
    y_idx = ((positions[:, 1] - FIELD_Y[0]) / (FIELD_Y[1] - FIELD_Y[0]) * ybins).astype(np.int64)
    in_bounds = (x_idx >= 0) & (x_idx < xbins) & (y_idx >= 0) & (y_idx < ybins)
    if np.any(in_bounds):
        in_legal = valid_mask[y_idx[in_bounds], x_idx[in_bounds]]
        keep = np.zeros_like(in_bounds, dtype=bool)
        keep[np.where(in_bounds)[0]] = in_legal
    else:
        keep = in_bounds

    sums = np.zeros((ybins, xbins), dtype=np.float64)
    counts = np.zeros((ybins, xbins), dtype=np.int64)
    if np.any(keep):
        np.add.at(sums, (y_idx[keep], x_idx[keep]), logits_arr[keep])
        np.add.at(counts, (y_idx[keep], x_idx[keep]), 1)

    visited_reward_map = np.full((ybins, xbins), np.nan, dtype=np.float32)
    nonzero = counts > 0
    visited_reward_map[nonzero] = (sums[nonzero] / counts[nonzero]).astype(np.float32)
    visited_reward_map[~valid_mask] = np.nan

    meta = {
        "target": str(target),
        "target_slice": [int(target_slice.start), int(target_slice.stop)],
        "bins": int(bins),
        "samples_total": int(obs.shape[0]),
        "samples_used": int(np.sum(keep)),
        "samples_out_of_bounds_or_illegal": int(obs.shape[0] - np.sum(keep)),
        "legal_region_only": bool(legal_region_only),
        "legal_bounds": legal_bounds,
        "visited_cells": int(np.sum(nonzero)),
        "visited_cell_fraction": float(np.mean(nonzero.astype(np.float32))),
        "counts_min_nonzero": int(np.min(counts[nonzero])) if np.any(nonzero) else 0,
        "counts_max": int(np.max(counts)) if counts.size > 0 else 0,
        "x_grid": [float(v) for v in x_grid.tolist()],
        "y_grid": [float(v) for v in y_grid.tolist()],
        "field_x": [float(FIELD_X[0]), float(FIELD_X[1])],
        "field_y": [float(FIELD_Y[0]), float(FIELD_Y[1])],
    }
    return visited_reward_map, counts, meta


def plot_reward_map(
    reward_map: np.ndarray,
    *,
    out_path: str,
    target: str,
    legal_bounds: Optional[Dict[str, Any]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)
    finite_mask = np.isfinite(reward_map)
    if np.any(finite_mask):
        vabs = float(np.nanmax(np.abs(reward_map[finite_mask])))
    else:
        vabs = 1.0
    if not np.isfinite(vabs) or vabs <= 0.0:
        vabs = 1.0
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#d0d0d0")
    im = ax.imshow(
        np.ma.array(reward_map, mask=~finite_mask),
        origin="lower",
        extent=[FIELD_X[0], FIELD_X[1], FIELD_Y[0], FIELD_Y[1]],
        cmap=cmap,
        vmin=-vabs,
        vmax=vabs,
        aspect="equal",
    )
    if legal_bounds is not None:
        xb = legal_bounds.get("x", None)
        yb = legal_bounds.get("y", None)
        if isinstance(xb, (list, tuple)) and len(xb) == 2 and isinstance(yb, (list, tuple)) and len(yb) == 2:
            rect = Rectangle(
                (float(xb[0]), float(yb[0])),
                float(xb[1] - xb[0]),
                float(yb[1] - yb[0]),
                fill=False,
                linewidth=1.2,
                linestyle="--",
                edgecolor="black",
                alpha=0.8,
            )
            ax.add_patch(rect)
    _draw_field_overlay(ax)
    ax.set_title(f"Marginalized discriminator reward map ({target})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("mean reward-net logit")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_discriminator_logits_hist(
    *,
    train_expert_logits: np.ndarray,
    val_expert_logits: np.ndarray,
    train_generated_logits: np.ndarray,
    val_generated_logits: np.ndarray,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    bins = 80
    ax.hist(train_expert_logits, bins=bins, alpha=0.4, label="train expert", density=True)
    ax.hist(val_expert_logits, bins=bins, alpha=0.4, label="val expert", density=True)
    ax.hist(train_generated_logits, bins=bins, alpha=0.4, label="train generated", density=True)
    ax.hist(val_generated_logits, bins=bins, alpha=0.4, label="val generated", density=True)
    ax.axvline(0.0, color="black", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.set_xlabel("reward-net logit")
    ax.set_ylabel("density")
    ax.set_title("Discriminator logit distributions")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def collect_generated_transitions(
    *,
    venv: Sb3VecEnvWrapper,
    policy: PPO,
    opponent_policy: Optional[PPO],
    n_samples: int,
    policy_layout: Dict[str, slice],
    peg1_only_self_play: bool = False,
    peg1_only_fixed_opponent: bool = False,
    mirror_opponent_x: bool = False,
    mirror_opponent_y: bool = False,
    deterministic_peg1: bool = False,
    deterministic_self_play: bool = False,
    deterministic_opponent: bool = True,
    invert_opponent_actions: bool = True,
) -> Dict[str, np.ndarray]:
    """Collect generated transitions from policy rollouts for discriminator diagnostics."""
    obs = unwrap_obs(venv.reset())
    num_envs = int(venv.num_envs)
    target = int(max(0, n_samples))

    obs_parts: List[np.ndarray] = []
    acts_parts: List[np.ndarray] = []
    next_obs_parts: List[np.ndarray] = []
    done_parts: List[np.ndarray] = []
    collected = 0

    while collected < target:
        if peg1_only_fixed_opponent:
            if opponent_policy is None:
                raise ValueError("Fixed-opponent mode requires opponent policy.")
            action = build_fixed_opponent_action_from_peg1_policy(
                obs=obs,
                peg1_policy=policy,
                opponent_policy=opponent_policy,
                deterministic_peg1=deterministic_peg1,
                deterministic_opponent=deterministic_opponent,
                invert_opponent_actions=invert_opponent_actions,
            )
        elif peg1_only_self_play:
            action = build_self_play_action_from_peg1_policy(
                obs=obs,
                policy=policy,
                policy_layout=policy_layout,
                mirror_x=mirror_opponent_x,
                mirror_y=mirror_opponent_y,
                deterministic=deterministic_self_play,
            )
        else:
            action, _ = policy.predict(obs, deterministic=False)
            action = ensure_2d_batch(action, dtype=np.float32)

        action_arr = ensure_2d_batch(action, dtype=np.float32)
        if action_arr.shape[1] < 4:
            raise ValueError(
                f"Generated-transition sampling expects 4D env actions, got {action_arr.shape}."
            )
        next_obs, _, done, _ = vec_step(venv, action_arr)
        next_obs = unwrap_obs(next_obs)

        obs_arr = ensure_2d_batch(obs_to_array(obs), dtype=np.float32)
        next_arr = ensure_2d_batch(obs_to_array(next_obs), dtype=np.float32)
        take = min(num_envs, target - collected)
        obs_parts.append(obs_arr[:take].copy())
        acts_parts.append(action_arr[:take].copy())
        next_obs_parts.append(next_arr[:take].copy())
        done_parts.append(done[:take].astype(np.float32, copy=True))
        collected += int(take)
        obs = next_obs

    return {
        "obs": np.concatenate(obs_parts, axis=0) if obs_parts else np.zeros((0, 12), dtype=np.float32),
        "acts": np.concatenate(acts_parts, axis=0) if acts_parts else np.zeros((0, 4), dtype=np.float32),
        "next_obs": np.concatenate(next_obs_parts, axis=0)
        if next_obs_parts
        else np.zeros((0, 12), dtype=np.float32),
        "done": np.concatenate(done_parts, axis=0) if done_parts else np.zeros((0,), dtype=np.float32),
    }


def resolve_demo_loader_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve demo source and preprocessing flags from saved run config."""
    cfg = config.get("config", {}) if isinstance(config, dict) else {}
    source_format = str(cfg.get("demo_source_format", DEMO_SOURCE_HUMAN_RAW))
    strip_sim_action_history = bool(cfg.get("strip_sim_action_history", True))
    masked_velocities = bool(cfg.get("masked_velocities", False))
    if source_format == DEMO_SOURCE_HUMAN_RAW:
        masked_velocities = masked_velocities or bool(cfg.get("zero_mask_velocity_obs_human", False))
    elif source_format == DEMO_SOURCE_SIM_NATIVE:
        masked_velocities = masked_velocities or bool(cfg.get("zero_mask_velocity_obs_sim", False))
    return {
        "source_format": source_format,
        "strip_sim_action_history": strip_sim_action_history,
        "masked_velocities": masked_velocities,
    }


def resolve_dataset_split_paths(
    *,
    demo_base: str,
    split_paths: List[str],
) -> List[str]:
    """Resolve paths stored in dataset_split.json to existing .pkl files."""
    base = Path(demo_base).expanduser().resolve()
    resolved: List[str] = []
    for raw in split_paths:
        p = Path(str(raw)).expanduser()
        candidates: List[Path] = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(base / p)
            candidates.append(base / p.name)
            candidates.append((Path.cwd() / p).resolve())
        chosen = None
        for cand in candidates:
            c = cand.resolve()
            if c.exists():
                chosen = c
                break
        if chosen is None:
            raise FileNotFoundError(
                f"Could not resolve split path '{raw}' under demo_base={base}."
            )
        resolved.append(str(chosen))
    return sorted(resolved)


def plot_heatmaps(
    heat_agent: Dict[str, np.ndarray],
    heat_demo: Optional[Dict[str, np.ndarray]],
    out_path: str,
    heatmap_targets: List[str],
) -> None:
    """
    Plot heatmaps for targets. If peg1 and peg2 are both specified, overlay them
    on the same field using RGB channels (red=peg1, blue=peg2, purple=overlap).
    """
    has_demo = heat_demo is not None
    ncols = 2 if has_demo else 1

    # Check if we should combine peg1 and peg2
    combine_pegs = "peg1" in heatmap_targets and "peg2" in heatmap_targets

    def draw_field(ax):
        # Board rectangle
        rect = Rectangle(
            (FIELD_X[0], FIELD_Y[0]),
            FIELD_X[1] - FIELD_X[0],
            FIELD_Y[1] - FIELD_Y[0],
            fill=False,
            linewidth=1.5,
            color="white",
        )
        ax.add_patch(rect)

        # Goals
        for goal in [GOAL_PLAYER, GOAL_OPPONENT]:
            cx, cy, r = goal
            circ = Circle((cx, cy), r, fill=False, linewidth=1.2, color="white")
            ax.add_patch(circ)

    def make_combined_rgb(heat_peg1: np.ndarray, heat_peg2: np.ndarray) -> np.ndarray:
        """Combine peg1 (red) and peg2 (blue) into an RGB image."""
        h1 = np.log1p(heat_peg1)
        h2 = np.log1p(heat_peg2)
        # Normalize each to [0, 1]
        max_val = max(h1.max(), h2.max(), 1e-9)
        h1_norm = h1 / max_val
        h2_norm = h2 / max_val
        # RGB: peg1=red, peg2=blue, overlap=purple
        rgb = np.zeros((*h1.shape, 3), dtype=np.float32)
        rgb[..., 0] = h1_norm  # Red channel for peg1
        rgb[..., 2] = h2_norm  # Blue channel for peg2
        # Add some green where both are present for better visibility
        rgb[..., 1] = np.minimum(h1_norm, h2_norm) * 0.3
        return rgb

    if combine_pegs:
        # Single row with combined peg1+peg2 visualization
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 8), constrained_layout=True)
        if ncols == 1:
            axes = [axes]

        # Agent combined heatmap
        rgb_agent = make_combined_rgb(heat_agent["peg1"], heat_agent["peg2"])
        ax = axes[0]
        ax.imshow(
            rgb_agent,
            origin="lower",
            extent=[FIELD_X[0], FIELD_X[1], FIELD_Y[0], FIELD_Y[1]],
            aspect="equal",
        )
        draw_field(ax)
        ax.set_title("Agent occupancy (red=peg1, blue=peg2)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Demo combined heatmap
        if has_demo:
            rgb_demo = make_combined_rgb(heat_demo["peg1"], heat_demo["peg2"])
            ax = axes[1]
            ax.imshow(
                rgb_demo,
                origin="lower",
                extent=[FIELD_X[0], FIELD_X[1], FIELD_Y[0], FIELD_Y[1]],
                aspect="equal",
            )
            draw_field(ax)
            ax.set_title("Demo occupancy (red=peg1, blue=peg2)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        # Add a simple legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='peg1'),
            Patch(facecolor='blue', label='peg2'),
        ]
        fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.0, 0.5))

    else:
        # Fallback: separate rows for each target (single target case)
        n_targets = len(heatmap_targets)
        fig, axes = plt.subplots(
            n_targets, ncols, figsize=(6 * ncols, 5 * n_targets), constrained_layout=True, squeeze=False
        )

        # Compute global vmax across all heatmaps for consistent colormap
        all_disp = []
        for target in heatmap_targets:
            all_disp.append(np.log1p(heat_agent[target]))
            if has_demo:
                all_disp.append(np.log1p(heat_demo[target]))
        vmax = max(np.max(d) for d in all_disp)
        if vmax == 0:
            vmax = 1.0

        im = None
        for row_idx, target in enumerate(heatmap_targets):
            # Agent heatmap
            ax = axes[row_idx, 0]
            heat_agent_disp = np.log1p(heat_agent[target])
            im = ax.imshow(
                heat_agent_disp,
                origin="lower",
                extent=[FIELD_X[0], FIELD_X[1], FIELD_Y[0], FIELD_Y[1]],
                cmap="viridis",
                vmax=vmax,
                aspect="equal",
            )
            draw_field(ax)
            ax.set_title(f"Agent occupancy ({target}, log1p(count))")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            # Demo heatmap (if available)
            if has_demo:
                ax = axes[row_idx, 1]
                heat_demo_disp = np.log1p(heat_demo[target])
                ax.imshow(
                    heat_demo_disp,
                    origin="lower",
                    extent=[FIELD_X[0], FIELD_X[1], FIELD_Y[0], FIELD_Y[1]],
                    cmap="viridis",
                    vmax=vmax,
                    aspect="equal",
                )
                draw_field(ax)
                ax.set_title(f"Demo occupancy ({target}, log1p(count))")
                ax.set_xlabel("x")
                ax.set_ylabel("y")

        cbar = fig.colorbar(im, ax=axes, shrink=0.8)
        cbar.set_label("log1p(count)")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_episode_metrics(path: str, episode_metrics: List[Dict[str, Any]]) -> None:
    import csv

    if not episode_metrics:
        return
    fieldnames = list(episode_metrics[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in episode_metrics:
            writer.writerow(row)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GAIL policy + reward-net alignment + heatmaps")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(_default_trained_models_dir() / "klask_gail_20260227_131330_GAIL1"),
        help="Path to the trained model directory",
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Number of evaluation episodes")
    parser.add_argument("--num-envs", type=int, default=300, help="Number of parallel envs")
    parser.add_argument("--bins", type=int, default=80, help="Heatmap bins per axis")
    parser.add_argument(
        "--velocity-hist-bins",
        type=int,
        default=360,
        help="Bins for velocity direction histogram plot (0..360 deg)",
    )
    parser.add_argument("--action-bias-bins", type=int, default=40, help="Bins for action_y vs peg1_y plot")
    parser.add_argument("--no-action-bias-plot", action="store_true", help="Skip action_y vs peg1_y plot")
    parser.add_argument(
        "--plot-directly-velocity-requested",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, action-bias plot uses requested policy action_y. "
            "If false, uses observed env y-velocity (delta_y / step_dt)."
        ),
    )
    parser.add_argument("--no-velocity-hist", action="store_true", help="Skip velocity histogram plot")
    parser.add_argument("--policy-path", type=str, default=None, help="Override policy checkpoint path")
    parser.add_argument("--reward-net-path", type=str, default=None, help="Override reward net path")
    parser.add_argument("--no-reward-net", action="store_true", help="Skip reward-net alignment")
    parser.add_argument(
        "--heatmap-targets",
        type=str,
        nargs="+",
        choices=["peg1", "ball", "peg2"],
        default=["peg1", "peg2"],
        help="What to visualize in heatmaps (can specify multiple)",
    )
    parser.add_argument("--no-demos", action="store_true", help="Skip demo heatmap")
    parser.add_argument("--demo-base", type=str, default=None, help="Override demo base path")
    parser.add_argument("--device", type=str, default="cuda:1", help="Override device (e.g., cuda:0 or cpu)")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument(
        "--peg1-only-self-play",
        action="store_true",
        help=(
            "Evaluate a 2D Peg1-only policy by reusing the same policy for Peg2 "
            "to construct full 4D actions."
        ),
    )
    parser.add_argument(
        "--peg1-only-fixed-opponent",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Evaluate a 2D Peg1-only policy against a fixed opponent policy using "
            "the same remap/invert action construction as training. "
            "If omitted, the evaluator falls back to the training run config."
        ),
    )
    parser.add_argument(
        "--opponent-policy-path",
        type=str,
        default=None,
        help=(
            "Path to fixed opponent policy "
            "(defaults to run config, then BC cloned policy if fixed-opponent mode is enabled)."
        ),
    )
    parser.add_argument(
        "--opponent-policy-format",
        type=str,
        default=None,
        help=(
            "Explicit fixed-opponent policy format. "
            f"Valid values: {POLICY_FORMAT_SB3_PPO}, {POLICY_FORMAT_RL_GAMES}. "
            "If omitted, inferred from checkpoint extension or run config."
        ),
    )
    parser.add_argument(
        "--opponent-rl-games-config-path",
        type=str,
        default=None,
        help=(
            "RL-Games agent.yaml for the fixed opponent. "
            f"Required when --opponent-policy-format={POLICY_FORMAT_RL_GAMES}, "
            "unless stored in the run config."
        ),
    )
    parser.add_argument(
        "--mirror-opponent-x",
        action="store_true",
        help=(
            "With --peg1-only-self-play, flip opponent x-components "
            "(left-right reflection)."
        ),
    )
    parser.add_argument(
        "--mirror-opponent-y",
        action="store_true",
        help=(
            "With --peg1-only-self-play, flip opponent y-components "
            "(top-bottom reflection)."
        ),
    )
    parser.add_argument(
        "--use-collision-avoidance",
        action="store_true",
        help="Enable KlaskCollisionAvoidanceWrapper during evaluation (default: disabled)",
    )
    parser.add_argument(
        "--collision-max-vel",
        type=float,
        default=0.2,
        help="Max velocity used by KlaskCollisionAvoidanceWrapper when enabled",
    )
    parser.add_argument(
        "--deterministic-peg1",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For --peg1-only-fixed-opponent: use deterministic actions for Peg1 policy",
    )
    parser.add_argument(
        "--deterministic-self-play",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For --peg1-only-self-play: use deterministic actions for both pegs",
    )
    parser.add_argument(
        "--deterministic-opponent",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="For --peg1-only-fixed-opponent: use deterministic opponent actions",
    )
    parser.add_argument(
        "--invert-opponent-actions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="For --peg1-only-fixed-opponent: negate opponent actions before env step",
    )
    parser.add_argument(
        "--run-discriminator-analysis",
        action="store_true",
        help="Run discriminator overfitting diagnostics + marginalized reward-map analysis.",
    )
    parser.add_argument(
        "--reward-map-target",
        type=str,
        choices=["peg1", "peg2", "ball"],
        default="peg1",
        help="Target entity whose XY position is swept for reward-map analysis.",
    )
    parser.add_argument(
        "--reward-map-bins",
        type=int,
        default=80,
        help="Grid resolution (per axis) for marginalized reward-map analysis.",
    )
    parser.add_argument(
        "--reward-map-context-samples",
        type=int,
        default=4096,
        help="Max validation transitions used as contexts for marginalized reward-map.",
    )
    parser.add_argument(
        "--reward-map-legal-region-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If enabled, sweep peg targets only in their legal board half "
            "(auto-selected from context). Disable for full-board OOD maps."
        ),
    )
    parser.add_argument(
        "--disc-generated-samples",
        type=int,
        default=0,
        help=(
            "Optional cap for balanced discriminator metrics. 0 means use all available "
            "expert transitions per split and match generated count to expert count."
        ),
    )
    parser.add_argument(
        "--analysis-output-dir",
        type=str,
        default=None,
        help="Optional output directory for discriminator analysis artifacts.",
    )
    parser.add_argument(
        "--save-eval-outputs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Persist rollout-evaluation artifacts (summary.json, episode_metrics.csv, "
            "heatmaps, demo comparison plots). Defaults to disabled when "
            "--run-discriminator-analysis is enabled, otherwise enabled."
        ),
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    if int(args.reward_map_bins) <= 0:
        parser.error("--reward-map-bins must be > 0")
    if int(args.reward_map_context_samples) <= 0:
        parser.error("--reward-map-context-samples must be > 0")
    if int(args.disc_generated_samples) < 0:
        parser.error("--disc-generated-samples must be >= 0")

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config_path = os.path.join(run_dir, "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    run_cfg = dict(config.get("config", {}))

    peg1_only_fixed_opponent = (
        bool(run_cfg.get("peg1_only_fixed_opponent", False))
        if args.peg1_only_fixed_opponent is None
        else bool(args.peg1_only_fixed_opponent)
    )
    deterministic_opponent = (
        bool(run_cfg.get("fixed_opponent_deterministic", True))
        if args.deterministic_opponent is None
        else bool(args.deterministic_opponent)
    )
    invert_opponent_actions = (
        bool(run_cfg.get("invert_opponent_actions", True))
        if args.invert_opponent_actions is None
        else bool(args.invert_opponent_actions)
    )
    opponent_policy_path = None
    opponent_policy_format = None
    opponent_rl_games_config_path = None

    if args.peg1_only_self_play and peg1_only_fixed_opponent:
        parser.error("--peg1-only-self-play and --peg1-only-fixed-opponent are mutually exclusive")
    if peg1_only_fixed_opponent and (args.mirror_opponent_x or args.mirror_opponent_y):
        print("[warn] --mirror-opponent-x/--mirror-opponent-y are ignored with --peg1-only-fixed-opponent")

    demo_base = args.demo_base or run_cfg.get(
        "demo_base", str(_default_data_dir() / "expert_trajectories_cleaned")
    )
    demo_loader_cfg = resolve_demo_loader_config(config)
    if args.seed is not None:
        seed = int(args.seed)
        seed_source = "cli"
    else:
        seed = random.SystemRandom().randint(0, 2**31 - 1)
        seed_source = "auto-random"
    print(f"[setup] Using seed: {seed} ({seed_source})")

    if args.device is not None:
        device_str = args.device
    else:
        device_str = config.get("config", {}).get("device", "cuda:1")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.save_eval_outputs is None:
        save_eval_outputs = not bool(args.run_discriminator_analysis)
    else:
        save_eval_outputs = bool(args.save_eval_outputs)

    out_dir = None
    if save_eval_outputs:
        if args.output_dir:
            out_dir = args.output_dir
        else:
            out_dir = os.path.join(
                str(_default_artifacts_dir() / "evaluations" / Path(run_dir).name),
                f"eval_{timestamp}",
            )
        os.makedirs(out_dir, exist_ok=True)
    elif args.output_dir:
        print(
            "[info] --output-dir was provided but eval artifact saving is disabled; "
            "rollout outputs will not be written."
        )

    ppo_path = args.policy_path or os.path.join(run_dir, "ppo_policy_gail_final.zip")
    reward_net_path = args.reward_net_path or os.path.join(run_dir, "gail_reward_net.pt")
    if peg1_only_fixed_opponent:
        opponent_policy_path = (
            args.opponent_policy_path
            or run_cfg.get("fixed_opponent_policy_path")
            or run_cfg.get("pretrained_policy_path")
            or DEFAULT_BC_OPPONENT_POLICY_PATH
        )
        opponent_policy_path = os.path.abspath(os.path.expanduser(str(opponent_policy_path)))
        opponent_policy_format = resolve_policy_format_value(
            args.opponent_policy_format or run_cfg.get("fixed_opponent_policy_format"),
            checkpoint=opponent_policy_path,
            flag_name="--opponent-policy-format",
        )
        opponent_rl_games_config_path = (
            args.opponent_rl_games_config_path
            or run_cfg.get("fixed_opponent_rl_games_config_path")
        )
        if opponent_rl_games_config_path is not None:
            opponent_rl_games_config_path = os.path.abspath(
                os.path.expanduser(str(opponent_rl_games_config_path))
            )
        if opponent_policy_format == POLICY_FORMAT_RL_GAMES and not opponent_rl_games_config_path:
            raise ValueError(
                "RL-Games fixed-opponent evaluation requires --opponent-rl-games-config-path "
                "or a saved fixed_opponent_rl_games_config_path in config.json."
            )
    if not os.path.exists(ppo_path):
        raise FileNotFoundError(f"PPO policy not found: {ppo_path}")
    if opponent_policy_path is not None and not os.path.exists(opponent_policy_path):
        raise FileNotFoundError(f"Opponent policy not found: {opponent_policy_path}")
    if opponent_rl_games_config_path is not None and not os.path.isfile(opponent_rl_games_config_path):
        raise FileNotFoundError(
            f"Opponent RL-Games config not found: {opponent_rl_games_config_path}"
        )
    if not args.no_reward_net and not os.path.exists(reward_net_path):
        raise FileNotFoundError(f"Reward net not found: {reward_net_path}")

    player_bypass_actuator = is_human_demo_bc_checkpoint(ppo_path)
    opponent_bypass_actuator = is_human_demo_bc_checkpoint(opponent_policy_path)
    if args.peg1_only_self_play and player_bypass_actuator:
        opponent_bypass_actuator = True
    if player_bypass_actuator:
        print("[setup] bypassing actuator model on player side for human-demo BC checkpoint.")
    if opponent_bypass_actuator:
        print("[setup] bypassing actuator model on opponent side for human-demo BC checkpoint.")

    venv = None
    try:
        print(f"[setup] Creating env with {args.num_envs} parallel instances")
        venv = make_env(
            num_envs=args.num_envs,
            seed=seed,
            use_collision_avoidance=args.use_collision_avoidance,
            collision_max_vel=args.collision_max_vel,
            actuator_apply_player=not player_bypass_actuator,
            actuator_apply_opponent=not opponent_bypass_actuator,
        )

        print("[setup] Loading PPO policy")
        if args.peg1_only_self_play or peg1_only_fixed_opponent:
            # Do not bind env during load; 2D-action Peg1-only policies are space-incompatible
            # with the base 4D-action environment.
            policy = PPO.load(ppo_path, device=device)
        else:
            policy = PPO.load(ppo_path, env=venv, device=device)
        reward_action_space = (
            policy.action_space if (args.peg1_only_self_play or peg1_only_fixed_opponent) else venv.action_space
        )
        demo_action_space_for_load = (
            getattr(venv.unwrapped, "action_space", venv.action_space)
            if peg1_only_fixed_opponent
            else reward_action_space
        )

        opponent_policy = None
        if peg1_only_fixed_opponent:
            print(
                "[setup] Loading fixed opponent policy: "
                f"path={opponent_policy_path} format={opponent_policy_format}"
            )
            if opponent_policy_format == POLICY_FORMAT_SB3_PPO:
                opponent_policy = PPO.load(opponent_policy_path, device=device)
            elif opponent_policy_format == POLICY_FORMAT_RL_GAMES:
                full_action_space = getattr(venv.unwrapped, "action_space", venv.action_space)
                opponent_policy = build_rl_games_predict_policy(
                    checkpoint_path=str(opponent_policy_path),
                    config_path=str(opponent_rl_games_config_path),
                    device=str(device),
                    num_envs=int(args.num_envs),
                    observation_space=venv.observation_space,
                    action_space=full_action_space,
                )
            else:
                raise ValueError(f"Unsupported opponent policy format: {opponent_policy_format!r}")

        reward_net = None
        if args.no_reward_net:
            print("[setup] Reward net disabled")
        else:
            print("[setup] Loading reward net")
            reward_net = load_reward_net(
                reward_net_path,
                observation_space=venv.observation_space,
                action_space=reward_action_space,
                device=device,
                reward_net_type=str(
                    run_cfg.get("reward_net_type", REWARD_NET_TYPE_BASIC_SHAPED)
                ),
                reward_hid_sizes=list(run_cfg.get("disc_reward_hid_sizes", [32])),
                potential_hid_sizes=list(run_cfg.get("disc_potential_hid_sizes", [32, 32])),
            )

        print("[setup] Building policy observation layout")
        policy_layout = build_policy_layout(venv)

        print("[eval] Running evaluation")
        start = time.time()
        summary, episode_metrics, heat_agent, policy_vel, policy_action_pos = evaluate_agent(
            venv=venv,
            policy=policy,
            opponent_policy=opponent_policy,
            reward_net=reward_net,
            n_episodes=args.episodes,
            bins=args.bins,
            heatmap_targets=args.heatmap_targets,
            device=device,
            policy_layout=policy_layout,
            peg1_only_self_play=bool(args.peg1_only_self_play),
            peg1_only_fixed_opponent=bool(peg1_only_fixed_opponent),
            mirror_opponent_x=bool(args.mirror_opponent_x),
            mirror_opponent_y=bool(args.mirror_opponent_y),
            deterministic_peg1=bool(args.deterministic_peg1),
            deterministic_self_play=bool(args.deterministic_self_play),
            deterministic_opponent=bool(deterministic_opponent),
            invert_opponent_actions=bool(invert_opponent_actions),
            plot_directly_velocity_requested=bool(args.plot_directly_velocity_requested),
        )
        elapsed = time.time() - start
        summary["elapsed_sec"] = elapsed
        summary["num_envs"] = args.num_envs
        summary["episodes_requested"] = args.episodes
        summary["device"] = str(device)
        summary["policy_path"] = ppo_path
        summary["reward_net_path"] = reward_net_path if reward_net is not None else None
        summary["use_collision_avoidance"] = bool(args.use_collision_avoidance)
        summary["collision_max_vel"] = float(args.collision_max_vel)
        summary["peg1_only_self_play"] = bool(args.peg1_only_self_play)
        summary["peg1_only_fixed_opponent"] = bool(peg1_only_fixed_opponent)
        summary["opponent_policy_path"] = opponent_policy_path
        summary["opponent_policy_format"] = opponent_policy_format
        summary["opponent_rl_games_config_path"] = opponent_rl_games_config_path
        summary["deterministic_peg1"] = bool(args.deterministic_peg1)
        summary["deterministic_self_play"] = bool(args.deterministic_self_play)
        summary["deterministic_opponent"] = bool(deterministic_opponent)
        summary["invert_opponent_actions"] = bool(invert_opponent_actions)
        summary["mirror_opponent_x"] = bool(args.mirror_opponent_x)
        summary["mirror_opponent_y"] = bool(args.mirror_opponent_y)
        summary["plot_directly_velocity_requested"] = bool(args.plot_directly_velocity_requested)
        summary["action_bias_y_source"] = (
            "requested_action_y" if args.plot_directly_velocity_requested else "observed_env_y_velocity"
        )

        print(f"[eval] Completed in {elapsed:.1f}s")

        # Demo heatmap + demo reward-net alignment
        heat_demo = None
        if save_eval_outputs and not args.no_demos:
            print(f"[eval] Loading demo trajectories from {demo_base}")
            demo_paths = collect_pkl_paths(demo_base)
            demo_trajs = load_trajectories(
                demo_paths,
                action_space=demo_action_space_for_load,
                min_len=2,
                source_format=str(demo_loader_cfg["source_format"]),
                strip_sim_action_history=bool(demo_loader_cfg["strip_sim_action_history"]),
                masked_velocities=bool(demo_loader_cfg["masked_velocities"]),
            )
            if peg1_only_fixed_opponent:
                demo_trajs = slice_peg1_actions(demo_trajs, action_start=0, action_dim=2)
            demo_obs_dim = int(
                next(
                    (
                        np.asarray(tr.obs).shape[1]
                        for tr in demo_trajs
                        if not isinstance(tr.obs, dict) and np.asarray(tr.obs).ndim == 2
                    ),
                    0,
                )
            )
            demo_layout = resolve_transition_layout(
                source_format=str(demo_loader_cfg["source_format"]),
                obs_dim=demo_obs_dim,
                policy_layout=policy_layout,
            )
            summary["demo_obs_dim"] = int(demo_obs_dim)
            summary["demo_transition_layout"] = {
                key: [int(sl.start), int(sl.stop)]
                for key, sl in demo_layout.items()
            }

            print("[eval] Writing demo segmentation debug CSV")
            demo_debug_rows = build_demo_segmentation_debug_rows(demo_trajs, layout=demo_layout)
            demo_debug_csv = os.path.join(out_dir, "demo_segmentation_debug.csv")
            save_demo_segmentation_debug_csv(demo_debug_csv, demo_debug_rows)
            summary["demo_seg_debug_rows"] = int(len(demo_debug_rows))
            print(f"[out] Wrote {demo_debug_csv}")

            print("[eval] Computing demo heatmap")
            heat_demo = compute_demo_heatmap_from_trajs(
                trajs=demo_trajs,
                bins=args.bins,
                heatmap_targets=args.heatmap_targets,
                layout=demo_layout,
            )

            print("[eval] Computing demo episode metrics")
            demo_metrics = evaluate_demo_episode_metrics(demo_trajs, layout=demo_layout)
            summary.update(demo_metrics)

            if reward_net is not None:
                print("[eval] Computing demo reward-net alignment")
                demo_alignment = evaluate_reward_net_on_demos(
                    trajs=demo_trajs,
                    reward_net=reward_net,
                    device=device,
                    layout=demo_layout,
                )
                summary.update(demo_alignment)

            if not args.no_velocity_hist:
                print("[eval] Plotting velocity histograms (demo vs policy)")
                demo_vel = collect_demo_velocity_orientations(demo_trajs)
                vel_png = os.path.join(out_dir, "velocity_histograms.png")
                plot_velocity_histograms(
                    demo_vel=demo_vel,
                    policy_vel=policy_vel,
                    out_path=vel_png,
                    bins=args.velocity_hist_bins,
                    policy_label="Policy",
                )
                print(f"[out] Wrote {vel_png}")

            if not args.no_action_bias_plot:
                print("[eval] Plotting action_y vs peg1_y (demo vs policy)")
                demo_action_pos = collect_demo_action_vs_pos(demo_trajs, layout=demo_layout)
                bias_png = os.path.join(out_dir, "action_bias_vs_pos.png")
                plot_action_bias_vs_pos(
                    demo_data=demo_action_pos,
                    policy_data=policy_action_pos,
                    out_path=bias_png,
                    bins=args.action_bias_bins,
                    y_value_label=(
                        "action_y"
                        if args.plot_directly_velocity_requested
                        else "observed_vel_y"
                    ),
                )
                print(f"[out] Wrote {bias_png}")

        if args.run_discriminator_analysis:
            if reward_net is None:
                raise ValueError("Discriminator analysis requires reward-net. Remove --no-reward-net.")

            analysis_out_dir = args.analysis_output_dir or out_dir
            if analysis_out_dir is None:
                analysis_out_dir = os.path.join(run_dir, f"analysis_{timestamp}")
            os.makedirs(analysis_out_dir, exist_ok=True)
            split_path = Path(run_dir) / "dataset_split.json"
            if not split_path.exists():
                raise FileNotFoundError(
                    "Discriminator analysis requires dataset_split.json generated by split-aware training. "
                    f"Missing: {split_path}"
                )
            with split_path.open("r") as fh:
                split_payload = json.load(fh)

            train_files_raw = list(split_payload.get("train_files", []))
            val_files_raw = list(split_payload.get("val_files", []))
            if not train_files_raw or not val_files_raw:
                raise ValueError(
                    "dataset_split.json must contain non-empty train_files and val_files "
                    "for discriminator overfitting analysis."
                )

            print("[analysis] Resolving split files")
            train_paths = resolve_dataset_split_paths(
                demo_base=demo_base,
                split_paths=train_files_raw,
            )
            val_paths = resolve_dataset_split_paths(
                demo_base=demo_base,
                split_paths=val_files_raw,
            )

            print("[analysis] Loading train split trajectories")
            train_trajs = load_trajectories(
                train_paths,
                action_space=demo_action_space_for_load,
                min_len=2,
                source_format=str(demo_loader_cfg["source_format"]),
                strip_sim_action_history=bool(demo_loader_cfg["strip_sim_action_history"]),
                masked_velocities=bool(demo_loader_cfg["masked_velocities"]),
            )
            if peg1_only_fixed_opponent:
                train_trajs = slice_peg1_actions(train_trajs, action_start=0, action_dim=2)
            print("[analysis] Loading val split trajectories")
            val_trajs = load_trajectories(
                val_paths,
                action_space=demo_action_space_for_load,
                min_len=2,
                source_format=str(demo_loader_cfg["source_format"]),
                strip_sim_action_history=bool(demo_loader_cfg["strip_sim_action_history"]),
                masked_velocities=bool(demo_loader_cfg["masked_velocities"]),
            )
            if peg1_only_fixed_opponent:
                val_trajs = slice_peg1_actions(val_trajs, action_start=0, action_dim=2)

            train_batch_full = trajectories_to_transition_batch(train_trajs)
            val_batch_full = trajectories_to_transition_batch(val_trajs)
            if train_batch_full["obs"].shape[0] == 0 or val_batch_full["obs"].shape[0] == 0:
                raise RuntimeError(
                    "Train/val split produced empty transition batches for discriminator analysis."
                )
            reward_action_dim = int(reward_action_space.shape[0])
            train_batch_full = align_transition_batch_action_dim(
                train_batch_full,
                action_dim=reward_action_dim,
                label="train expert batch",
            )
            val_batch_full = align_transition_batch_action_dim(
                val_batch_full,
                action_dim=reward_action_dim,
                label="val expert batch",
            )
            analysis_layout = resolve_transition_layout(
                source_format=str(demo_loader_cfg["source_format"]),
                obs_dim=int(val_batch_full["obs"].shape[1]),
                policy_layout=policy_layout,
            )
            print(
                "[analysis] Using transition layout "
                + ", ".join(
                    f"{k}=[{int(v.start)}:{int(v.stop)}]" for k, v in analysis_layout.items()
                )
            )

            print("[analysis] Computing expert logits on train/val splits")
            train_expert_logits_full = reward_logits_batched(
                reward_net=reward_net,
                obs=train_batch_full["obs"],
                acts=train_batch_full["acts"],
                next_obs=train_batch_full["next_obs"],
                done=train_batch_full["done"],
                device=device,
            )
            val_expert_logits_full = reward_logits_batched(
                reward_net=reward_net,
                obs=val_batch_full["obs"],
                acts=val_batch_full["acts"],
                next_obs=val_batch_full["next_obs"],
                done=val_batch_full["done"],
                device=device,
            )
            expert_positive_acc_train = float(np.mean(train_expert_logits_full > 0.0))
            expert_positive_acc_val = float(np.mean(val_expert_logits_full > 0.0))

            cap = int(args.disc_generated_samples)
            train_batch_bal = subsample_transition_batch(
                train_batch_full,
                max_samples=cap if cap > 0 else int(train_batch_full["obs"].shape[0]),
                seed=int(seed) + 101,
            )
            val_batch_bal = subsample_transition_batch(
                val_batch_full,
                max_samples=cap if cap > 0 else int(val_batch_full["obs"].shape[0]),
                seed=int(seed) + 202,
            )

            print("[analysis] Collecting generated transitions (train-sized)")
            train_generated_batch = collect_generated_transitions(
                venv=venv,
                policy=policy,
                opponent_policy=opponent_policy,
                n_samples=int(train_batch_bal["obs"].shape[0]),
                policy_layout=policy_layout,
                peg1_only_self_play=bool(args.peg1_only_self_play),
                peg1_only_fixed_opponent=bool(peg1_only_fixed_opponent),
                mirror_opponent_x=bool(args.mirror_opponent_x),
                mirror_opponent_y=bool(args.mirror_opponent_y),
                deterministic_peg1=bool(args.deterministic_peg1),
                deterministic_self_play=bool(args.deterministic_self_play),
                deterministic_opponent=bool(deterministic_opponent),
                invert_opponent_actions=bool(invert_opponent_actions),
            )
            print("[analysis] Collecting generated transitions (val-sized)")
            val_generated_batch = collect_generated_transitions(
                venv=venv,
                policy=policy,
                opponent_policy=opponent_policy,
                n_samples=int(val_batch_bal["obs"].shape[0]),
                policy_layout=policy_layout,
                peg1_only_self_play=bool(args.peg1_only_self_play),
                peg1_only_fixed_opponent=bool(peg1_only_fixed_opponent),
                mirror_opponent_x=bool(args.mirror_opponent_x),
                mirror_opponent_y=bool(args.mirror_opponent_y),
                deterministic_peg1=bool(args.deterministic_peg1),
                deterministic_self_play=bool(args.deterministic_self_play),
                deterministic_opponent=bool(deterministic_opponent),
                invert_opponent_actions=bool(invert_opponent_actions),
            )
            train_generated_batch = align_transition_batch_action_dim(
                train_generated_batch,
                action_dim=reward_action_dim,
                label="train generated batch",
            )
            val_generated_batch = align_transition_batch_action_dim(
                val_generated_batch,
                action_dim=reward_action_dim,
                label="val generated batch",
            )

            train_expert_logits_bal = reward_logits_batched(
                reward_net=reward_net,
                obs=train_batch_bal["obs"],
                acts=train_batch_bal["acts"],
                next_obs=train_batch_bal["next_obs"],
                done=train_batch_bal["done"],
                device=device,
            )
            val_expert_logits_bal = reward_logits_batched(
                reward_net=reward_net,
                obs=val_batch_bal["obs"],
                acts=val_batch_bal["acts"],
                next_obs=val_batch_bal["next_obs"],
                done=val_batch_bal["done"],
                device=device,
            )
            train_generated_logits = reward_logits_batched(
                reward_net=reward_net,
                obs=train_generated_batch["obs"],
                acts=train_generated_batch["acts"],
                next_obs=train_generated_batch["next_obs"],
                done=train_generated_batch["done"],
                device=device,
            )
            val_generated_logits = reward_logits_batched(
                reward_net=reward_net,
                obs=val_generated_batch["obs"],
                acts=val_generated_batch["acts"],
                next_obs=val_generated_batch["next_obs"],
                done=val_generated_batch["done"],
                device=device,
            )

            balanced_train = compute_balanced_binary_metrics(
                expert_logits=train_expert_logits_bal,
                generated_logits=train_generated_logits,
            )
            balanced_val = compute_balanced_binary_metrics(
                expert_logits=val_expert_logits_bal,
                generated_logits=val_generated_logits,
            )
            balanced_gaps = {}
            for key in (
                "accuracy_total",
                "accuracy_expert",
                "accuracy_generated",
                "precision_expert",
                "recall_expert",
                "f1_expert",
            ):
                train_v = float(balanced_train.get(key, float("nan")))
                val_v = float(balanced_val.get(key, float("nan")))
                balanced_gaps[f"{key}_train_minus_val"] = (
                    float(train_v - val_v)
                    if np.isfinite(train_v) and np.isfinite(val_v)
                    else float("nan")
                )

            context_batch = subsample_transition_batch(
                val_batch_full,
                max_samples=int(args.reward_map_context_samples),
                seed=int(seed) + 303,
            )
            print(
                f"[analysis] Computing marginalized reward map target={args.reward_map_target} "
                f"bins={int(args.reward_map_bins)} contexts={int(context_batch['obs'].shape[0])}"
            )
            reward_map, reward_map_meta = compute_marginalized_reward_map(
                reward_net=reward_net,
                context_batch=context_batch,
                target=str(args.reward_map_target),
                layout=analysis_layout,
                bins=int(args.reward_map_bins),
                legal_region_only=bool(args.reward_map_legal_region_only),
                device=device,
            )

            reward_map_npy = os.path.join(
                analysis_out_dir,
                f"reward_map_{args.reward_map_target}.npy",
            )
            reward_map_png = os.path.join(
                analysis_out_dir,
                f"reward_map_{args.reward_map_target}.png",
            )
            reward_map_meta_path = os.path.join(
                analysis_out_dir,
                f"reward_map_{args.reward_map_target}_metadata.json",
            )
            np.save(reward_map_npy, reward_map)
            plot_reward_map(
                reward_map,
                out_path=reward_map_png,
                target=str(args.reward_map_target),
                legal_bounds=reward_map_meta.get("legal_bounds", None),
            )
            with open(reward_map_meta_path, "w") as fh:
                json.dump(reward_map_meta, fh, indent=2)

            _, _, _, peg1_ball_proximity_legal_bounds = _build_reward_grid_and_valid_mask(
                obs_ctx=context_batch["obs"],
                target="peg1",
                target_slice=analysis_layout["peg1_pos"],
                bins=max(2, int(args.reward_map_bins)),
                legal_region_only=bool(args.reward_map_legal_region_only),
            )

            def _score_counterfactual_logits(
                obs_batch: np.ndarray,
                acts_batch: np.ndarray,
                next_obs_batch: np.ndarray,
                done_batch: np.ndarray,
            ) -> np.ndarray:
                return reward_logits_batched(
                    reward_net=reward_net,
                    obs=obs_batch,
                    acts=acts_batch,
                    next_obs=next_obs_batch,
                    done=done_batch,
                    device=device,
                )

            print(
                "[analysis] Computing peg1-ball proximity probe "
                f"distances={PEG_BALL_PROXIMITY_DISTANCE_SAMPLES} "
                f"angles={PEG_BALL_PROXIMITY_ANGLE_SAMPLES} contexts={int(context_batch['obs'].shape[0])}"
            )
            peg1_ball_proximity = compute_peg_ball_proximity_profile(
                context_batch=context_batch,
                layout=analysis_layout,
                score_fn=_score_counterfactual_logits,
                field_x=FIELD_X,
                field_y=FIELD_Y,
                peg_key="peg1_pos",
                ball_key="ball_pos",
                distance_samples=PEG_BALL_PROXIMITY_DISTANCE_SAMPLES,
                angle_samples=PEG_BALL_PROXIMITY_ANGLE_SAMPLES,
                legal_bounds=peg1_ball_proximity_legal_bounds,
            )
            peg1_ball_proximity_png = os.path.join(
                analysis_out_dir,
                "peg1_ball_proximity_profile.png",
            )
            plot_peg_ball_proximity_profile(
                peg1_ball_proximity,
                out_path=peg1_ball_proximity_png,
            )

            print(
                f"[analysis] Computing visited-state reward map target={args.reward_map_target} "
                f"bins={int(args.reward_map_bins)} samples={int(val_generated_batch['obs'].shape[0])}"
            )
            visited_reward_map, visited_reward_counts, visited_reward_meta = compute_visited_state_reward_map(
                transition_batch=val_generated_batch,
                logits=val_generated_logits,
                target=str(args.reward_map_target),
                layout=analysis_layout,
                bins=int(args.reward_map_bins),
                legal_region_only=bool(args.reward_map_legal_region_only),
            )
            visited_reward_map_npy = os.path.join(
                analysis_out_dir,
                f"visited_reward_map_{args.reward_map_target}.npy",
            )
            visited_reward_counts_npy = os.path.join(
                analysis_out_dir,
                f"visited_reward_map_{args.reward_map_target}_counts.npy",
            )
            visited_reward_map_png = os.path.join(
                analysis_out_dir,
                f"visited_reward_map_{args.reward_map_target}.png",
            )
            visited_reward_map_meta_path = os.path.join(
                analysis_out_dir,
                f"visited_reward_map_{args.reward_map_target}_metadata.json",
            )
            np.save(visited_reward_map_npy, visited_reward_map)
            np.save(visited_reward_counts_npy, visited_reward_counts)
            plot_reward_map(
                visited_reward_map,
                out_path=visited_reward_map_png,
                target=f"{args.reward_map_target}, visited states",
                legal_bounds=visited_reward_meta.get("legal_bounds", None),
            )
            with open(visited_reward_map_meta_path, "w") as fh:
                json.dump(visited_reward_meta, fh, indent=2)

            logits_hist_png = os.path.join(analysis_out_dir, "discriminator_logits_hist.png")
            plot_discriminator_logits_hist(
                train_expert_logits=train_expert_logits_bal,
                val_expert_logits=val_expert_logits_bal,
                train_generated_logits=train_generated_logits,
                val_generated_logits=val_generated_logits,
                out_path=logits_hist_png,
            )

            discriminator_metrics_path = os.path.join(analysis_out_dir, "discriminator_metrics.json")
            discriminator_metrics = {
                "created_at": datetime.datetime.now().isoformat(),
                "run_dir": run_dir,
                "analysis_output_dir": analysis_out_dir,
                "dataset_split_path": str(split_path),
                "demo_source_format": str(demo_loader_cfg["source_format"]),
                "strip_sim_action_history": bool(demo_loader_cfg["strip_sim_action_history"]),
                "masked_velocities": bool(demo_loader_cfg["masked_velocities"]),
                "analysis_transition_layout": {
                    key: [int(sl.start), int(sl.stop)]
                    for key, sl in analysis_layout.items()
                },
                "reward_map_target": str(args.reward_map_target),
                "reward_map_bins": int(args.reward_map_bins),
                "reward_map_context_samples_used": int(context_batch["obs"].shape[0]),
                "reward_map_legal_region_only": bool(args.reward_map_legal_region_only),
                "reward_map_evaluated_cells": int(reward_map_meta.get("evaluated_cells", 0)),
                "reward_map_skipped_cells": int(reward_map_meta.get("skipped_cells", 0)),
                "peg1_ball_proximity_distance_samples": int(PEG_BALL_PROXIMITY_DISTANCE_SAMPLES),
                "peg1_ball_proximity_angle_samples": int(PEG_BALL_PROXIMITY_ANGLE_SAMPLES),
                "peg1_ball_proximity": peg1_ball_proximity,
                "visited_reward_map_samples_used": int(visited_reward_meta.get("samples_used", 0)),
                "visited_reward_map_visited_cells": int(visited_reward_meta.get("visited_cells", 0)),
                "disc_generated_samples_cap": int(args.disc_generated_samples),
                "split_counts": {
                    "train_files": int(len(train_paths)),
                    "val_files": int(len(val_paths)),
                    "train_expert_steps_full": int(train_batch_full["obs"].shape[0]),
                    "val_expert_steps_full": int(val_batch_full["obs"].shape[0]),
                    "train_expert_steps_balanced": int(train_batch_bal["obs"].shape[0]),
                    "val_expert_steps_balanced": int(val_batch_bal["obs"].shape[0]),
                },
                "expert_positive_accuracy": {
                    "train": float(expert_positive_acc_train),
                    "val": float(expert_positive_acc_val),
                    "train_minus_val": float(expert_positive_acc_train - expert_positive_acc_val),
                },
                "balanced_metrics": {
                    "train": balanced_train,
                    "val": balanced_val,
                    "gaps_train_minus_val": balanced_gaps,
                },
            }
            with open(discriminator_metrics_path, "w") as fh:
                json.dump(discriminator_metrics, fh, indent=2)

            analysis_summary_path = os.path.join(analysis_out_dir, "analysis_summary.json")
            analysis_summary = {
                "run_dir": run_dir,
                "analysis_output_dir": analysis_out_dir,
                "expert_positive_accuracy_train": float(expert_positive_acc_train),
                "expert_positive_accuracy_val": float(expert_positive_acc_val),
                "balanced_accuracy_total_train": float(balanced_train["accuracy_total"]),
                "balanced_accuracy_total_val": float(balanced_val["accuracy_total"]),
                "balanced_accuracy_total_train_minus_val": float(
                    balanced_gaps["accuracy_total_train_minus_val"]
                ),
                "reward_map_target": str(args.reward_map_target),
                "reward_map_bins": int(args.reward_map_bins),
                "reward_map_context_samples_used": int(context_batch["obs"].shape[0]),
                "reward_map_legal_region_only": bool(args.reward_map_legal_region_only),
                "reward_map_evaluated_cells": int(reward_map_meta.get("evaluated_cells", 0)),
                "reward_map_skipped_cells": int(reward_map_meta.get("skipped_cells", 0)),
                "reward_map_npy": reward_map_npy,
                "reward_map_png": reward_map_png,
                "reward_map_metadata_json": reward_map_meta_path,
                "peg1_ball_proximity_png": peg1_ball_proximity_png,
                "peg1_ball_proximity_near_minus_far_mean_logit": float(
                    peg1_ball_proximity["summary"].get("near_minus_far_mean_logit", float("nan"))
                ),
                "peg1_ball_proximity_spearman_distance_vs_mean_logit": float(
                    peg1_ball_proximity["summary"].get("spearman_distance_vs_mean_logit", float("nan"))
                ),
                "visited_reward_map_npy": visited_reward_map_npy,
                "visited_reward_map_counts_npy": visited_reward_counts_npy,
                "visited_reward_map_png": visited_reward_map_png,
                "visited_reward_map_metadata_json": visited_reward_map_meta_path,
                "logits_hist_png": logits_hist_png,
                "discriminator_metrics_json": discriminator_metrics_path,
            }
            with open(analysis_summary_path, "w") as fh:
                json.dump(analysis_summary, fh, indent=2)

            summary["discriminator_analysis"] = {
                "enabled": True,
                "analysis_output_dir": analysis_out_dir,
                "discriminator_metrics_path": discriminator_metrics_path,
                "analysis_summary_path": analysis_summary_path,
                "reward_map_path": reward_map_npy,
                "reward_map_plot_path": reward_map_png,
                "peg1_ball_proximity_plot_path": peg1_ball_proximity_png,
                "visited_reward_map_path": visited_reward_map_npy,
                "visited_reward_map_plot_path": visited_reward_map_png,
                "logits_hist_path": logits_hist_png,
            }
            print(f"[out] Wrote {discriminator_metrics_path}")
            print(f"[out] Wrote {reward_map_npy}")
            print(f"[out] Wrote {reward_map_png}")
            print(f"[out] Wrote {peg1_ball_proximity_png}")
            print(f"[out] Wrote {visited_reward_map_npy}")
            print(f"[out] Wrote {visited_reward_map_png}")
            print(f"[out] Wrote {logits_hist_png}")
            print(f"[out] Wrote {analysis_summary_path}")

        # Save outputs
        if save_eval_outputs and out_dir is not None:
            summary_path = os.path.join(out_dir, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[out] Wrote {summary_path}")

            episode_csv = os.path.join(out_dir, "episode_metrics.csv")
            save_episode_metrics(episode_csv, episode_metrics)
            print(f"[out] Wrote {episode_csv}")

            for target in args.heatmap_targets:
                np.save(os.path.join(out_dir, f"heatmap_agent_{target}.npy"), heat_agent[target])
                if heat_demo is not None:
                    np.save(os.path.join(out_dir, f"heatmap_demo_{target}.npy"), heat_demo[target])

            heatmap_png = os.path.join(out_dir, "heatmap_agent_vs_demo.png")
            plot_heatmaps(heat_agent, heat_demo, heatmap_png, args.heatmap_targets)
            print(f"[out] Wrote {heatmap_png}")
        else:
            print("[out] Eval artifact saving disabled; wrote discriminator analysis only.")

    finally:
        close_env(venv)


if __name__ == "__main__":
    main()
