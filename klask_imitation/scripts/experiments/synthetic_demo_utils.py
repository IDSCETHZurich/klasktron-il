"""Utilities for generating synthetic demonstration datasets."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO


POLICY_FORMAT_SB3_PPO = "sb3_ppo"
POLICY_FORMAT_RL_GAMES = "rl_games"
VALID_POLICY_FORMATS = (POLICY_FORMAT_SB3_PPO, POLICY_FORMAT_RL_GAMES)

MODE_PEG1_FIXED_OPPONENT = "peg1_fixed_opponent"
MODE_PEG1_SELF_PLAY = "peg1_self_play"
MODE_FULL_4D = "full_4d"
VALID_EXPERT_MODES = (
    MODE_PEG1_FIXED_OPPONENT,
    MODE_PEG1_SELF_PLAY,
    MODE_FULL_4D,
)


@dataclass
class SyntheticExpertSpec:
    expert_id: str
    policy_path: str
    mode: str
    policy_format: str = POLICY_FORMAT_SB3_PPO
    opponent_policy_path: Optional[str] = None
    deterministic_expert: bool = True
    deterministic_opponent: bool = True
    invert_opponent_actions: bool = True
    mirror_opponent_x: bool = False
    mirror_opponent_y: bool = False
    rl_games_task: str = "Isaac-Klask-v0"
    rl_games_config_path: Optional[str] = None
    action_clip: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def unwrap_obs(obs: Any) -> Any:
    if isinstance(obs, tuple):
        return obs[0]
    return obs


def vec_step(venv: Any, action: np.ndarray) -> Tuple[Any, np.ndarray, np.ndarray, Any]:
    step_action: Any = action
    if not torch.is_tensor(step_action) and not hasattr(venv, "step_async"):
        sim_device = getattr(getattr(venv, "unwrapped", venv), "device", None)
        if sim_device is not None:
            step_action = torch.as_tensor(action, dtype=torch.float32, device=sim_device)
    out = venv.step(step_action)
    if len(out) == 4:
        obs, reward, done, info = out
    else:
        obs, reward, terminated, truncated, info = out
        done = np.logical_or(terminated, truncated)
    return obs, to_numpy(reward), to_numpy(done).astype(bool), info


def obs_to_array(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        if "policy" in obs:
            return to_numpy(obs["policy"])
        return to_numpy(next(iter(obs.values())))
    return to_numpy(obs)


def ensure_2d_batch(arr: Any, dtype=np.float32) -> np.ndarray:
    out = to_numpy(arr).astype(dtype, copy=False)
    if out.ndim == 1:
        out = out[None, :]
    return out


class RLGamesPlayerAdapter:
    """Adapter exposing SB3-like predict() using RL-Games BasePlayer."""

    def __init__(self, player: Any):
        self.player = player
        self._batch_initialized = False

    def predict(self, obs: Any, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        obs_arr = ensure_2d_batch(obs, dtype=np.float32)
        with torch.inference_mode():
            obs_torch = self.player.obs_to_torch(obs_arr)
            if not self._batch_initialized:
                _ = self.player.get_batch_size(obs_torch, 1)
                if getattr(self.player, "is_rnn", False):
                    self.player.init_rnn()
                self._batch_initialized = True
            actions = self.player.get_action(
                obs_torch, is_deterministic=bool(deterministic)
            )
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = np.asarray(actions, dtype=np.float32)
        return ensure_2d_batch(actions_np, dtype=np.float32), None

    def on_dones(self, done: np.ndarray) -> None:
        """Reset RNN states for terminated envs, matching play_klask behavior."""
        if not getattr(self.player, "is_rnn", False):
            return
        states = getattr(self.player, "states", None)
        if states is None:
            return
        done_np = np.asarray(done, dtype=bool).reshape(-1)
        if done_np.size == 0 or not done_np.any():
            return
        done_t = torch.as_tensor(done_np, device=states[0].device, dtype=torch.bool)
        for s in states:
            s[:, done_t, :] = 0.0


def load_rl_games_player_like_play_klask(
    *,
    checkpoint_path: str,
    device: str,
    num_envs: int,
    task: str = "Isaac-Klask-v0",
    config_path: Optional[str] = None,
    env_info: Optional[Dict[str, Any]] = None,
) -> RLGamesPlayerAdapter:
    """Load RL-Games player using the same flow as play_klask.py.

    This follows:
    - load_cfg_from_registry(task, "rl_games_cfg_entry_point")
    - optional YAML override
    - Runner.load(...)
    - create_player()
    - restore(checkpoint)
    - reset()
    - move tensors to target device
    """
    from rl_games.torch_runner import Runner
    agent_cfg = load_rl_games_agent_cfg(task=task, config_path=config_path)

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint_path
    agent_cfg["params"]["config"]["num_actors"] = int(num_envs)
    agent_cfg["params"]["config"]["device"] = device
    if "device_name" in agent_cfg["params"]["config"]:
        agent_cfg["params"]["config"]["device_name"] = device
    if env_info is not None:
        # If env_info is provided, RL-Games BasePlayer will skip env creation.
        agent_cfg["params"]["config"]["env_info"] = env_info

    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.restore(checkpoint_path)
    player.reset()

    # Keep this identical to play_klask.py.
    player.device = torch.device(device)
    player.model.to(device)
    if hasattr(player, "actions_low"):
        player.actions_low = player.actions_low.to(device)
    if hasattr(player, "actions_high"):
        player.actions_high = player.actions_high.to(device)

    return RLGamesPlayerAdapter(player)


def load_rl_games_agent_cfg(*, task: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    import yaml
    from isaaclab_tasks.utils import load_cfg_from_registry

    # If an explicit RL-Games config is provided and it is a full config
    # (contains "params"), use it directly to avoid stale registry paths.
    if config_path is not None:
        with open(config_path, "r") as fh:
            explicit_cfg = yaml.safe_load(fh)
        if not isinstance(explicit_cfg, dict):
            raise ValueError(f"RL-Games config at {config_path} is not a dictionary.")
        if "params" in explicit_cfg:
            return explicit_cfg

    agent_cfg: Dict[str, Any] = {}
    try:
        agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
    except FileNotFoundError as exc:
        if config_path is None:
            raise
        # Some local installs register a stale/missing rl_games_cfg_entry_point path.
        # In that case, we fall back to the user-provided config file directly.
        print(
            "[warn] failed to load rl_games_cfg_entry_point from registry "
            f"for task={task!r}: {exc}. Falling back to --rl-games-config-path."
        )

    if config_path is not None:
        cfg_override = explicit_cfg
        if agent_cfg:
            agent_cfg.update(cfg_override)
        else:
            agent_cfg = cfg_override
    return agent_cfg


def load_expert_policy(
    *,
    expert_spec: SyntheticExpertSpec,
    device: str,
    num_envs: int,
    env_info: Optional[Dict[str, Any]] = None,
) -> Any:
    """Load expert policy according to requested format."""
    if expert_spec.policy_format == POLICY_FORMAT_SB3_PPO:
        return PPO.load(expert_spec.policy_path, device=device)
    if expert_spec.policy_format == POLICY_FORMAT_RL_GAMES:
        return load_rl_games_player_like_play_klask(
            checkpoint_path=expert_spec.policy_path,
            device=device,
            num_envs=num_envs,
            task=expert_spec.rl_games_task,
            config_path=expert_spec.rl_games_config_path,
            env_info=env_info,
        )
    raise ValueError(
        f"Unsupported policy_format={expert_spec.policy_format!r}. "
        f"Valid: {VALID_POLICY_FORMATS}"
    )


def build_rl_games_env_info_from_spaces(
    *,
    observation_space: Optional[Any] = None,
    action_space: Optional[Any] = None,
    obs_dim: Optional[int] = None,
    action_dims: int = 2,
    action_clip: Optional[float] = None,
) -> Dict[str, Any]:
    """Build RL-Games env_info and force action dims to match policy head."""
    import gym

    if obs_dim is not None:
        obs_dim_int = int(obs_dim)
        if obs_dim_int <= 0:
            raise ValueError(f"obs_dim must be >0, got {obs_dim_int}")
        obs_low = np.full((obs_dim_int,), -np.inf, dtype=np.float32)
        obs_high = np.full((obs_dim_int,), np.inf, dtype=np.float32)
    else:
        if observation_space is None:
            raise ValueError("Either obs_dim or observation_space must be provided.")
        if not hasattr(observation_space, "low") or not hasattr(observation_space, "high"):
            raise TypeError("observation_space must expose low/high arrays for RL-Games env_info.")
        obs_low = np.asarray(observation_space.low, dtype=np.float32)
        obs_high = np.asarray(observation_space.high, dtype=np.float32)

    if action_clip is not None:
        clip = abs(float(action_clip))
        if not np.isfinite(clip) or clip <= 0.0:
            clip = 1.0
        act_low = -np.full((int(action_dims),), clip, dtype=np.float32)
        act_high = np.full((int(action_dims),), clip, dtype=np.float32)
    else:
        if action_space is None:
            raise ValueError("Either action_clip or action_space must be provided.")
        if not hasattr(action_space, "low") or not hasattr(action_space, "high"):
            raise TypeError("action_space must expose low/high arrays for RL-Games env_info.")
        act_low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
        act_high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
        if act_low.shape[0] < int(action_dims):
            raise ValueError(
                f"Action space has {act_low.shape[0]} dims, but {action_dims} requested."
            )
        act_low = act_low[:action_dims]
        act_high = act_high[:action_dims]

    return {
        "observation_space": gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        ),
        "action_space": gym.spaces.Box(
            low=act_low,
            high=act_high,
            dtype=np.float32,
        ),
        "state_space": None,
    }


def build_policy_obs_slices(venv: Any) -> Dict[str, slice]:
    env = venv.unwrapped
    obs_mgr = env.observation_manager
    group = "policy"
    if group not in obs_mgr.active_terms:
        raise KeyError(f"Observation group '{group}' not found.")
    names = obs_mgr.active_terms[group]
    dims = obs_mgr.group_obs_term_dim[group]
    idx = 0
    out: Dict[str, slice] = {}
    for name, dim in zip(names, dims):
        width = int(np.prod(dim))
        out[name] = slice(idx, idx + width)
        idx += width
    return out


def build_policy_layout(venv: Any) -> Dict[str, slice]:
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
        raise KeyError(f"Missing observation terms: {missing}")
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


def build_training_style_opponent_obs(obs: Any) -> np.ndarray:
    """Remap exactly like OpponentVecEnvWrapper._to_opponent_obs in training."""
    obs_arr = ensure_2d_batch(obs_to_array(obs), dtype=np.float32)
    if obs_arr.shape[1] < 12:
        raise ValueError(f"Expected >=12 observation dims, got {obs_arr.shape[1]}.")
    remapped = np.array(obs_arr, copy=True)
    remapped[:, :8] = obs_arr[:, [4, 5, 6, 7, 0, 1, 2, 3]]
    return remapped


def build_opponent_obs_from_policy_obs(
    obs: Any,
    policy_layout: Dict[str, slice],
    *,
    mirror_x: bool,
    mirror_y: bool,
) -> np.ndarray:
    src = ensure_2d_batch(obs_to_array(obs), dtype=np.float32)
    dst = np.array(src, copy=True)

    p1 = policy_layout["peg1_pos"]
    p2 = policy_layout["peg2_pos"]
    p1_pos = src[:, p1].copy()
    p2_pos = src[:, p2].copy()
    dst[:, p1] = p2_pos
    dst[:, p2] = p1_pos

    p1_vx = policy_layout["peg1_vel_x"].start
    p1_vy = policy_layout["peg1_vel_y"].start
    p2_vx = policy_layout["peg2_vel_x"].start
    p2_vy = policy_layout["peg2_vel_y"].start
    dst[:, p1_vx] = src[:, p2_vx]
    dst[:, p1_vy] = src[:, p2_vy]
    dst[:, p2_vx] = src[:, p1_vx]
    dst[:, p2_vy] = src[:, p1_vy]

    if mirror_x:
        dst[:, p1.start + 0] *= -1.0
        dst[:, p2.start + 0] *= -1.0
        dst[:, p1_vx] *= -1.0
        dst[:, p2_vx] *= -1.0
        dst[:, policy_layout["ball_pos"].start + 0] *= -1.0
        dst[:, policy_layout["ball_vel"].start + 0] *= -1.0
    if mirror_y:
        dst[:, p1.start + 1] *= -1.0
        dst[:, p2.start + 1] *= -1.0
        dst[:, p1_vy] *= -1.0
        dst[:, p2_vy] *= -1.0
        dst[:, policy_layout["ball_pos"].start + 1] *= -1.0
        dst[:, policy_layout["ball_vel"].start + 1] *= -1.0
    return dst


def unmirror_opponent_action(
    action_opp: Any,
    *,
    mirror_x: bool,
    mirror_y: bool,
) -> np.ndarray:
    out = ensure_2d_batch(action_opp, dtype=np.float32)
    if out.shape[1] < 2:
        raise ValueError(f"Opponent action must have >=2 dims, got {out.shape}.")
    opp = out[:, :2].copy()
    if mirror_x:
        opp[:, 0] *= -1.0
    if mirror_y:
        opp[:, 1] *= -1.0
    return opp


def maybe_clip_actions(action: np.ndarray, expert_spec: SyntheticExpertSpec) -> np.ndarray:
    if expert_spec.action_clip is None:
        return action
    clip = abs(float(expert_spec.action_clip))
    if not np.isfinite(clip) or clip <= 0.0:
        return action
    return np.clip(action, -clip, clip)


def build_full_action(
    *,
    obs: Any,
    expert_policy: Any,
    expert_spec: SyntheticExpertSpec,
    opponent_policy: Optional[Any],
    policy_layout: Optional[Dict[str, slice]] = None,
) -> np.ndarray:
    obs_arr = ensure_2d_batch(obs_to_array(obs), dtype=np.float32)
    mode = expert_spec.mode

    if mode == MODE_FULL_4D:
        action, _ = expert_policy.predict(
            obs_arr, deterministic=expert_spec.deterministic_expert
        )
        action = ensure_2d_batch(action, dtype=np.float32)
        if action.shape[1] < 4:
            raise ValueError(
                f"full_4d mode requires >=4 action dims from policy, got {action.shape}."
            )
        return maybe_clip_actions(action[:, :4], expert_spec).astype(np.float32)

    if mode == MODE_PEG1_FIXED_OPPONENT:
        if opponent_policy is None:
            raise ValueError("peg1_fixed_opponent mode requires opponent policy.")
        act_peg1, _ = expert_policy.predict(
            obs_arr, deterministic=expert_spec.deterministic_expert
        )
        act_peg1 = ensure_2d_batch(act_peg1, dtype=np.float32)[:, :2]
        opp_obs = build_training_style_opponent_obs(obs_arr)
        act_opp, _ = opponent_policy.predict(
            opp_obs, deterministic=expert_spec.deterministic_opponent
        )
        act_opp = ensure_2d_batch(act_opp, dtype=np.float32)[:, :2]
        if expert_spec.invert_opponent_actions:
            act_opp = -act_opp
        act_opp = np.clip(act_opp, -1.0, 1.0)
        out = np.concatenate([act_peg1, act_opp], axis=1).astype(np.float32)
        return maybe_clip_actions(out, expert_spec).astype(np.float32)

    if mode == MODE_PEG1_SELF_PLAY:
        act_peg1, _ = expert_policy.predict(
            obs_arr, deterministic=expert_spec.deterministic_expert
        )
        act_peg1 = ensure_2d_batch(act_peg1, dtype=np.float32)[:, :2]
        if expert_spec.policy_format == POLICY_FORMAT_RL_GAMES:
            # Match play_klask self-play path:
            # opponent acts on negated opponent-view observation and we apply
            # full_action = [peg1_action, -opponent_action].
            if isinstance(obs, dict) and "opponent" in obs:
                opp_obs = ensure_2d_batch(obs_to_array(obs["opponent"]), dtype=np.float32)
            else:
                if policy_layout is None:
                    raise ValueError("peg1_self_play mode requires policy layout.")
                opp_obs = build_opponent_obs_from_policy_obs(
                    obs_arr,
                    policy_layout,
                    mirror_x=False,
                    mirror_y=False,
                )
            if opp_obs.shape[1] >= 12:
                opp_obs = np.array(opp_obs, copy=True)
                opp_obs[:, :12] *= -1.0
            act_opp_local, _ = expert_policy.predict(
                opp_obs, deterministic=expert_spec.deterministic_opponent
            )
            act_opp = ensure_2d_batch(act_opp_local, dtype=np.float32)[:, :2]
            act_opp = -act_opp
        else:
            if policy_layout is None:
                raise ValueError("peg1_self_play mode requires policy layout.")
            opp_obs = build_opponent_obs_from_policy_obs(
                obs_arr,
                policy_layout,
                mirror_x=expert_spec.mirror_opponent_x,
                mirror_y=expert_spec.mirror_opponent_y,
            )
            act_opp_local, _ = expert_policy.predict(
                opp_obs, deterministic=expert_spec.deterministic_opponent
            )
            act_opp = unmirror_opponent_action(
                act_opp_local,
                mirror_x=expert_spec.mirror_opponent_x,
                mirror_y=expert_spec.mirror_opponent_y,
            )
        out = np.concatenate([act_peg1, act_opp], axis=1).astype(np.float32)
        return maybe_clip_actions(out, expert_spec).astype(np.float32)

    raise ValueError(f"Unsupported mode={mode!r}")


def collect_synthetic_trajectories(
    *,
    venv: Any,
    expert_policy: Any,
    expert_spec: SyntheticExpertSpec,
    target_steps: int,
    opponent_policy: Optional[Any] = None,
    min_traj_len: int = 2,
) -> List[Dict[str, Any]]:
    """Roll out expert policy and return sim-native trajectories."""
    if target_steps <= 0:
        raise ValueError("target_steps must be > 0")
    if expert_spec.mode not in VALID_EXPERT_MODES:
        raise ValueError(f"Invalid expert mode {expert_spec.mode!r}")

    policy_layout: Optional[Dict[str, slice]] = None
    if expert_spec.mode == MODE_PEG1_SELF_PLAY:
        policy_layout = build_policy_layout(venv)

    obs = unwrap_obs(venv.reset())
    obs_arr = ensure_2d_batch(obs_to_array(obs), dtype=np.float32)
    num_envs = obs_arr.shape[0]

    obs_buffers: List[List[np.ndarray]] = [[obs_arr[i].copy()] for i in range(num_envs)]
    act_buffers: List[List[np.ndarray]] = [[] for _ in range(num_envs)]
    trajectories: List[Dict[str, Any]] = []
    collected_steps = 0

    while collected_steps < target_steps:
        action = build_full_action(
            obs=obs,
            expert_policy=expert_policy,
            expert_spec=expert_spec,
            opponent_policy=opponent_policy,
            policy_layout=policy_layout,
        )
        next_obs, _, done, _ = vec_step(venv, action)
        next_obs = unwrap_obs(next_obs)
        next_arr = ensure_2d_batch(obs_to_array(next_obs), dtype=np.float32)
        action_arr = ensure_2d_batch(action, dtype=np.float32)

        for i in range(num_envs):
            act_buffers[i].append(action_arr[i].copy())
            obs_buffers[i].append(next_arr[i].copy())
            collected_steps += 1

            if done[i]:
                if len(act_buffers[i]) >= min_traj_len:
                    trajectories.append(
                        {
                            "obs": np.asarray(obs_buffers[i], dtype=np.float32),
                            "acts": np.asarray(act_buffers[i], dtype=np.float32),
                            "rews": np.zeros((len(act_buffers[i]),), dtype=np.float32),
                            "infos": None,
                            "terminal": True,
                        }
                    )
                obs_buffers[i] = [next_arr[i].copy()]
                act_buffers[i] = []
        # Keep RL-Games RNN state handling aligned with play_klask.py.
        if hasattr(expert_policy, "on_dones"):
            expert_policy.on_dones(done)
        if opponent_policy is not None and hasattr(opponent_policy, "on_dones"):
            opponent_policy.on_dones(done)
        obs = next_obs

    # Flush remaining partial trajectories.
    for i in range(num_envs):
        if len(act_buffers[i]) >= min_traj_len:
            trajectories.append(
                {
                    "obs": np.asarray(obs_buffers[i], dtype=np.float32),
                    "acts": np.asarray(act_buffers[i], dtype=np.float32),
                    "rews": np.zeros((len(act_buffers[i]),), dtype=np.float32),
                    "infos": None,
                    "terminal": False,
                }
            )
    return trim_trajectories_to_target_steps(trajectories, target_steps=target_steps)


def trim_trajectories_to_target_steps(
    trajectories: List[Dict[str, Any]],
    *,
    target_steps: int,
) -> List[Dict[str, Any]]:
    """Trim concatenated trajectory stream to exactly target_steps."""
    out: List[Dict[str, Any]] = []
    total = 0
    for tr in trajectories:
        acts = np.asarray(tr["acts"], dtype=np.float32)
        obs = np.asarray(tr["obs"], dtype=np.float32)
        rews = np.asarray(tr.get("rews", np.zeros((len(acts),), dtype=np.float32)))
        t = len(acts)
        if t <= 0:
            continue

        remaining = target_steps - total
        if remaining <= 0:
            break

        if t <= remaining:
            out.append(
                {
                    "obs": obs,
                    "acts": acts,
                    "rews": rews[:t],
                    "infos": tr.get("infos", None),
                    "terminal": bool(tr.get("terminal", True)),
                }
            )
            total += t
        else:
            k = int(remaining)
            out.append(
                {
                    "obs": obs[: k + 1],
                    "acts": acts[:k],
                    "rews": rews[:k],
                    "infos": tr.get("infos", None),
                    "terminal": False,
                }
            )
            total += k
            break

    if total != target_steps:
        raise RuntimeError(f"Failed to trim exactly to {target_steps} steps (got {total}).")
    return out


def write_trajectory_chunks(
    *,
    trajectories: List[Dict[str, Any]],
    output_dir: str,
    chunk_size: int = 64,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    paths: List[str] = []
    for start in range(0, len(trajectories), chunk_size):
        chunk = trajectories[start : start + chunk_size]
        path = os.path.join(output_dir, f"chunk_{start // chunk_size:05d}.pkl")
        with open(path, "wb") as fh:
            pickle.dump(chunk, fh, protocol=pickle.HIGHEST_PROTOCOL)
        paths.append(path)
    return paths


def file_sha1(path: str) -> str:
    hasher = hashlib.sha1()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def write_manifest(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
