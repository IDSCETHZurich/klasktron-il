"""
Fine-tune a PPO policy with configurable rewards and initialization:
- `learner_init_mode="bootstrap"`: initialize Peg1 from BC/GAIL checkpoint
- `learner_init_mode="random"`: random Peg1 initialization (same architecture/hyperparams)
- `reward_mode="sparse"`: `goal_scored` = +goal_reward, `goal_conceded` = -goal_reward, all others 0
- `reward_mode="dense"`: keep dense default reward terms from EnvCfg (optional global scaling)

The learner outputs only Peg1 actions (2D). Peg2 is controlled by a fixed
opponent policy, initialized from the same bootstrap checkpoint. When Peg1
clearly dominates Peg2, the Peg2 policy is refreshed from the current Peg1
policy and frozen again.

Release-facing defaults now live under `configs/training/`.
"""

import argparse
import csv
import faulthandler
import json
import os
import sys
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from config_utils import (
    dataclass_from_mapping,
    dataclass_to_dict,
    default_config_path,
    merge_config_file,
    resolve_repo_path,
)


# =============================================================================
# Config
# =============================================================================


DEFAULT_CONFIG_PATH = default_config_path("sparse_rl_bootstrap.json")


@dataclass
class Config:
    # Bootstrapping policy (either BC or GAIL PPO zip)
    # Used as:
    # - Peg1 initialization source when learner_init_mode="bootstrap"
    # - Hyperparameter/architecture template for Peg1 model construction
    # - Default fixed-opponent source when opponent_init_mode="from_pretrained"
    # This older BC1 checkpoint is retained for sparse-RL provenance; the
    # release-facing BC baseline defaults point to HUMAN_DEMO_V2 instead.
    pretrained_policy_path: str = (
        "trained_models/klask_bc_20260123_142654_BC1/ppo_policy_bc_final.zip"
    )

    # Output
    save_root: str = "trained_models"
    run_tag: str = "GAIL_BC_sparse_RL"
    run_name: Optional[str] = None

    # WandB
    wandb_project: str = "klask_dense_rl"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"  # "online" | "offline" | "disabled"

    # Training
    total_timesteps: int = 50_000_000
    train_num_envs: int = 4096
    device: str = "cuda:0"
    seed: int = 42
    learner_init_mode: str = "random"  # "bootstrap" | "random"

    # Reward profile
    # - "sparse": zero-sum sparse (+goal_scored, -goal_conceded, all other terms 0)
    # - "dense": keep the environment's default dense reward terms from EnvCfg
    reward_mode: str = "sparse"  # "sparse" | "dense"
    goal_reward: float = 10.0  # used in sparse mode
    dense_reward_scale: float = 1.0  # optional multiplier for all dense reward terms

    # Evaluation
    eval_episodes: int = 4096
    eval_freq_timesteps: int = 1_500_000

    # Logging/checkpoint cadence
    rollout_log_freq_timesteps: int = 100_000
    checkpoint_freq_timesteps: int = 1_500_000
    recent_episode_window: int = 500

    # Optional env wrappers
    use_collision_avoidance: bool = False
    collision_max_vel: float = 0.2
    use_action_history: bool = False
    action_history_length: int = 4

    # PPO LR override for Peg1 learner (if None, uses bootstrap PPO LR)
    override_learning_rate: Optional[float] = 5e-5
    # PPO entropy coefficient override for Peg1 learner (if None, uses bootstrap PPO value)
    override_ent_coef: Optional[float] = 1e-3

    # Action distribution:
    # - When enabled, use tanh-squashed actions (SB3 squash_output=True).
    # - SB3 requires use_sde=True when squash_output=True.
    enable_action_squash: bool = True

    # Per-axis velocity command bound for the learner AND the frozen opponent, m/s.
    # 0.2 is the limit the physical robot enforces, so it is the deployment-faithful
    # default. Pass --action-limit 1.0 to reproduce the paper's Fig. 5 / Sec. 5.4.
    action_limit: float = 0.2

    # Opponent behavior (Peg2 fixed policy)
    deterministic_opponent: bool = True
    invert_opponent_actions: bool = True
    opponent_init_mode: str = "from_pretrained"  # "from_learner" | "from_pretrained"
    fixed_opponent_policy_path: Optional[str] = None  # defaults to pretrained_policy_path

    # Exploration diagnostics
    log_exploration_metrics: bool = True
    peg1_obs_x_index: int = 0
    peg1_obs_y_index: int = 1
    occupancy_bins: int = 20
    occupancy_x_range: Tuple[float, float] = (-0.25, 0.25)
    occupancy_y_range: Tuple[float, float] = (-0.25, 0.25)
    occupancy_sample_envs: int = 128

    # Opponent refresh (snapshot Peg1 -> Peg2 when Peg1 dominates)
    enable_opponent_refresh: bool = True
    opponent_refresh_eval_only: bool = True
    opponent_refresh_min_timesteps: int = 2_097_152
    opponent_refresh_min_interval_timesteps: int = 2_097_152
    opponent_refresh_peg1_ratio_high: float = 1.8
    opponent_refresh_peg2_ratio_low: float = 0.56
    opponent_refresh_min_goals_per_episode: float = 0.008

    # Debug logging for startup/hang diagnosis
    startup_debug: bool = False
    gym_make_stack_dump_interval_s: int = 30

    # Workaround for occasional hangs in articulation link-transform queries
    # during observation-manager initialization on some GPU setups.
    avoid_body_pos_queries_in_observations: bool = False

def build_arg_parser(defaults: Config) -> argparse.ArgumentParser:
    """Build the sparse-RL CLI from merged config defaults."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a sparse-RL Klask agent from a curated preset."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--print-config", action="store_true", help="Print resolved config and exit.")
    parser.add_argument("--pretrained-policy-path", type=str, default=defaults.pretrained_policy_path)
    parser.add_argument("--fixed-opponent-policy-path", type=str, default=defaults.fixed_opponent_policy_path)
    parser.add_argument("--save-root", type=str, default=defaults.save_root)
    parser.add_argument("--run-tag", type=str, default=defaults.run_tag)
    parser.add_argument("--run-name", type=str, default=defaults.run_name)
    parser.add_argument("--wandb-mode", type=str, default=defaults.wandb_mode)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--train-num-envs", type=int, default=defaults.train_num_envs)
    parser.add_argument("--total-timesteps", type=int, default=defaults.total_timesteps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--learner-init-mode",
        choices=("bootstrap", "random"),
        default=defaults.learner_init_mode,
    )
    parser.add_argument(
        "--opponent-init-mode",
        choices=("from_learner", "from_pretrained"),
        default=defaults.opponent_init_mode,
    )
    parser.add_argument(
        "--reward-mode",
        choices=("sparse", "dense"),
        default=defaults.reward_mode,
    )
    parser.add_argument("--goal-reward", type=float, default=defaults.goal_reward)
    parser.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes)
    parser.add_argument(
        "--action-limit",
        type=float,
        default=defaults.action_limit,
        help="Per-axis velocity command bound (m/s) for learner and frozen "
             "opponent. 0.2 = hardware limit (default); 1.0 = paper reproduction.",
    )
    parser.add_argument(
        "--enable-opponent-refresh",
        action=argparse.BooleanOptionalAction,
        default=defaults.enable_opponent_refresh,
    )
    parser.add_argument(
        "--startup-debug",
        action=argparse.BooleanOptionalAction,
        default=defaults.startup_debug,
    )
    return parser


def _resolve_cfg_paths(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    for key in ("pretrained_policy_path", "fixed_opponent_policy_path", "save_root"):
        out[key] = resolve_repo_path(out.get(key))
    return out


def build_config_from_cli() -> Config:
    """Merge config file defaults with explicit CLI overrides."""
    base_cfg = Config()
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args()

    merged = merge_config_file(
        dataclass_to_dict(base_cfg),
        pre_args.config,
        label="sparse RL",
    )
    merged = _resolve_cfg_paths({k: v for k, v in merged.items() if not k.startswith("_")})
    config_defaults = dataclass_from_mapping(base_cfg, merged, label="sparse RL config")

    args = build_arg_parser(config_defaults).parse_args()
    cli_overrides = {
        "pretrained_policy_path": resolve_repo_path(args.pretrained_policy_path),
        "fixed_opponent_policy_path": resolve_repo_path(args.fixed_opponent_policy_path),
        "save_root": resolve_repo_path(args.save_root),
        "run_tag": args.run_tag,
        "run_name": args.run_name,
        "wandb_mode": args.wandb_mode,
        "device": args.device,
        "train_num_envs": int(args.train_num_envs),
        "total_timesteps": int(args.total_timesteps),
        "seed": int(args.seed),
        "learner_init_mode": args.learner_init_mode,
        "opponent_init_mode": args.opponent_init_mode,
        "reward_mode": args.reward_mode,
        "goal_reward": float(args.goal_reward),
        "eval_episodes": int(args.eval_episodes),
        "action_limit": float(args.action_limit),
        "enable_opponent_refresh": bool(args.enable_opponent_refresh),
        "startup_debug": bool(args.startup_debug),
    }
    cfg = dataclass_from_mapping(config_defaults, cli_overrides, label="sparse RL CLI")
    if args.print_config:
        print(json.dumps(dataclass_to_dict(cfg), indent=2))
        raise SystemExit(0)
    return cfg


CFG = build_config_from_cli()


# Launch Omniverse in headless mode for training
from isaaclab.app import AppLauncher

# The device MUST be passed: without it Isaac Sim places the simulation context on
# cuda:0 regardless of --device, so parallel runs all pile onto card 0.
app = AppLauncher(headless=True, device=str(CFG.device)).app

import gymnasium as gym
import isaaclab.envs.mdp as mdp
import numpy as np
import torch
import wandb
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnvWrapper

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab.managers import SceneEntityCfg
from my_isaaclab_tasks.manager_based.klask.actuator_model import ActuatorModelWrapper
from my_isaaclab_tasks.manager_based.klask.env_cfg import EnvCfg
from my_isaaclab_tasks.manager_based.klask.env_wrapper import (
    ActionHistoryWrapper,
    KlaskCollisionAvoidanceWrapper,
    RewardInfoWrapper,
)


# =============================================================================
# Helpers
# =============================================================================


def _unwrap_obs(obs: Any) -> Any:
    if isinstance(obs, tuple):
        return obs[0]
    return obs


def _scalar_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        if value.numel() == 1:
            return bool(value.detach().item())
        return bool(torch.any(value.detach()).item())
    arr = np.asarray(value)
    if arr.size == 0:
        return False
    if arr.size == 1:
        return bool(arr.reshape(-1)[0])
    return bool(np.any(arr))


def resolve_device(requested_device: str) -> str:
    """
    Resolve a requested torch device string against visible CUDA devices.
    Important when CUDA_VISIBLE_DEVICES remaps indices inside the process.
    """
    req = str(requested_device).strip().lower()
    if req == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda:0"
        return "cpu"

    if not req.startswith("cuda"):
        return req

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        print(f"[warn] Requested device '{requested_device}' but CUDA not available. Falling back to CPU.")
        return "cpu"

    # cuda or cuda:<idx>
    idx = 0
    if ":" in req:
        try:
            idx = int(req.split(":", 1)[1])
        except ValueError:
            idx = 0

    n_visible = torch.cuda.device_count()
    if idx < 0 or idx >= n_visible:
        print(
            f"[warn] Requested '{requested_device}', but only {n_visible} CUDA device(s) visible "
            f"inside this process (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}). "
            "Using cuda:0."
        )
        idx = 0
    return f"cuda:{idx}"


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dbg(cfg: Config, msg: str) -> None:
    if not cfg.startup_debug:
        return
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[debug {ts}] {msg}", flush=True)


_REWARD_TERM_NAMES = [
    "time_punishment",
    "time_out_punishment",
    "shot_over_middle_line",
    "player_in_goal",
    "opponent_in_goal",
    "goal_scored",
    "goal_conceded",
    "distance_player_ball_own_half",
    "distance_ball_opponent_goal",
    "distance_ball_own_goal",
    "ball_speed",
    "ball_stationary",
    "collision_player_ball",
    "ball_in_own_half",
    "close_to_boundaries",
    "player_strategically_positioned",
]


def _reward_weights_dict(rew: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in _REWARD_TERM_NAMES:
        term = getattr(rew, name, None)
        if term is not None and hasattr(term, "weight"):
            out[name] = float(term.weight)
    return out


def apply_reward_profile(env_cfg: EnvCfg, cfg: Config) -> Dict[str, float]:
    """
    Apply reward profile and return resulting reward weights for logging.
    """
    rew = env_cfg.rewards
    mode = str(cfg.reward_mode).strip().lower()

    if mode == "sparse":
        # Zero-sum sparse: terminal signal only.
        for name in _REWARD_TERM_NAMES:
            term = getattr(rew, name, None)
            if term is not None and hasattr(term, "weight"):
                term.weight = 0.0
        if hasattr(rew, "goal_scored"):
            rew.goal_scored.weight = float(cfg.goal_reward)
        if hasattr(rew, "goal_conceded"):
            rew.goal_conceded.weight = -float(cfg.goal_reward)
        return _reward_weights_dict(rew)

    if mode == "dense":
        # Dense: keep EnvCfg defaults, optionally scale all active terms.
        if float(cfg.dense_reward_scale) != 1.0:
            scale = float(cfg.dense_reward_scale)
            for name in _REWARD_TERM_NAMES:
                term = getattr(rew, name, None)
                if term is not None and hasattr(term, "weight"):
                    term.weight = float(term.weight) * scale
        return _reward_weights_dict(rew)

    raise ValueError(
        f"Unsupported reward_mode='{cfg.reward_mode}'. Expected 'sparse' or 'dense'."
    )


def _joint_xy_pos_obs(
    env,
    x_joint_cfg: SceneEntityCfg,
    y_joint_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Peg XY from joint positions (avoids body/link transform tensor queries)."""
    x = mdp.joint_pos_rel(env, x_joint_cfg)
    y = mdp.joint_pos_rel(env, y_joint_cfg)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    return torch.cat([x, y], dim=1)


def _ball_state_root_obs(env, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Ball state = [x, y, vx, vy] from rigid-body root data.
    Avoids KalmanFilter's dependency on peg body positions.
    """
    ball = env.scene[ball_cfg.name]
    pos_xy = ball.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    vel_xy = ball.data.root_lin_vel_w[:, :2]
    return torch.cat([pos_xy, vel_xy], dim=1)


def apply_observation_hang_workaround(cfg: Config, env_cfg: EnvCfg) -> None:
    if not cfg.avoid_body_pos_queries_in_observations:
        return

    _dbg(cfg, "applying observation workaround: avoid body_pos_w/link transforms")

    # Policy observation group
    env_cfg.observations.policy.peg_1_pos.func = _joint_xy_pos_obs
    env_cfg.observations.policy.peg_1_pos.params = {
        "x_joint_cfg": SceneEntityCfg(name="klask", joint_names=["slider_to_peg_1"]),
        "y_joint_cfg": SceneEntityCfg(name="klask", joint_names=["ground_to_slider_1"]),
    }
    env_cfg.observations.policy.peg_2_pos.func = _joint_xy_pos_obs
    env_cfg.observations.policy.peg_2_pos.params = {
        "x_joint_cfg": SceneEntityCfg(name="klask", joint_names=["slider_to_peg_2"]),
        "y_joint_cfg": SceneEntityCfg(name="klask", joint_names=["ground_to_slider_2"]),
    }
    env_cfg.observations.policy.ball_state.func = _ball_state_root_obs
    env_cfg.observations.policy.ball_state.params = {
        "ball_cfg": SceneEntityCfg(name="ball"),
    }

    # Opponent observation group
    env_cfg.observations.opponent.peg_1_pos.func = _joint_xy_pos_obs
    env_cfg.observations.opponent.peg_1_pos.params = {
        "x_joint_cfg": SceneEntityCfg(name="klask", joint_names=["slider_to_peg_1"]),
        "y_joint_cfg": SceneEntityCfg(name="klask", joint_names=["ground_to_slider_1"]),
    }
    env_cfg.observations.opponent.peg_2_pos.func = _joint_xy_pos_obs
    env_cfg.observations.opponent.peg_2_pos.params = {
        "x_joint_cfg": SceneEntityCfg(name="klask", joint_names=["slider_to_peg_2"]),
        "y_joint_cfg": SceneEntityCfg(name="klask", joint_names=["ground_to_slider_2"]),
    }
    env_cfg.observations.opponent.ball_state.func = _ball_state_root_obs
    env_cfg.observations.opponent.ball_state.params = {
        "ball_cfg": SceneEntityCfg(name="ball"),
    }


def make_env(cfg: Config, *, num_envs: int, seed: int, sim_device: str) -> Sb3VecEnvWrapper:
    _dbg(cfg, f"make_env start: num_envs={num_envs}, seed={seed}, sim_device={sim_device}")
    _dbg(cfg, "constructing EnvCfg()")
    env_cfg = EnvCfg()
    _dbg(cfg, "setting env_cfg scene/seed/device")
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.sim.device = sim_device
    _dbg(cfg, f"applying reward profile: mode={cfg.reward_mode}")
    reward_spec = apply_reward_profile(env_cfg, cfg)
    setattr(env_cfg, "_reward_spec", reward_spec)
    apply_observation_hang_workaround(cfg, env_cfg)

    _dbg(cfg, "calling gym.make('Isaac-Klask-v0')")
    dump_enabled = bool(cfg.startup_debug and cfg.gym_make_stack_dump_interval_s > 0)
    if dump_enabled:
        try:
            faulthandler.enable(file=sys.stderr, all_threads=True)
            faulthandler.dump_traceback_later(
                timeout=cfg.gym_make_stack_dump_interval_s,
                repeat=True,
                file=sys.stderr,
            )
            _dbg(
                cfg,
                f"enabled periodic stack dumps every {cfg.gym_make_stack_dump_interval_s}s while inside gym.make",
            )
        except Exception as exc:
            _dbg(cfg, f"could not enable faulthandler stack dumps: {exc}")
            dump_enabled = False

    try:
        base_env = gym.make("Isaac-Klask-v0", cfg=env_cfg)
    finally:
        if dump_enabled:
            try:
                faulthandler.cancel_dump_traceback_later()
                _dbg(cfg, "disabled periodic gym.make stack dumps")
            except Exception as exc:
                _dbg(cfg, f"could not disable faulthandler stack dumps cleanly: {exc}")

    _dbg(cfg, "gym.make returned")

    _dbg(cfg, "wrapping with ActuatorModelWrapper")
    env = ActuatorModelWrapper(base_env, device=sim_device)
    _dbg(cfg, "ActuatorModelWrapper ready")

    if cfg.use_collision_avoidance:
        _dbg(cfg, f"wrapping with KlaskCollisionAvoidanceWrapper(max_vel={cfg.collision_max_vel})")
        env = KlaskCollisionAvoidanceWrapper(env, max_vel=cfg.collision_max_vel)
        _dbg(cfg, "KlaskCollisionAvoidanceWrapper ready")
    _dbg(cfg, "wrapping with RewardInfoWrapper")
    env = RewardInfoWrapper(env)
    _dbg(cfg, "RewardInfoWrapper ready")

    _dbg(cfg, "wrapping with Sb3VecEnvWrapper")
    venv = Sb3VecEnvWrapper(env)
    _dbg(cfg, "Sb3VecEnvWrapper ready")
    if cfg.use_action_history:
        _dbg(cfg, f"wrapping with ActionHistoryWrapper(length={cfg.action_history_length})")
        venv = ActionHistoryWrapper(venv, history_length=cfg.action_history_length)
        _dbg(cfg, "ActionHistoryWrapper ready")

    # Keep a handle for explicit cleanup in case wrappers hide the base env.
    setattr(venv, "_base_env", base_env)
    setattr(venv, "_reward_spec", reward_spec)
    _dbg(cfg, "make_env complete")
    return venv


def _copy_policy_weights_with_action_slice(
    source_policy: ActorCriticPolicy,
    target_policy: ActorCriticPolicy,
    *,
    source_action_slice: slice,
    skip_log_std_mismatch: bool = False,
    action_head_scale: float = 1.0,
) -> None:
    """Copy shared policy weights and slice action heads from source->target.

    `action_head_scale` rescales the copied action head. The BC source emits raw
    m/s, while a squashed learner emits `action_limit * tanh(h)`, so the head must
    be divided by the limit to reproduce the same command. At the paper's limit of
    1.0 the scale is 1.0 and the copy is bit-identical to the original behaviour.
    """
    src_state = source_policy.state_dict()
    dst_state = target_policy.state_dict()

    for key, dst_tensor in dst_state.items():
        src_tensor = src_state.get(key, None)
        if src_tensor is None:
            continue

        if src_tensor.shape == dst_tensor.shape:
            dst_state[key] = src_tensor.detach().clone()
            continue

        if key == "action_net.weight":
            sliced = src_tensor[source_action_slice, :]
            if sliced.shape != dst_tensor.shape:
                raise ValueError(
                    f"Cannot slice '{key}' from {tuple(src_tensor.shape)} to {tuple(dst_tensor.shape)}"
                )
            dst_state[key] = sliced.detach().clone() * float(action_head_scale)
            continue

        if key == "action_net.bias":
            sliced = src_tensor[source_action_slice]
            if sliced.shape != dst_tensor.shape:
                raise ValueError(
                    f"Cannot slice '{key}' from {tuple(src_tensor.shape)} to {tuple(dst_tensor.shape)}"
                )
            dst_state[key] = sliced.detach().clone() * float(action_head_scale)
            continue

        if key == "log_std":
            sliced = None
            if src_tensor.ndim == 1:
                candidate = src_tensor[source_action_slice]
                if candidate.shape == dst_tensor.shape:
                    sliced = candidate
            elif src_tensor.ndim == 2:
                candidate_row = src_tensor[source_action_slice, :]
                if candidate_row.shape == dst_tensor.shape:
                    sliced = candidate_row
                candidate_col = src_tensor[:, source_action_slice]
                if candidate_col.shape == dst_tensor.shape:
                    sliced = candidate_col
            if sliced is None:
                if skip_log_std_mismatch:
                    continue
                raise ValueError(f"Cannot slice '{key}' from {tuple(src_tensor.shape)} to {tuple(dst_tensor.shape)}")
            if sliced.shape != dst_tensor.shape:
                if skip_log_std_mismatch:
                    continue
                raise ValueError(f"Cannot slice '{key}' from {tuple(src_tensor.shape)} to {tuple(dst_tensor.shape)}")
            dst_state[key] = sliced.detach().clone()

    target_policy.load_state_dict(dst_state, strict=False)


def build_peg1_model_from_bootstrap(
    cfg: Config,
    *,
    bootstrap_model: PPO,
    train_env: VecEnvWrapper,
    device: str,
    initialize_from_bootstrap_weights: bool = True,
) -> PPO:
    """
    Create a 2D-action PPO learner from a template PPO.
    Optionally initialize learner weights from template Peg1 action slice.
    """
    learning_rate = (
        float(cfg.override_learning_rate)
        if cfg.override_learning_rate is not None
        else bootstrap_model.learning_rate
    )
    ent_coef = (
        float(cfg.override_ent_coef)
        if cfg.override_ent_coef is not None
        else bootstrap_model.ent_coef
    )
    use_sde = bool(bootstrap_model.use_sde)
    policy_kwargs = dict(bootstrap_model.policy_kwargs or {})
    if cfg.enable_action_squash:
        # SB3 enforces: squash_output=True requires use_sde=True.
        use_sde = True
        policy_kwargs["squash_output"] = True

    learner = PPO(
        policy=bootstrap_model.policy.__class__,
        env=train_env,
        learning_rate=learning_rate,
        n_steps=bootstrap_model.n_steps,
        batch_size=bootstrap_model.batch_size,
        n_epochs=bootstrap_model.n_epochs,
        gamma=bootstrap_model.gamma,
        gae_lambda=bootstrap_model.gae_lambda,
        clip_range=bootstrap_model.clip_range,
        clip_range_vf=bootstrap_model.clip_range_vf,
        normalize_advantage=bootstrap_model.normalize_advantage,
        ent_coef=ent_coef,
        vf_coef=bootstrap_model.vf_coef,
        max_grad_norm=bootstrap_model.max_grad_norm,
        use_sde=use_sde,
        sde_sample_freq=bootstrap_model.sde_sample_freq,
        rollout_buffer_class=bootstrap_model.rollout_buffer_class,
        rollout_buffer_kwargs=bootstrap_model.rollout_buffer_kwargs,
        target_kl=bootstrap_model.target_kl,
        stats_window_size=bootstrap_model._stats_window_size,
        tensorboard_log=bootstrap_model.tensorboard_log,
        policy_kwargs=policy_kwargs,
        verbose=bootstrap_model.verbose,
        seed=cfg.seed,
        device=device,
    )

    if initialize_from_bootstrap_weights:
        skip_log_std_mismatch = bool(cfg.enable_action_squash and not bootstrap_model.use_sde)
        action_head_scale = 1.0
        if cfg.enable_action_squash and float(cfg.action_limit) > 0.0:
            action_head_scale = 1.0 / float(cfg.action_limit)
        _copy_policy_weights_with_action_slice(
            source_policy=bootstrap_model.policy,
            target_policy=learner.policy,
            source_action_slice=slice(0, 2),
            skip_log_std_mismatch=skip_log_std_mismatch,
            action_head_scale=action_head_scale,
        )
        print(
            f"[setup] bootstrap action head scaled by {action_head_scale:.3g} "
            f"for action_limit={float(cfg.action_limit):.3g} m/s.",
            flush=True,
        )
        if skip_log_std_mismatch:
            print(
                "[setup] action squash enabled with gSDE: keeping learner log_std initialization "
                "(bootstrap log_std shape mismatch expected).",
                flush=True,
            )
    return learner


def build_peg1_policy_from_pretrained_source(
    cfg: Config,
    *,
    source_model: PPO,
    train_env: VecEnvWrapper,
    device: str,
) -> ActorCriticPolicy:
    """
    Build a Peg1-compatible (2D action) policy from an arbitrary source PPO checkpoint.
    Uses the same action-head slicing path as learner bootstrap initialization.
    """
    tmp_model = build_peg1_model_from_bootstrap(
        cfg,
        bootstrap_model=source_model,
        train_env=train_env,
        device=device,
        initialize_from_bootstrap_weights=True,
    )
    fixed_policy = clone_policy_from_model(tmp_model)
    del tmp_model
    return fixed_policy


def clone_policy_from_model(model: PPO) -> ActorCriticPolicy:
    """
    Clone an SB3 policy without Python deepcopy.
    This avoids torch deepcopy issues for non-leaf tensors.
    """
    policy_cls = model.policy.__class__
    policy_kwargs = dict(model.policy_kwargs or {})
    cloned_policy = policy_cls(
        observation_space=model.observation_space,
        action_space=model.action_space,
        lr_schedule=model.lr_schedule,
        use_sde=model.use_sde,
        **policy_kwargs,
    )
    cloned_policy.load_state_dict(model.policy.state_dict(), strict=True)
    cloned_policy.to(next(model.policy.parameters()).device)
    cloned_policy.set_training_mode(False)
    return cloned_policy


class OpponentVecEnvWrapper(VecEnvWrapper):
    """
    Expose only Peg1 actions (2D) to the learner.
    Peg2 actions are produced by a frozen opponent policy.
    """

    def __init__(
        self,
        venv: VecEnvWrapper,
        opponent_policy: Any,
        *,
        deterministic_opponent: bool = True,
        invert_opponent_actions: bool = True,
        action_limit: float = 1.0,
    ):
        # The base env is unbounded, so this Box *is* the policy's action bound. With
        # squash_output=True SB3 emits `action_limit * tanh(h)`, so changing the limit
        # rescales the learner's commands directly rather than clipping them.
        self.action_limit = float(action_limit)
        action_space = spaces.Box(low=-self.action_limit, high=self.action_limit,
                                  shape=(2,), dtype=np.float32)
        super().__init__(venv, action_space=action_space)
        self.opponent_policy = opponent_policy
        self.deterministic_opponent = deterministic_opponent
        self.invert_opponent_actions = invert_opponent_actions
        self._last_obs = None

    def reset(self, **kwargs):
        obs = self.venv.reset(**kwargs)
        self._last_obs = obs
        return obs

    def step_async(self, actions):
        if self._last_obs is None:
            raise RuntimeError("OpponentVecEnvWrapper.step_async called before reset().")

        player_actions = np.asarray(actions, dtype=np.float32)
        if player_actions.ndim == 1:
            player_actions = player_actions.reshape(1, -1)
        if player_actions.shape[1] != 2:
            raise ValueError(f"Expected Peg1 actions with shape (*,2), got {player_actions.shape}.")

        obs_for_opp = self._to_opponent_obs(self._last_obs)
        opp_actions, _ = self.opponent_policy.predict(
            obs_for_opp,
            deterministic=self.deterministic_opponent,
        )
        opp_actions = np.asarray(opp_actions, dtype=np.float32)
        if opp_actions.ndim == 1:
            opp_actions = opp_actions.reshape(1, -1)
        if opp_actions.shape[1] != 2:
            raise ValueError(
                f"Opponent policy must output 2 actions for Peg2, got {opp_actions.shape}."
            )

        if self.invert_opponent_actions:
            opp_actions = -opp_actions
        # Clip the frozen opponent to the learner's bound, so both sides of the
        # match respect the limit the hardware enforces.
        opp_actions = np.clip(opp_actions, -self.action_limit, self.action_limit)

        full_actions = np.concatenate([player_actions, opp_actions], axis=1)
        self.venv.step_async(full_actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self._last_obs = obs
        return obs, rewards, dones, infos

    def update_opponent_from_policy(self, learner_policy: ActorCriticPolicy) -> None:
        """Hard-update frozen Peg2 from current Peg1 learner weights."""
        self.opponent_policy.load_state_dict(learner_policy.state_dict(), strict=True)
        self.opponent_policy.to(next(learner_policy.parameters()).device)
        self.opponent_policy.set_training_mode(False)

    @staticmethod
    def _to_opponent_obs(obs: np.ndarray) -> np.ndarray:
        """
        Remap Peg1/Peg2 fields so Peg2 can act from its own perspective.
        For the standard 12D obs this swaps [0:8] as [4:8,0:4].
        """
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)
        if obs_arr.shape[1] < 12:
            raise ValueError(f"Opponent obs remap expects >=12 dims, got {obs_arr.shape[1]}.")
        remapped = np.array(obs_arr, copy=True)
        remapped[:, :8] = obs_arr[:, [4, 5, 6, 7, 0, 1, 2, 3]]
        return remapped


class ZeroOpponentPolicy:
    """Placeholder Peg2 policy used only during initialization."""

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        obs_arr = np.asarray(obs)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)
        actions = np.zeros((obs_arr.shape[0], 2), dtype=np.float32)
        return actions, None


def close_env(venv: Optional[VecEnvWrapper]) -> None:
    if venv is None:
        return
    try:
        venv.close()
    except Exception as exc:
        print(f"[warn] Failed to close vec env cleanly: {exc}")
    base_env = getattr(venv, "_base_env", None)
    if base_env is not None:
        try:
            base_env.close()
        except Exception:
            pass


def _default_run_name(cfg: Config, timestamp: str) -> str:
    """Build a mode-aware run name so dense/sparse runs are easy to distinguish."""
    mode = str(cfg.reward_mode).strip().lower()
    mode_token = "dense" if mode == "dense" else "sparse"

    run_tag = str(cfg.run_tag)
    if mode_token == "dense":
        run_tag = run_tag.replace("sparse", "dense").replace("SPARSE", "DENSE")
    else:
        run_tag = run_tag.replace("dense", "sparse").replace("DENSE", "SPARSE")

    return f"klask_{mode_token}_rl_{timestamp}_{run_tag}"


def evaluate_policy_metrics(model: PPO, venv: VecEnvWrapper, n_episodes: int) -> Dict[str, float]:
    """Deterministic rollout evaluation with goal statistics."""
    obs = _unwrap_obs(venv.reset())
    n_envs = venv.num_envs

    ep_reward = np.zeros(n_envs, dtype=np.float64)
    ep_len = np.zeros(n_envs, dtype=np.int64)
    ep_goals = np.zeros(n_envs, dtype=np.int64)
    ep_conceded = np.zeros(n_envs, dtype=np.int64)

    all_rewards = []
    all_lengths = []
    all_goals = []
    all_conceded = []

    completed = 0
    while completed < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = venv.step(action)
        obs = _unwrap_obs(obs)

        reward = np.asarray(reward, dtype=np.float32)
        done = np.asarray(done, dtype=bool)
        infos = infos if isinstance(infos, (list, tuple)) else [infos] * n_envs

        ep_reward += reward
        ep_len += 1

        for i, info in enumerate(infos):
            if isinstance(info, dict):
                ep_goals[i] += int(_scalar_bool(info.get("goal_scored")))
                ep_conceded[i] += int(_scalar_bool(info.get("goal_conceded")))

        for i, is_done in enumerate(done):
            if not is_done:
                continue
            all_rewards.append(float(ep_reward[i]))
            all_lengths.append(float(ep_len[i]))
            all_goals.append(float(ep_goals[i]))
            all_conceded.append(float(ep_conceded[i]))
            ep_reward[i] = 0.0
            ep_len[i] = 0
            ep_goals[i] = 0
            ep_conceded[i] = 0
            completed += 1
            if completed >= n_episodes:
                break

    goals_per_ep = float(np.mean(all_goals)) if all_goals else 0.0
    conceded_per_ep = float(np.mean(all_conceded)) if all_conceded else 0.0
    eps = 1e-6
    peg1_ratio = float((goals_per_ep + eps) / (conceded_per_ep + eps))
    peg2_ratio = float((conceded_per_ep + eps) / (goals_per_ep + eps))
    return {
        "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "mean_episode_length": float(np.mean(all_lengths)) if all_lengths else 0.0,
        "goals_per_episode": goals_per_ep,
        "goals_conceded_per_episode": conceded_per_ep,
        "goal_diff_per_episode": goals_per_ep - conceded_per_ep,
        "peg1_goal_ratio": peg1_ratio,
        "peg2_goal_ratio": peg2_ratio,
    }


# =============================================================================
# Callback
# =============================================================================


class SparseRLTrainingCallback(BaseCallback):
    def __init__(
        self,
        cfg: Config,
        run_dir: Path,
        eval_env: VecEnvWrapper,
        opponent_env: OpponentVecEnvWrapper,
        baseline_metrics: Dict[str, float],
        wandb_run: Optional[Any],
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.cfg = cfg
        self.run_dir = run_dir
        self.eval_env = eval_env
        self.opponent_env = opponent_env
        self.baseline = baseline_metrics
        self.wandb_run = wandb_run

        self.best_peg1_goal_ratio = baseline_metrics["peg1_goal_ratio"]
        self.last_rollout_log_step = 0
        self.last_eval_step = 0
        self.last_ckpt_step = 0
        self.last_opponent_refresh_step = 0
        self.opponent_update_count = 0

        self.recent_rewards = deque(maxlen=cfg.recent_episode_window)
        self.recent_lengths = deque(maxlen=cfg.recent_episode_window)
        self.recent_goals = deque(maxlen=cfg.recent_episode_window)
        self.recent_conceded = deque(maxlen=cfg.recent_episode_window)

        self.ep_reward = None
        self.ep_len = None
        self.ep_goals = None
        self.ep_conceded = None
        self.action_count = 0
        self.action_sum = np.zeros(2, dtype=np.float64)
        self.action_sumsq = np.zeros(2, dtype=np.float64)
        bins = max(2, int(cfg.occupancy_bins))
        self.occupancy_hist = np.zeros((bins, bins), dtype=np.int64)
        self.occupancy_total = 0
        self.rollout_window_steps = 0
        self.rollout_term_total = 0
        self.rollout_term_goal_scored = 0
        self.rollout_term_goal_conceded = 0
        self.rollout_term_player_in_goal = 0
        self.rollout_term_opponent_in_goal = 0
        self.rollout_term_time_out = 0
        self.rollout_term_unknown = 0

        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.opponent_refresh_dir = self.run_dir / "opponent_refresh"
        self.opponent_refresh_dir.mkdir(parents=True, exist_ok=True)

        self.eval_csv_file = open(self.run_dir / "eval_metrics.csv", "w", newline="")
        self.eval_writer = csv.DictWriter(
            self.eval_csv_file,
            fieldnames=[
                "timestep",
                "mean_reward",
                "mean_episode_length",
                "goals_per_episode",
                "goals_conceded_per_episode",
                "goal_diff_per_episode",
                "peg1_goal_ratio",
                "peg2_goal_ratio",
                "delta_goals_vs_bootstrap",
                "delta_reward_vs_bootstrap",
                "opponent_update_count",
            ],
        )
        self.eval_writer.writeheader()

    def _reset_exploration_buffers(self) -> None:
        self.action_count = 0
        self.action_sum[:] = 0.0
        self.action_sumsq[:] = 0.0
        self.occupancy_hist.fill(0)
        self.occupancy_total = 0

    def _reset_termination_counters(self) -> None:
        self.rollout_window_steps = 0
        self.rollout_term_total = 0
        self.rollout_term_goal_scored = 0
        self.rollout_term_goal_conceded = 0
        self.rollout_term_player_in_goal = 0
        self.rollout_term_opponent_in_goal = 0
        self.rollout_term_time_out = 0
        self.rollout_term_unknown = 0

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self.ep_reward = np.zeros(n_envs, dtype=np.float64)
        self.ep_len = np.zeros(n_envs, dtype=np.int64)
        self.ep_goals = np.zeros(n_envs, dtype=np.int64)
        self.ep_conceded = np.zeros(n_envs, dtype=np.int64)
        self._reset_exploration_buffers()
        self._reset_termination_counters()

    def _log_rollout_window(self) -> None:
        if len(self.recent_rewards) == 0:
            return
        mean_goals = float(np.mean(self.recent_goals))
        mean_conceded = float(np.mean(self.recent_conceded))
        eps = 1e-6
        payload = {
            "rollout/mean_reward": float(np.mean(self.recent_rewards)),
            "rollout/mean_episode_length": float(np.mean(self.recent_lengths)),
            "rollout/goals_per_episode": mean_goals,
            "rollout/goals_conceded_per_episode": mean_conceded,
            "rollout/goal_diff_per_episode": float(mean_goals - mean_conceded),
            "rollout/peg1_goal_ratio": float((mean_goals + eps) / (mean_conceded + eps)),
            "rollout/peg2_goal_ratio": float((mean_conceded + eps) / (mean_goals + eps)),
        }
        if self.rollout_window_steps > 0:
            per_step = float(self.rollout_window_steps)
            payload.update(
                {
                    "rollout/terminations/total_per_step": float(self.rollout_term_total / per_step),
                    "rollout/terminations/goal_scored_per_step": float(
                        self.rollout_term_goal_scored / per_step
                    ),
                    "rollout/terminations/goal_conceded_per_step": float(
                        self.rollout_term_goal_conceded / per_step
                    ),
                    "rollout/terminations/player_in_goal_per_step": float(
                        self.rollout_term_player_in_goal / per_step
                    ),
                    "rollout/terminations/opponent_in_goal_per_step": float(
                        self.rollout_term_opponent_in_goal / per_step
                    ),
                    "rollout/terminations/time_out_per_step": float(
                        self.rollout_term_time_out / per_step
                    ),
                    "rollout/terminations/unknown_per_step": float(
                        self.rollout_term_unknown / per_step
                    ),
                }
            )
        if self.rollout_term_total > 0:
            per_term = float(self.rollout_term_total)
            payload.update(
                {
                    "rollout/terminations/goal_scored_fraction": float(
                        self.rollout_term_goal_scored / per_term
                    ),
                    "rollout/terminations/goal_conceded_fraction": float(
                        self.rollout_term_goal_conceded / per_term
                    ),
                    "rollout/terminations/player_in_goal_fraction": float(
                        self.rollout_term_player_in_goal / per_term
                    ),
                    "rollout/terminations/opponent_in_goal_fraction": float(
                        self.rollout_term_opponent_in_goal / per_term
                    ),
                    "rollout/terminations/time_out_fraction": float(
                        self.rollout_term_time_out / per_term
                    ),
                    "rollout/terminations/unknown_fraction": float(
                        self.rollout_term_unknown / per_term
                    ),
                }
            )
        if self.cfg.log_exploration_metrics:
            if self.action_count > 0:
                mean_actions = self.action_sum / float(self.action_count)
                var_actions = self.action_sumsq / float(self.action_count) - mean_actions * mean_actions
                std_actions = np.sqrt(np.maximum(var_actions, 0.0))
                payload.update(
                    {
                        "rollout/action_mean_x": float(mean_actions[0]),
                        "rollout/action_mean_y": float(mean_actions[1]),
                        "rollout/action_std_x": float(std_actions[0]),
                        "rollout/action_std_y": float(std_actions[1]),
                    }
                )

            policy_log_std = getattr(self.model.policy, "log_std", None)
            if policy_log_std is not None:
                log_std = policy_log_std.detach().cpu().numpy().reshape(-1)
                action_std = np.exp(np.clip(log_std, -20.0, 5.0))
                payload["policy/action_std_mean"] = float(np.mean(action_std))
                if action_std.size > 0:
                    payload["policy/action_std_dim0"] = float(action_std[0])
                if action_std.size > 1:
                    payload["policy/action_std_dim1"] = float(action_std[1])
                # Differential entropy of diagonal Gaussian policy.
                entropy_per_dim = 0.5 * (1.0 + np.log(2.0 * np.pi)) + log_std
                payload["policy/diag_gaussian_entropy_sum"] = float(np.sum(entropy_per_dim))

            if self.occupancy_total > 0:
                counts = self.occupancy_hist.astype(np.float64).reshape(-1)
                total = float(np.sum(counts))
                if total > 0.0:
                    occupied = float(np.count_nonzero(counts))
                    total_bins = float(counts.size)
                    p = counts[counts > 0.0] / total
                    entropy = float(-np.sum(p * np.log(p)))
                    entropy_norm = float(entropy / np.log(total_bins)) if total_bins > 1.0 else 0.0
                    payload.update(
                        {
                            "rollout/peg1_occupancy_coverage": float(occupied / total_bins),
                            "rollout/peg1_occupancy_entropy": entropy_norm,
                            "rollout/peg1_occupancy_top_bin_fraction": float(np.max(counts) / total),
                        }
                    )

        if self.wandb_run is not None:
            self.wandb_run.log(payload, step=int(self.num_timesteps))
        if self.verbose > 0:
            print(
                f"[rollout @ {self.num_timesteps:,}] "
                f"reward={payload['rollout/mean_reward']:.3f}, "
                f"goals/ep={payload['rollout/goals_per_episode']:.3f}, "
                f"conceded/ep={payload['rollout/goals_conceded_per_episode']:.3f}, "
                f"peg1_ratio={payload['rollout/peg1_goal_ratio']:.2f}, "
                f"term_goal_conceded/step={payload.get('rollout/terminations/goal_conceded_per_step', 0.0):.3f}, "
                f"term_player_in_goal/step={payload.get('rollout/terminations/player_in_goal_per_step', 0.0):.3f}"
            )
        if self.cfg.log_exploration_metrics:
            self._reset_exploration_buffers()
        self._reset_termination_counters()

    def _maybe_refresh_opponent(self, metrics: Dict[str, float], source: str) -> None:
        if not self.cfg.enable_opponent_refresh:
            return
        if self.num_timesteps < self.cfg.opponent_refresh_min_timesteps:
            return
        if (
            self.last_opponent_refresh_step > 0
            and self.num_timesteps - self.last_opponent_refresh_step
            < self.cfg.opponent_refresh_min_interval_timesteps
        ):
            return

        peg1_ratio = float(metrics.get("peg1_goal_ratio", 1.0))
        peg2_ratio = float(metrics.get("peg2_goal_ratio", 1.0))
        goals_per_ep = float(metrics.get("goals_per_episode", 0.0))

        if peg1_ratio < self.cfg.opponent_refresh_peg1_ratio_high:
            return
        if peg2_ratio > self.cfg.opponent_refresh_peg2_ratio_low:
            return
        if goals_per_ep < self.cfg.opponent_refresh_min_goals_per_episode:
            return

        self.opponent_env.update_opponent_from_policy(self.model.policy)
        self.opponent_update_count += 1
        self.last_opponent_refresh_step = int(self.num_timesteps)

        refresh_ckpt_path = (
            self.opponent_refresh_dir
            / f"peg2_refreshed_from_peg1_{self.opponent_update_count}_{self.num_timesteps}.zip"
        )
        self.model.save(str(refresh_ckpt_path))

        payload = {
            "selfplay/opponent_update_count": float(self.opponent_update_count),
            "selfplay/opponent_last_update_step": float(self.last_opponent_refresh_step),
            "selfplay/opponent_update_source": 1.0 if source == "eval" else 0.0,
            "selfplay/trigger_peg1_goal_ratio": peg1_ratio,
            "selfplay/trigger_peg2_goal_ratio": peg2_ratio,
            "selfplay/trigger_goals_per_episode": goals_per_ep,
        }
        if self.wandb_run is not None:
            self.wandb_run.log(payload, step=int(self.num_timesteps))
        if self.verbose > 0:
            print(
                f"[self-play @ {self.num_timesteps:,}] refreshed Peg2 from Peg1 "
                f"(source={source}, peg1_ratio={peg1_ratio:.2f}, peg2_ratio={peg2_ratio:.2f})"
            )

    def _run_eval(self) -> None:
        metrics = evaluate_policy_metrics(self.model, self.eval_env, self.cfg.eval_episodes)
        delta_goals = metrics["goals_per_episode"] - self.baseline["goals_per_episode"]
        delta_reward = metrics["mean_reward"] - self.baseline["mean_reward"]

        eval_payload = {
            "eval/mean_reward": metrics["mean_reward"],
            "eval/mean_episode_length": metrics["mean_episode_length"],
            "eval/goals_per_episode": metrics["goals_per_episode"],
            "eval/goals_conceded_per_episode": metrics["goals_conceded_per_episode"],
            "eval/goal_diff_per_episode": metrics["goal_diff_per_episode"],
            "eval/peg1_goal_ratio": metrics["peg1_goal_ratio"],
            "eval/peg2_goal_ratio": metrics["peg2_goal_ratio"],
            "eval/delta_goals_vs_bootstrap": delta_goals,
            "eval/delta_reward_vs_bootstrap": delta_reward,
            "eval/best_peg1_goal_ratio": self.best_peg1_goal_ratio,
            "eval/opponent_update_count": self.opponent_update_count,
        }
        if self.wandb_run is not None:
            self.wandb_run.log(eval_payload, step=int(self.num_timesteps))

        row = {
            "timestep": int(self.num_timesteps),
            "mean_reward": float(metrics["mean_reward"]),
            "mean_episode_length": float(metrics["mean_episode_length"]),
            "goals_per_episode": float(metrics["goals_per_episode"]),
            "goals_conceded_per_episode": float(metrics["goals_conceded_per_episode"]),
            "goal_diff_per_episode": float(metrics["goal_diff_per_episode"]),
            "peg1_goal_ratio": float(metrics["peg1_goal_ratio"]),
            "peg2_goal_ratio": float(metrics["peg2_goal_ratio"]),
            "delta_goals_vs_bootstrap": float(delta_goals),
            "delta_reward_vs_bootstrap": float(delta_reward),
            "opponent_update_count": int(self.opponent_update_count),
        }
        self.eval_writer.writerow(row)
        self.eval_csv_file.flush()

        if self.verbose > 0:
            print(
                f"[eval @ {self.num_timesteps:,}] "
                f"goals/ep={metrics['goals_per_episode']:.3f}, "
                f"conceded/ep={metrics['goals_conceded_per_episode']:.3f}, "
                f"peg1_ratio={metrics['peg1_goal_ratio']:.2f}, "
                f"delta_goals={delta_goals:+.3f}, reward={metrics['mean_reward']:.3f}"
            )

        if metrics["peg1_goal_ratio"] > self.best_peg1_goal_ratio:
            self.best_peg1_goal_ratio = metrics["peg1_goal_ratio"]
            best_path = self.run_dir / "ppo_policy_sparse_rl_best_peg1_ratio.zip"
            self.model.save(str(best_path))
            if self.verbose > 0:
                print(
                    f"  Saved new best-peg1-ratio checkpoint: {best_path} "
                    f"(peg1_ratio={self.best_peg1_goal_ratio:.3f})"
                )

        if self.cfg.opponent_refresh_eval_only:
            self._maybe_refresh_opponent(metrics, source="eval")

        # Eval uses the same env as training. Resync SB3 state after eval reset/steps.
        self._resync_training_state_after_eval()

    def _save_periodic_checkpoint(self) -> None:
        ckpt_path = self.checkpoint_dir / f"ppo_sparse_rl_{self.num_timesteps}.zip"
        self.model.save(str(ckpt_path))
        if self.verbose > 0:
            print(f"[checkpoint] saved {ckpt_path}")

    def _resync_training_state_after_eval(self) -> None:
        try:
            obs = _unwrap_obs(self.training_env.reset())
            self.model._last_obs = obs
            self.model._last_episode_starts = np.ones(
                (self.training_env.num_envs,), dtype=bool
            )
        except Exception as exc:
            if self.verbose > 0:
                print(f"[warn] Failed to resync training env after eval: {exc}")

    def _on_step(self) -> bool:
        rewards_raw = self.locals.get("rewards", None)
        if rewards_raw is None:
            return True
        rewards = np.asarray(rewards_raw, dtype=np.float32)
        if rewards.size == 0:
            return True
        dones = np.asarray(self.locals.get("dones", np.zeros_like(rewards, dtype=bool)), dtype=bool)
        infos = self.locals.get("infos", [{} for _ in range(len(rewards))])
        infos = infos if isinstance(infos, (list, tuple)) else [infos] * len(rewards)
        self.rollout_window_steps += 1

        if self.cfg.log_exploration_metrics:
            actions_raw = self.locals.get("actions", None)
            if actions_raw is not None:
                actions = np.asarray(actions_raw, dtype=np.float32)
                if actions.ndim == 1:
                    actions = actions.reshape(1, -1)
                if actions.ndim == 2 and actions.shape[1] >= 2:
                    peg1_actions = actions[:, :2]
                    self.action_count += int(peg1_actions.shape[0])
                    self.action_sum += np.sum(peg1_actions, axis=0)
                    self.action_sumsq += np.sum(np.square(peg1_actions), axis=0)

            obs_raw = self.locals.get("new_obs", None)
            if obs_raw is not None:
                obs = np.asarray(_unwrap_obs(obs_raw), dtype=np.float32)
                if obs.ndim == 1:
                    obs = obs.reshape(1, -1)
                if obs.ndim == 2:
                    sample_n = min(obs.shape[0], max(1, int(self.cfg.occupancy_sample_envs)))
                    obs = obs[:sample_n]
                    x_idx = int(self.cfg.peg1_obs_x_index)
                    y_idx = int(self.cfg.peg1_obs_y_index)
                    max_idx = max(x_idx, y_idx)
                    if max_idx < obs.shape[1]:
                        x = obs[:, x_idx]
                        y = obs[:, y_idx]
                        finite = np.isfinite(x) & np.isfinite(y)
                        if np.any(finite):
                            x = x[finite]
                            y = y[finite]
                            bins = self.occupancy_hist.shape[0]
                            x_min, x_max = self.cfg.occupancy_x_range
                            y_min, y_max = self.cfg.occupancy_y_range
                            if x_max > x_min and y_max > y_min:
                                x_bin = np.floor((x - x_min) / (x_max - x_min) * bins).astype(np.int64)
                                y_bin = np.floor((y - y_min) / (y_max - y_min) * bins).astype(np.int64)
                                valid = (
                                    (x_bin >= 0)
                                    & (x_bin < bins)
                                    & (y_bin >= 0)
                                    & (y_bin < bins)
                                )
                                if np.any(valid):
                                    np.add.at(self.occupancy_hist, (y_bin[valid], x_bin[valid]), 1)
                                    self.occupancy_total += int(np.count_nonzero(valid))

        self.ep_reward += rewards
        self.ep_len += 1

        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            self.ep_goals[i] += int(_scalar_bool(info.get("goal_scored")))
            self.ep_conceded[i] += int(_scalar_bool(info.get("goal_conceded")))

        for i, is_done in enumerate(dones):
            if not is_done:
                continue
            info_i = infos[i] if i < len(infos) and isinstance(infos[i], dict) else {}
            term_goal_scored = _scalar_bool(info_i.get("goal_scored"))
            term_goal_conceded = _scalar_bool(info_i.get("goal_conceded"))
            term_player_in_goal = _scalar_bool(info_i.get("player_in_goal"))
            term_opponent_in_goal = _scalar_bool(info_i.get("opponent_in_goal"))
            term_time_out = _scalar_bool(info_i.get("time_out")) or _scalar_bool(
                info_i.get("TimeLimit.truncated")
            )
            if not (
                term_goal_scored
                or term_goal_conceded
                or term_player_in_goal
                or term_opponent_in_goal
                or term_time_out
            ):
                self.rollout_term_unknown += 1
            self.rollout_term_total += 1
            self.rollout_term_goal_scored += int(term_goal_scored)
            self.rollout_term_goal_conceded += int(term_goal_conceded)
            self.rollout_term_player_in_goal += int(term_player_in_goal)
            self.rollout_term_opponent_in_goal += int(term_opponent_in_goal)
            self.rollout_term_time_out += int(term_time_out)
            self.recent_rewards.append(float(self.ep_reward[i]))
            self.recent_lengths.append(float(self.ep_len[i]))
            self.recent_goals.append(float(self.ep_goals[i]))
            self.recent_conceded.append(float(self.ep_conceded[i]))
            self.ep_reward[i] = 0.0
            self.ep_len[i] = 0
            self.ep_goals[i] = 0
            self.ep_conceded[i] = 0

        if self.num_timesteps - self.last_rollout_log_step >= self.cfg.rollout_log_freq_timesteps:
            self._log_rollout_window()
            self.last_rollout_log_step = self.num_timesteps

        if not self.cfg.opponent_refresh_eval_only:
            mean_goals = float(np.mean(self.recent_goals)) if len(self.recent_goals) else 0.0
            mean_conceded = float(np.mean(self.recent_conceded)) if len(self.recent_conceded) else 0.0
            eps = 1e-6
            rollout_metrics = {
                "goals_per_episode": mean_goals,
                "goals_conceded_per_episode": mean_conceded,
                "peg1_goal_ratio": float((mean_goals + eps) / (mean_conceded + eps)),
                "peg2_goal_ratio": float((mean_conceded + eps) / (mean_goals + eps)),
            }
            self._maybe_refresh_opponent(rollout_metrics, source="rollout")

        return True

    def _on_rollout_end(self) -> None:
        # Run eval/checkpoint only between rollouts to avoid interfering with rollout collection.
        if self.num_timesteps - self.last_eval_step >= self.cfg.eval_freq_timesteps:
            self._run_eval()
            self.last_eval_step = self.num_timesteps

        if self.num_timesteps - self.last_ckpt_step >= self.cfg.checkpoint_freq_timesteps:
            self._save_periodic_checkpoint()
            self.last_ckpt_step = self.num_timesteps

    def _on_training_end(self) -> None:
        self._log_rollout_window()
        self.eval_csv_file.close()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    cfg = CFG
    _dbg(cfg, "main start")
    set_global_seeds(cfg.seed)
    _dbg(cfg, f"global seeds set to {cfg.seed}")
    resolved_device = resolve_device(cfg.device)
    _dbg(cfg, f"resolved device: requested={cfg.device} -> resolved={resolved_device}")

    learner_init_mode = str(cfg.learner_init_mode).strip().lower()
    if learner_init_mode not in ("bootstrap", "random"):
        raise ValueError(
            f"Unsupported learner_init_mode='{cfg.learner_init_mode}'. Expected 'bootstrap' or 'random'."
        )
    opponent_init_mode = str(cfg.opponent_init_mode).strip().lower()
    if opponent_init_mode not in ("from_learner", "from_pretrained"):
        raise ValueError(
            f"Unsupported opponent_init_mode='{cfg.opponent_init_mode}'. "
            "Expected 'from_learner' or 'from_pretrained'."
        )

    template_policy_path = Path(cfg.pretrained_policy_path)
    if not template_policy_path.exists():
        raise FileNotFoundError(f"Template policy not found: {template_policy_path}")
    opponent_policy_path = (
        Path(cfg.fixed_opponent_policy_path)
        if cfg.fixed_opponent_policy_path is not None
        else template_policy_path
    )
    if opponent_init_mode == "from_pretrained" and not opponent_policy_path.exists():
        raise FileNotFoundError(f"Fixed opponent policy not found: {opponent_policy_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name or _default_run_name(cfg, timestamp)
    run_dir = Path(cfg.save_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    shared_env = None
    train_env = None
    eval_env = None
    template_model = None
    wandb_run = None

    try:
        print(f"[setup] run_dir: {run_dir}", flush=True)
        print(
            f"[setup] requested device={cfg.device}, resolved device={resolved_device}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
            , flush=True
        )
        if cfg.use_action_history:
            raise ValueError(
                "This script expects base 12D observations for Peg2 remapping. "
                "Set cfg.use_action_history = False."
            )

        _dbg(cfg, "calling make_env")
        shared_env = make_env(
            cfg,
            num_envs=cfg.train_num_envs,
            seed=cfg.seed,
            sim_device=resolved_device,
        )
        _dbg(cfg, "make_env returned")

        _dbg(cfg, "loading PPO template policy (full action model)")
        print(f"[setup] loading template policy: {template_policy_path}", flush=True)
        template_model = PPO.load(str(template_policy_path), device=resolved_device)
        _dbg(cfg, "template PPO.load returned")
        print(
            f"[setup] learner_init_mode={learner_init_mode}, opponent_init_mode={opponent_init_mode}",
            flush=True,
        )

        _dbg(cfg, "wrapping env with OpponentVecEnvWrapper (Peg1 learns, Peg2 fixed)")
        train_env = OpponentVecEnvWrapper(
            shared_env,
            opponent_policy=ZeroOpponentPolicy(),
            deterministic_opponent=cfg.deterministic_opponent,
            invert_opponent_actions=cfg.invert_opponent_actions,
            action_limit=cfg.action_limit,
        )
        setattr(train_env, "_base_env", getattr(shared_env, "_base_env", None))
        eval_env = train_env
        print("[setup] using shared env for train+eval with Peg1-only learning", flush=True)

        _dbg(cfg, "building Peg1 learner from template")
        model = build_peg1_model_from_bootstrap(
            cfg,
            bootstrap_model=template_model,
            train_env=train_env,
            device=resolved_device,
            initialize_from_bootstrap_weights=(learner_init_mode == "bootstrap"),
        )
        if learner_init_mode == "bootstrap":
            _dbg(cfg, "Peg1 learner initialized from template Peg1 slice")
        else:
            _dbg(cfg, "Peg1 learner initialized randomly (template arch/hparams)")
            print("[setup] Peg1 learner uses random initialization", flush=True)
        if cfg.override_learning_rate is not None:
            print(f"[setup] learner learning rate overridden to {float(cfg.override_learning_rate)}", flush=True)
        if cfg.override_ent_coef is not None:
            print(f"[setup] learner ent_coef overridden to {float(cfg.override_ent_coef)}", flush=True)

        if opponent_init_mode == "from_pretrained":
            if opponent_policy_path == template_policy_path:
                opponent_source_model = template_model
            else:
                print(f"[setup] loading fixed opponent source policy: {opponent_policy_path}", flush=True)
                opponent_source_model = PPO.load(str(opponent_policy_path), device=resolved_device)
            fixed_opponent_policy = build_peg1_policy_from_pretrained_source(
                cfg,
                source_model=opponent_source_model,
                train_env=train_env,
                device=resolved_device,
            )
            print("[setup] fixed Peg2 opponent initialized from pretrained policy", flush=True)
        else:
            fixed_opponent_policy = clone_policy_from_model(model)
            print("[setup] fixed Peg2 opponent initialized from learner init weights", flush=True)
        train_env.opponent_policy = fixed_opponent_policy

        _dbg(cfg, f"running baseline eval for {cfg.eval_episodes} episodes")
        baseline_metrics = evaluate_policy_metrics(model, eval_env, cfg.eval_episodes)
        print(
            "[baseline] "
            f"goals/ep={baseline_metrics['goals_per_episode']:.3f}, "
            f"conceded/ep={baseline_metrics['goals_conceded_per_episode']:.3f}, "
            f"peg1_ratio={baseline_metrics['peg1_goal_ratio']:.2f}, "
            f"reward={baseline_metrics['mean_reward']:.3f}"
            , flush=True
        )

        reward_spec = getattr(shared_env, "_reward_spec", None)
        config_payload = {
            "config": asdict(cfg),
            "resolved_device": resolved_device,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "created_at": datetime.now().isoformat(),
            "baseline_metrics": baseline_metrics,
            "learner_init_mode": learner_init_mode,
            "opponent_init_mode": opponent_init_mode,
            "template_policy_path": str(template_policy_path),
            "opponent_policy_path": (
                str(opponent_policy_path) if opponent_init_mode == "from_pretrained" else None
            ),
            "reward_mode": cfg.reward_mode,
            "reward_spec": reward_spec,
            "control_spec": {
                "learner_controls": "peg1_only",
                "fixed_opponent_controls": "peg2_only",
                "learner_initialized_from": (
                    "template Peg1-sliced weights"
                    if learner_init_mode == "bootstrap"
                    else "random initialization (template architecture/hyperparameters)"
                ),
                "opponent_initialized_from": (
                    f"pretrained policy ({opponent_policy_path}) -> Peg2 (training remap/invert)"
                    if opponent_init_mode == "from_pretrained"
                    else "learner init policy -> Peg2"
                ),
            },
        }
        with open(run_dir / "config.json", "w") as f:
            json.dump(config_payload, f, indent=2)

        _dbg(cfg, "initializing wandb")
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=run_name,
            dir=str(run_dir),
            mode=cfg.wandb_mode,
            config=config_payload,
        )
        wandb_run.log(
            {f"baseline/{k}": float(v) for k, v in baseline_metrics.items()},
            step=0,
        )

        callback = SparseRLTrainingCallback(
            cfg=cfg,
            run_dir=run_dir,
            eval_env=eval_env,
            opponent_env=train_env,
            baseline_metrics=baseline_metrics,
            wandb_run=wandb_run,
            verbose=1,
        )

        print(f"[train] starting PPO fine-tuning for {cfg.total_timesteps:,} timesteps", flush=True)
        model.learn(total_timesteps=cfg.total_timesteps, callback=callback, progress_bar=True)
        _dbg(cfg, "model.learn returned")

        final_policy_path = run_dir / "ppo_policy_sparse_rl_final.zip"
        model.save(str(final_policy_path))

        final_metrics = evaluate_policy_metrics(model, eval_env, cfg.eval_episodes)
        summary = {
            "run_name": run_name,
            "learner_init_mode": learner_init_mode,
            "opponent_init_mode": opponent_init_mode,
            "template_policy_path": str(template_policy_path),
            "bootstrap_policy_path": (
                str(template_policy_path) if learner_init_mode == "bootstrap" else None
            ),
            "opponent_policy_path": (
                str(opponent_policy_path) if opponent_init_mode == "from_pretrained" else None
            ),
            "final_policy_path": str(final_policy_path),
            "baseline_metrics": baseline_metrics,
            "final_metrics": final_metrics,
            "improvement": {
                "delta_goals_per_episode": float(
                    final_metrics["goals_per_episode"] - baseline_metrics["goals_per_episode"]
                ),
                "delta_mean_reward": float(
                    final_metrics["mean_reward"] - baseline_metrics["mean_reward"]
                ),
            },
            "self_play": {
                "opponent_update_count": int(callback.opponent_update_count),
                "last_opponent_refresh_step": int(callback.last_opponent_refresh_step),
            },
            "finished_at": datetime.now().isoformat(),
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        if wandb_run is not None:
            wandb_run.log(
                {f"final/{k}": float(v) for k, v in final_metrics.items()},
                step=int(model.num_timesteps),
            )
            wandb_run.log(
                {f"improvement/{k}": float(v) for k, v in summary["improvement"].items()},
                step=int(model.num_timesteps),
            )

        print(f"[done] final policy saved to: {final_policy_path}", flush=True)
        print(f"[done] config saved to: {run_dir / 'config.json'}", flush=True)
        print(f"[done] summary saved to: {run_dir / 'summary.json'}", flush=True)

    finally:
        if wandb_run is not None:
            wandb_run.finish()
        close_env(train_env if train_env is not None else shared_env)


if __name__ == "__main__":
    main()
