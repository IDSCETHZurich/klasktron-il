"""Evaluate policy-vs-policy Klask matchups and write summary metrics as JSON."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

simulation_app = None
_RUNTIME_IMPORTS_READY = False


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _artifacts_dir() -> Path:
    return Path(
        os.environ.get("KLASKTRON_ARTIFACTS_DIR", _repo_root() / "artifacts")
    ).expanduser()


def _get_app_launcher() -> Any:
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Isaac Lab is not available in the current Python environment. "
            "Activate your Isaac Lab environment before running this script."
        ) from exc
    return AppLauncher


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_") or "unnamed"


def _checkpoint_label(path: Optional[str]) -> str:
    if not path:
        return "self_play"
    ckpt_path = Path(path).expanduser()
    if ckpt_path.suffix == ".zip" and ckpt_path.parent.name:
        return _slugify(ckpt_path.parent.name)
    return _slugify(ckpt_path.stem or ckpt_path.name)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate policy-vs-policy Klask matchups (SB3 and RL-Games).",
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate_policy_matchup.py --checkpoint <policy.zip> --policy-format sb3_ppo --episodes 1000\n"
            "  python evaluate_policy_matchup.py --checkpoint <policy.zip> --opponent-policy-path <opp.pth> \\\n"
            "      --episodes 1000 --reward-profile sparse_plus_collision"
        ),
    )

    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("--disable_fabric", action="store_true", default=False)
    eval_group.add_argument("--num_envs", type=int, default=300)
    eval_group.add_argument("--task", type=str, default="Isaac-Klask-v0")
    eval_group.add_argument("--config", type=str, default=None)
    eval_group.add_argument("--episodes", type=int, required=True)
    eval_group.add_argument("--seed", type=int, default=42)
    eval_group.add_argument(
        "--deterministic-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    eval_group.add_argument(
        "--deterministic-opponent",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    eval_group.add_argument("--summary-min-episode-length", type=int, default=3)
    eval_group.add_argument("--log-every-episodes", type=int, default=100)
    eval_group.add_argument(
        "--reward-profile",
        type=str,
        default="sparse_competitive",
        choices=["sparse_competitive", "sparse_plus_collision", "dense_default", "zero"],
    )

    policy_group = parser.add_argument_group("Policy")
    policy_group.add_argument("--checkpoint", type=str, default=None)
    policy_group.add_argument("--policy-format", type=str, default=None)

    opponent_group = parser.add_argument_group("Opponent")
    opponent_group.add_argument("--opponent-policy-path", type=str, default=None)
    opponent_group.add_argument("--opponent-policy-format", type=str, default=None)
    opponent_group.add_argument("--opponent-rl-games-task", type=str, default=None)
    opponent_group.add_argument("--opponent-rl-games-config-path", type=str, default=None)
    opponent_group.add_argument(
        "--force-self-play",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Evaluation output directory. Defaults to artifacts/evaluations/policy_matchups/<derived-name>.",
    )
    output_group.add_argument(
        "--matchup-id",
        type=str,
        default=None,
        help="Label stored in summary metadata. Defaults to a derived policy/opponent name.",
    )
    output_group.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    compat_group = parser.add_argument_group("Compatibility")
    compat_group.add_argument("--policy-path", type=str, default=None)
    compat_group.add_argument("--rl-games-task", type=str, default=None)
    compat_group.add_argument("--rl-games-config-path", type=str, default=None)

    app_launcher_cls = None
    try:
        app_launcher_cls = _get_app_launcher()
    except ModuleNotFoundError:
        app_launcher_cls = None
    if app_launcher_cls is not None:
        app_launcher_cls.add_app_launcher_args(parser)
    return parser


def _parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not bool(getattr(args, "headless", False)):
        args.headless = True
        print("[setup] forcing headless mode by default.")
    return args


def _launch_simulation_app(args: argparse.Namespace) -> None:
    global simulation_app
    app_launcher = _get_app_launcher()(args)
    simulation_app = app_launcher.app


def _import_runtime_dependencies() -> None:
    global _RUNTIME_IMPORTS_READY
    if _RUNTIME_IMPORTS_READY:
        return

    global gym, np, torch, yaml
    global env_configurations, vecenv, BasePlayer, Runner
    global DirectMARLEnv, multi_agent_to_single_agent, retrieve_file_path
    global RlGamesGpuEnv, RlGamesVecEnvWrapper
    global load_cfg_from_registry, parse_env_cfg
    global ActionHistoryWrapper, KlaskAgentOpponentWrapper
    global KlaskCollisionAvoidanceWrapper, KlaskRandomOpponentWrapper
    global ObservationNoiseWrapper, RlGamesGpuEnvSelfPlay, find_wrapper
    global klask_utils_manager_based, ActuatorModelWrapper, KLASK_PARAMS

    import gymnasium as gym
    import numpy as np
    import torch
    import yaml

    from rl_games.common import env_configurations, vecenv
    from rl_games.common.player import BasePlayer
    from rl_games.torch_runner import Runner

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    from isaaclab_tasks.manager_based.klask import (
        ActionHistoryWrapper,
        KlaskAgentOpponentWrapper,
        KlaskCollisionAvoidanceWrapper,
        KlaskRandomOpponentWrapper,
        ObservationNoiseWrapper,
        RlGamesGpuEnvSelfPlay,
        find_wrapper,
    )
    import isaaclab_tasks.manager_based.klask.utils_manager_based as klask_utils_manager_based
    from isaaclab_tasks.manager_based.klask.actuator_model import ActuatorModelWrapper
    from my_isaaclab_tasks.manager_based.klask.assets.klask_cfg import KLASK_PARAMS

    _RUNTIME_IMPORTS_READY = True


POLICY_FORMAT_RL_GAMES = "rl_games"
POLICY_FORMAT_SB3_PPO = "sb3_ppo"
TERMINATION_KEYS = [
    "time_out",
    "goal_scored",
    "goal_conceded",
    "player_in_goal",
    "opponent_in_goal",
]


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def ensure_2d(arr: Any, *, dtype: Any = None) -> np.ndarray:
    if dtype is None:
        dtype = np.float32
    out = to_numpy(arr).astype(dtype, copy=False)
    if out.ndim == 1:
        out = out[None, :]
    return out


def _replace_cfg(obj: Any, **kwargs) -> Any:
    if obj is None or not kwargs:
        return obj
    if hasattr(obj, "replace"):
        return obj.replace(**kwargs)
    for key, value in kwargs.items():
        setattr(obj, key, value)
    return obj


def _zeros_xy_like_env(env: Any) -> torch.Tensor:
    num_envs = int(getattr(env, "num_envs", 1))
    device = getattr(env, "device", "cpu")
    return torch.zeros((num_envs, 2), dtype=torch.float32, device=device)


def _safe_body_xy_pos_w(env: Any, asset_cfg: Any) -> torch.Tensor:
    try:
        return klask_utils_manager_based.body_xy_pos_w(env, asset_cfg)
    except AttributeError as exc:
        if "_data" not in str(exc):
            raise
        return _zeros_xy_like_env(env)


def _safe_body_lin_xy_vel_w(env: Any, asset_cfg: Any) -> torch.Tensor:
    try:
        return klask_utils_manager_based.body_lin_xy_vel_w(env, asset_cfg)
    except AttributeError as exc:
        if "_data" not in str(exc):
            raise
        return _zeros_xy_like_env(env)


def _safe_root_xy_pos_w(env: Any, asset_cfg: Any) -> torch.Tensor:
    try:
        return klask_utils_manager_based.root_xy_pos_w(env, asset_cfg)
    except AttributeError as exc:
        if "_data" not in str(exc):
            raise
        return _zeros_xy_like_env(env)


def _safe_root_lin_xy_vel_w(env: Any, asset_cfg: Any) -> torch.Tensor:
    try:
        return klask_utils_manager_based.root_lin_xy_vel_w(env, asset_cfg)
    except AttributeError as exc:
        if "_data" not in str(exc):
            raise
        return _zeros_xy_like_env(env)


def _patch_observation_terms_for_startup(env_cfg: Any) -> Any:
    obs_cfg = getattr(env_cfg, "observations", None)
    if obs_cfg is None:
        return env_cfg

    safe_func_by_name = {
        "body_xy_pos_w": _safe_body_xy_pos_w,
        "body_lin_xy_vel_w": _safe_body_lin_xy_vel_w,
        "root_xy_pos_w": _safe_root_xy_pos_w,
        "root_lin_xy_vel_w": _safe_root_lin_xy_vel_w,
    }
    obs_updates: Dict[str, Any] = {}
    patched_terms = 0
    for group_name, group_cfg in vars(obs_cfg).items():
        if group_name.startswith("_") or group_cfg is None:
            continue
        term_updates: Dict[str, Any] = {}
        for term_name, term_cfg in vars(group_cfg).items():
            if term_name.startswith("_") or term_cfg is None:
                continue
            func = getattr(term_cfg, "func", None)
            if not callable(func):
                continue
            safe_func = safe_func_by_name.get(getattr(func, "__name__", ""))
            if safe_func is None or func is safe_func:
                continue
            term_updates[term_name] = _replace_cfg(term_cfg, func=safe_func)
        if term_updates:
            patched_terms += len(term_updates)
            obs_updates[group_name] = _replace_cfg(group_cfg, **term_updates)
    if obs_updates:
        obs_cfg = _replace_cfg(obs_cfg, **obs_updates)
        env_cfg = _replace_cfg(env_cfg, observations=obs_cfg)
        print(f"[setup] patched startup-safe observation terms: {patched_terms}")
    return env_cfg


def _patch_velocity_action_cfg_offsets(env_cfg: Any) -> Any:
    actions_cfg = getattr(env_cfg, "actions", None)
    if actions_cfg is None:
        return env_cfg

    updates: Dict[str, Any] = {}
    for action_name in ("player_x", "player_y", "opponent_x", "opponent_y"):
        term_cfg = getattr(actions_cfg, action_name, None)
        if term_cfg is None or not hasattr(term_cfg, "use_default_offset"):
            continue
        if bool(getattr(term_cfg, "use_default_offset", False)):
            updates[action_name] = _replace_cfg(term_cfg, use_default_offset=False)
    if updates:
        actions_cfg = _replace_cfg(actions_cfg, **updates)
        env_cfg = _replace_cfg(env_cfg, actions=actions_cfg)
        print(f"[setup] patched velocity action offsets: {', '.join(sorted(updates.keys()))}")
    return env_cfg


def _install_joint_action_startup_guard() -> None:
    from isaaclab.envs.mdp.actions import joint_actions as joint_action_terms

    if bool(getattr(joint_action_terms, "_klask_joint_action_startup_guard", False)):
        return

    def _guarded_joint_velocity_init(self, cfg, env):
        joint_action_terms.JointAction.__init__(self, cfg, env)
        if bool(getattr(cfg, "use_default_offset", False)):
            data = getattr(self._asset, "_data", None)
            if data is not None and hasattr(data, "default_joint_vel"):
                self._offset = data.default_joint_vel[:, self._joint_ids].clone()

    def _guarded_joint_position_init(self, cfg, env):
        joint_action_terms.JointAction.__init__(self, cfg, env)
        if bool(getattr(cfg, "use_default_offset", False)):
            data = getattr(self._asset, "_data", None)
            if data is not None and hasattr(data, "default_joint_pos"):
                self._offset = data.default_joint_pos[:, self._joint_ids].clone()

    joint_action_terms.JointVelocityAction.__init__ = _guarded_joint_velocity_init
    joint_action_terms.JointPositionAction.__init__ = _guarded_joint_position_init
    joint_action_terms._klask_joint_action_startup_guard = True
    print("[setup] installed joint-action startup guard.")


def _patch_klask_init_joint_pos(env_cfg: Any) -> Any:
    try:
        scene_cfg = getattr(env_cfg, "scene", None)
        klask_cfg = getattr(scene_cfg, "klask", None)
        init_state = getattr(klask_cfg, "init_state", None)
        joint_pos = dict(getattr(init_state, "joint_pos", {}) or {})
        wildcard_default = float(joint_pos.get(".*", 0.0))
        y1 = float(joint_pos.get("ground_to_slider_1", wildcard_default))
        y2 = float(joint_pos.get("ground_to_slider_2", wildcard_default))
        if abs(y1) < 1e-9 or abs(y2) < 1e-9:
            joint_pos.pop(".*", None)
            joint_pos["ground_to_slider_1"] = -0.1
            joint_pos["ground_to_slider_2"] = 0.1
            env_cfg.scene.klask = klask_cfg.replace(
                init_state=init_state.replace(joint_pos=joint_pos)
            )
            print(
                "[setup] patched klask init joint_pos: "
                "ground_to_slider_1=-0.1, ground_to_slider_2=0.1"
            )
    except Exception:
        pass
    return env_cfg


def _apply_reward_profile(env_cfg: Any, reward_profile: str) -> Any:
    if reward_profile == "dense_default":
        print("[setup] reward_profile=dense_default (no reward override).")
        return env_cfg
    rewards_cfg = getattr(env_cfg, "rewards", None)
    if rewards_cfg is None:
        print("[warn] env_cfg has no rewards attribute; reward profile ignored.")
        return env_cfg

    updates: Dict[str, Any] = {}
    for name, term_cfg in vars(rewards_cfg).items():
        if name.startswith("_") or term_cfg is None or not hasattr(term_cfg, "weight"):
            continue
        updates[name] = _replace_cfg(term_cfg, weight=0.0)
    rewards_cfg = _replace_cfg(rewards_cfg, **updates)

    profile_details = "all reward terms set to 0"
    if reward_profile == "sparse_competitive":
        if hasattr(rewards_cfg, "goal_scored"):
            rewards_cfg = _replace_cfg(
                rewards_cfg, goal_scored=_replace_cfg(getattr(rewards_cfg, "goal_scored"), weight=1.0)
            )
        if hasattr(rewards_cfg, "goal_conceded"):
            rewards_cfg = _replace_cfg(
                rewards_cfg, goal_conceded=_replace_cfg(getattr(rewards_cfg, "goal_conceded"), weight=-1.0)
            )
        profile_details = "goal_scored=+1, goal_conceded=-1, others=0"
    elif reward_profile == "sparse_plus_collision":
        if hasattr(rewards_cfg, "goal_scored"):
            rewards_cfg = _replace_cfg(
                rewards_cfg, goal_scored=_replace_cfg(getattr(rewards_cfg, "goal_scored"), weight=1.0)
            )
        if hasattr(rewards_cfg, "goal_conceded"):
            rewards_cfg = _replace_cfg(
                rewards_cfg, goal_conceded=_replace_cfg(getattr(rewards_cfg, "goal_conceded"), weight=-1.0)
            )
        collision_enabled = False
        if hasattr(rewards_cfg, "collision_player_ball"):
            rewards_cfg = _replace_cfg(
                rewards_cfg,
                collision_player_ball=_replace_cfg(
                    getattr(rewards_cfg, "collision_player_ball"),
                    weight=0.1,
                ),
            )
            collision_enabled = True
        profile_details = (
            "goal_scored=+1, goal_conceded=-1, "
            f"collision_player_ball={'+0.1' if collision_enabled else 'unavailable'}, others=0"
        )
    elif reward_profile != "zero":
        raise ValueError(f"Unsupported reward_profile={reward_profile!r}")

    env_cfg = _replace_cfg(env_cfg, rewards=rewards_cfg)
    print(f"[setup] reward_profile={reward_profile} ({profile_details})")
    return env_cfg


def load_agent_cfg(task: str, config_path: Optional[str]) -> Dict[str, Any]:
    if config_path is not None:
        with open(config_path, "r") as fh:
            cfg = yaml.safe_load(fh)
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid RL-Games config at {config_path}.")
        return cfg
    return load_cfg_from_registry(task, "rl_games_cfg_entry_point")


def resolve_checkpoint(args: argparse.Namespace) -> str:
    ckpt = args.checkpoint or args.policy_path
    if ckpt is None:
        raise ValueError("Provide --checkpoint (or legacy --policy-path).")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return retrieve_file_path(ckpt)


def resolve_policy_format_value(raw_fmt: str, *, checkpoint: str, flag_name: str) -> str:
    if raw_fmt in {"sb3", "sb3_ppo", "ppo"}:
        return POLICY_FORMAT_SB3_PPO
    if raw_fmt in {"rl_games", "rlgames", "rl"}:
        return POLICY_FORMAT_RL_GAMES
    if raw_fmt:
        raise ValueError(
            f"Unsupported {flag_name} {raw_fmt!r}. "
            f"Valid values: {POLICY_FORMAT_RL_GAMES}, {POLICY_FORMAT_SB3_PPO}."
        )
    checkpoint_l = checkpoint.lower()
    if checkpoint_l.endswith(".zip"):
        inferred = POLICY_FORMAT_SB3_PPO
    else:
        inferred = POLICY_FORMAT_RL_GAMES
    print(f"[setup] inferred {flag_name}={inferred} from checkpoint path")
    return inferred


def resolve_policy_format(args: argparse.Namespace, checkpoint: str) -> str:
    raw_fmt = str(args.policy_format).strip().lower() if args.policy_format is not None else ""
    return resolve_policy_format_value(raw_fmt, checkpoint=checkpoint, flag_name="--policy-format")


def resolve_opponent_checkpoint_and_format(args: argparse.Namespace) -> Tuple[Optional[str], Optional[str]]:
    if args.opponent_policy_path is None:
        return None, None
    if not os.path.exists(args.opponent_policy_path):
        raise FileNotFoundError(f"Opponent policy not found: {args.opponent_policy_path}")
    ckpt = retrieve_file_path(args.opponent_policy_path)
    raw_fmt = str(args.opponent_policy_format).strip().lower() if args.opponent_policy_format is not None else ""
    fmt = resolve_policy_format_value(raw_fmt, checkpoint=ckpt, flag_name="--opponent-policy-format")
    return ckpt, fmt


def resolve_matchup_id(
    args: argparse.Namespace,
    *,
    checkpoint: str,
    opponent_checkpoint: Optional[str],
) -> str:
    if args.matchup_id:
        return str(args.matchup_id)
    return f"{_checkpoint_label(checkpoint)}__vs__{_checkpoint_label(opponent_checkpoint)}"


def resolve_output_dir(
    args: argparse.Namespace,
    *,
    checkpoint: str,
    opponent_checkpoint: Optional[str],
) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    matchup_id = resolve_matchup_id(args, checkpoint=checkpoint, opponent_checkpoint=opponent_checkpoint)
    return (_artifacts_dir() / "evaluations" / "policy_matchups" / matchup_id).resolve()


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
    if str(cfg.get("demo_source_format", "")).strip().lower() == "human_raw":
        return True
    run_name = os.path.basename(os.path.dirname(os.path.abspath(checkpoint))).upper()
    return "HUMAN_DEMO" in run_name


class Sb3PpoPolicyAdapter:
    """Adapter exposing RL-Games-like methods."""

    def __init__(
        self,
        model: Any,
        *,
        device: str,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
    ):
        self.model = model
        self.device = torch.device(device)
        self.is_rnn = False
        self.states = None
        self.has_batch_dimension = True
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        if self.obs_dim is None:
            space = getattr(model, "observation_space", None)
            shape = getattr(space, "shape", None)
            if shape is not None and len(shape) > 0:
                self.obs_dim = int(shape[0])
        if self.action_dim is None:
            space = getattr(model, "action_space", None)
            shape = getattr(space, "shape", None)
            if shape is not None and len(shape) > 0:
                self.action_dim = int(shape[0])

    def reset(self) -> None:
        return

    def obs_to_torch(self, obs: Any) -> np.ndarray:
        obs_np = ensure_2d(obs, dtype=np.float32)
        if self.obs_dim is not None:
            if obs_np.shape[1] < int(self.obs_dim):
                raise ValueError(
                    f"SB3 policy expects obs_dim={self.obs_dim}, got {obs_np.shape[1]}."
                )
            if obs_np.shape[1] > int(self.obs_dim):
                obs_np = obs_np[:, : int(self.obs_dim)]
        return obs_np

    def get_action(self, obs: Any, is_deterministic: bool = True) -> torch.Tensor:
        obs_np = ensure_2d(obs, dtype=np.float32)
        action_np, _ = self.model.predict(obs_np, deterministic=bool(is_deterministic))
        action_np = ensure_2d(action_np, dtype=np.float32)
        if self.action_dim is not None:
            if action_np.shape[1] < int(self.action_dim):
                raise ValueError(
                    f"SB3 policy returned action_dim={action_np.shape[1]} < expected {self.action_dim}."
                )
            if action_np.shape[1] > int(self.action_dim):
                action_np = action_np[:, : int(self.action_dim)]
        return torch.as_tensor(action_np, dtype=torch.float32, device=self.device)


def build_runner_player(
    *,
    agent_cfg: Dict[str, Any],
    checkpoint: str,
    num_envs: int,
    device: str,
) -> Tuple[Runner, BasePlayer]:
    agent_cfg = dict(agent_cfg)
    agent_cfg["params"] = dict(agent_cfg.get("params", {}))
    agent_cfg["params"]["config"] = dict(agent_cfg["params"].get("config", {}))
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint
    agent_cfg["params"]["config"]["num_actors"] = int(num_envs)
    agent_cfg["params"]["config"]["device"] = device
    if "device_name" in agent_cfg["params"]["config"]:
        agent_cfg["params"]["config"]["device_name"] = device

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    try:
        agent.restore(checkpoint)
    except Exception as exc:
        print(
            f"[warn] restore() failed ({exc}). "
            f"Falling back to direct torch.load(map_location={device})."
        )
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        if "model" not in ckpt:
            raise RuntimeError(f"Checkpoint missing 'model' key: {checkpoint}")
        agent.model.load_state_dict(ckpt["model"])
        if getattr(agent, "normalize_input", False) and "running_mean_std" in ckpt:
            agent.model.running_mean_std.load_state_dict(ckpt["running_mean_std"])
        env_state = ckpt.get("env_state", None)
        if agent.env is not None and env_state is not None:
            agent.env.set_env_state(env_state)
    agent.reset()
    agent.device = torch.device(device)
    agent.model.to(device)
    agent.actions_low = agent.actions_low.to(device)
    agent.actions_high = agent.actions_high.to(device)
    return runner, agent


def build_sb3_player(
    *,
    checkpoint: str,
    device: str,
    obs_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
) -> Sb3PpoPolicyAdapter:
    from stable_baselines3 import PPO

    model = PPO.load(checkpoint, device=device)
    return Sb3PpoPolicyAdapter(
        model,
        device=device,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )


def infer_action_dim_from_env(env: Any) -> Optional[int]:
    try:
        space = getattr(env, "action_space", None)
        shape = getattr(space, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
    except Exception:
        return None
    return None


def build_policy_player(
    *,
    policy_format: str,
    checkpoint: str,
    device: str,
    num_envs: int,
    action_dim: Optional[int],
    rl_games_task: str,
    rl_games_config_path: Optional[str],
) -> Any:
    if policy_format == POLICY_FORMAT_SB3_PPO:
        return build_sb3_player(
            checkpoint=checkpoint,
            device=device,
            action_dim=action_dim,
        )
    if policy_format == POLICY_FORMAT_RL_GAMES:
        rl_cfg = load_agent_cfg(rl_games_task, rl_games_config_path)
        _, player = build_runner_player(
            agent_cfg=rl_cfg,
            checkpoint=checkpoint,
            num_envs=num_envs,
            device=device,
        )
        return player
    raise ValueError(f"Unsupported policy_format={policy_format!r}")


def get_rl_games_action_clip(
    *,
    task: str,
    config_path: Optional[str],
) -> Optional[float]:
    try:
        cfg = load_agent_cfg(task, config_path)
        clip = cfg.get("params", {}).get("env", {}).get("clip_actions", None)
        if clip is None:
            return None
        clip_f = float(clip)
        if not np.isfinite(clip_f) or clip_f <= 0.0:
            return None
        return clip_f
    except Exception:
        return None


def apply_policy_action_clip(
    *,
    policy: Any,
    clip_value: Optional[float],
    label: str,
) -> None:
    if clip_value is None:
        return
    if bool(getattr(policy, "_klask_action_clip_patched", False)):
        return
    original_get_action = getattr(policy, "get_action", None)
    if not callable(original_get_action):
        return
    clip = float(clip_value)

    def _clipped_get_action(obs: Any, is_deterministic: bool = True):
        out = original_get_action(obs, is_deterministic)
        if torch.is_tensor(out):
            return torch.clamp(out, -clip, clip)
        out_arr = np.clip(ensure_2d(out, dtype=np.float32), -clip, clip)
        device = getattr(policy, "device", "cpu")
        return torch.as_tensor(out_arr, dtype=torch.float32, device=device)

    setattr(policy, "get_action", _clipped_get_action)
    setattr(policy, "_klask_action_clip_patched", True)
    setattr(policy, "_klask_action_clip_value", clip)
    print(f"[setup] action clip applied to {label}: +/-{clip}")


def build_env(
    *,
    task: str,
    num_envs: int,
    device: str,
    disable_fabric: bool,
    agent_cfg: Dict[str, Any],
    reward_profile: str,
    seed: Optional[int],
    actuator_apply_player: bool = True,
    actuator_apply_opponent: bool = True,
) -> Any:
    _install_joint_action_startup_guard()
    env_cfg = parse_env_cfg(
        task,
        device=device,
        num_envs=num_envs,
        use_fabric=not disable_fabric,
    )
    if seed is not None and hasattr(env_cfg, "seed"):
        env_cfg = _replace_cfg(env_cfg, seed=int(seed))
    env_cfg = _patch_klask_init_joint_pos(env_cfg)
    env_cfg = _patch_velocity_action_cfg_offsets(env_cfg)
    env_cfg = _patch_observation_terms_for_startup(env_cfg)
    env_cfg = _apply_reward_profile(env_cfg, reward_profile=reward_profile)

    env = gym.make(task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.get("env", {}).get("actuator_model", False):
        env = ActuatorModelWrapper(
            env,
            device=device,
            apply_to_player=bool(actuator_apply_player),
            apply_to_opponent=bool(actuator_apply_opponent),
        )
    if agent_cfg.get("env", {}).get("collision_avoidance", False):
        env = KlaskCollisionAvoidanceWrapper(env)
    if KLASK_PARAMS.get("action_history", 0) > 0:
        env = ActionHistoryWrapper(env, history_length=KLASK_PARAMS["action_history"])
    obs_noise = float(agent_cfg.get("env", {}).get("obs_noise", 0.0))
    if obs_noise > 0.0:
        env = ObservationNoiseWrapper(env, obs_noise)

    self_play = bool(agent_cfg.get("params", {}).get("config", {}).get("self_play", False))
    if self_play:
        print("[setup] opponent_mode=self_play (policy-controlled opponent)")
        env = KlaskAgentOpponentWrapper(env)
        if hasattr(env.unwrapped, "single_action_space"):
            original_space = env.unwrapped.single_action_space
            if hasattr(original_space, "shape") and int(original_space.shape[0]) == 4:
                env.unwrapped._klask_original_single_action_space = original_space
                player_space = gym.spaces.Box(
                    low=original_space.low[:2],
                    high=original_space.high[:2],
                    dtype=original_space.dtype,
                )
                env.unwrapped.single_action_space = player_space
                try:
                    env.action_space = player_space
                except Exception:
                    pass
    else:
        print("[setup] opponent_mode=random (KlaskRandomOpponentWrapper)")
        env = KlaskRandomOpponentWrapper(env)

    clip_obs = float(agent_cfg.get("params", {}).get("env", {}).get("clip_observations", np.inf))
    clip_actions = float(agent_cfg.get("params", {}).get("env", {}).get("clip_actions", np.inf))
    rl_env = RlGamesVecEnvWrapper(env, device, clip_obs=clip_obs, clip_actions=clip_actions)

    if self_play:
        try:
            vecenv.register(
                "IsaacRlgWrapper",
                lambda config_name, num_actors, **kwargs: RlGamesGpuEnvSelfPlay(
                    config_name, num_actors, agent_cfg.copy(), **kwargs
                ),
            )
        except Exception:
            pass
    else:
        try:
            vecenv.register(
                "IsaacRlgWrapper",
                lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(
                    config_name, num_actors, **kwargs
                ),
            )
        except Exception:
            pass
    env_configurations.configurations["rlgpu"] = {
        "vecenv_type": "IsaacRlgWrapper",
        "env_creator": lambda **kwargs: rl_env,
    }
    return rl_env


def _find_termination_manager(env: Any) -> Optional[Any]:
    queue = [env]
    visited = set()
    while queue:
        cur = queue.pop(0)
        if cur is None:
            continue
        cur_id = id(cur)
        if cur_id in visited:
            continue
        visited.add(cur_id)
        term_mgr = getattr(cur, "termination_manager", None)
        if term_mgr is not None:
            return term_mgr
        for attr in ("env", "_env", "unwrapped", "venv"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                queue.append(nxt)
    return None


def _to_bool_vector(x: Any, num_envs: int) -> np.ndarray:
    arr = to_numpy(x).reshape(-1)
    if arr.size == 0:
        return np.zeros((num_envs,), dtype=bool)
    out = arr.astype(bool)
    if out.size != num_envs:
        out = np.resize(out, (num_envs,))
    return out


def _term_flags_from_info(infos: Any, num_envs: int) -> Dict[str, np.ndarray]:
    flags = {name: np.zeros((num_envs,), dtype=bool) for name in TERMINATION_KEYS}
    if infos is None:
        return flags
    info_list: List[Any]
    if isinstance(infos, (list, tuple)):
        info_list = list(infos)
    else:
        info_list = [infos] * num_envs
    if len(info_list) < num_envs:
        info_list += [{} for _ in range(num_envs - len(info_list))]
    for i in range(num_envs):
        info_i = info_list[i] if isinstance(info_list[i], dict) else {}
        for name in TERMINATION_KEYS:
            flags[name][i] = bool(info_i.get(name, False))
    return flags


def evaluate_matchup(
    *,
    env: Any,
    agent: Any,
    episodes: int,
    deterministic_policy: bool,
    log_every_episodes: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    obs_arr = ensure_2d(obs)
    num_envs = int(obs_arr.shape[0])

    if hasattr(agent, "get_batch_size"):
        _ = agent.get_batch_size(obs, 1)
    if getattr(agent, "is_rnn", False) and hasattr(agent, "init_rnn"):
        agent.init_rnn()

    self_play_wrapper = None
    try:
        self_play_wrapper = find_wrapper(env.env, KlaskAgentOpponentWrapper)
    except Exception:
        self_play_wrapper = None

    term_mgr = _find_termination_manager(env)
    term_names = list(TERMINATION_KEYS)
    if term_mgr is not None:
        active_terms = list(getattr(term_mgr, "active_terms", []))
        if active_terms:
            term_names = [name for name in TERMINATION_KEYS if name in active_terms]
            if not term_names:
                term_names = list(TERMINATION_KEYS)

    ep_reward = np.zeros((num_envs,), dtype=np.float64)
    ep_len = np.zeros((num_envs,), dtype=np.int64)
    ep_goals = np.zeros((num_envs,), dtype=np.int64)
    ep_conceded = np.zeros((num_envs,), dtype=np.int64)

    rows: List[Dict[str, Any]] = []
    total_steps = 0
    total_goal_steps = 0
    total_conceded_steps = 0

    episodes_completed = 0
    started = time.perf_counter()
    last_log_ep = 0

    while simulation_app.is_running() and episodes_completed < int(episodes):
        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs)
            act_peg1 = agent.get_action(obs_t, is_deterministic=bool(deterministic_policy))
            act_peg1 = torch.clamp(act_peg1, -1.0, 1.0)

            next_obs, reward, dones, infos = env.step(act_peg1)
            if isinstance(next_obs, dict):
                next_obs = next_obs["obs"]

            done_np = to_numpy(dones).astype(bool).reshape(-1)
            rew_np = to_numpy(reward).astype(np.float64).reshape(-1)
            total_steps += int(rew_np.size)

            term_flags = {name: np.zeros((num_envs,), dtype=bool) for name in TERMINATION_KEYS}
            if term_mgr is not None:
                try:
                    for name in term_names:
                        term_flags[name] = _to_bool_vector(term_mgr.get_term(name), num_envs)
                except Exception:
                    term_flags = _term_flags_from_info(infos, num_envs)
            else:
                term_flags = _term_flags_from_info(infos, num_envs)

            goal_event = np.logical_and(term_flags.get("goal_scored", np.zeros(num_envs, dtype=bool)), done_np)
            conceded_event = np.logical_and(
                term_flags.get("goal_conceded", np.zeros(num_envs, dtype=bool)), done_np
            )

            ep_reward += rew_np
            ep_len += 1
            ep_goals += goal_event.astype(np.int64)
            ep_conceded += conceded_event.astype(np.int64)

            total_goal_steps += int(np.sum(goal_event))
            total_conceded_steps += int(np.sum(conceded_event))

            if np.any(done_np):
                done_idx = np.where(done_np)[0]
                for i in done_idx:
                    row = {
                        "episode": int(episodes_completed),
                        "reward": float(ep_reward[i]),
                        "length": int(ep_len[i]),
                        "goals_scored": int(ep_goals[i]),
                        "goals_conceded": int(ep_conceded[i]),
                    }
                    for name in TERMINATION_KEYS:
                        row[f"term_{name}"] = int(term_flags.get(name, np.zeros(num_envs, dtype=bool))[i])
                    rows.append(row)
                    episodes_completed += 1
                    if episodes_completed >= int(episodes):
                        break

                ep_reward[done_idx] = 0.0
                ep_len[done_idx] = 0
                ep_goals[done_idx] = 0
                ep_conceded[done_idx] = 0

            if (
                episodes_completed > 0
                and (episodes_completed - last_log_ep) >= int(max(1, log_every_episodes))
            ):
                elapsed = max(1e-6, time.perf_counter() - started)
                ep_per_s = episodes_completed / elapsed
                remaining = int(max(0, int(episodes) - episodes_completed))
                eta = remaining / max(1e-6, ep_per_s)
                print(
                    f"[progress] episodes={episodes_completed}/{int(episodes)} "
                    f"({100.0 * episodes_completed / max(1, int(episodes)):5.1f}%) "
                    f"eps/s={ep_per_s:.2f} eta_sec={eta:.1f}"
                )
                last_log_ep = episodes_completed

            if getattr(agent, "is_rnn", False) and getattr(agent, "states", None) is not None:
                done_t = torch.as_tensor(done_np, dtype=torch.bool, device=agent.device)
                for s in agent.states:
                    s[:, done_t, :] = 0.0

            if self_play_wrapper is not None and self_play_wrapper.opponent is not None:
                opp = self_play_wrapper.opponent
                if getattr(opp, "is_rnn", False) and getattr(opp, "states", None) is not None:
                    done_t = torch.as_tensor(done_np, dtype=torch.bool, device=opp.device)
                    for s in opp.states:
                        s[:, done_t, :] = 0.0

            obs = next_obs

    elapsed_sec = float(time.perf_counter() - started)
    counters = {
        "total_steps": int(total_steps),
        "total_goal_steps": int(total_goal_steps),
        "total_conceded_steps": int(total_conceded_steps),
        "elapsed_sec": elapsed_sec,
        "num_envs": int(num_envs),
    }
    return rows, counters


def summarize_rows(
    *,
    rows: List[Dict[str, Any]],
    summary_min_episode_length: int,
    counters: Dict[str, Any],
) -> Dict[str, Any]:
    summary_rows = [r for r in rows if int(r["length"]) > int(summary_min_episode_length)]
    total_eps = int(len(rows))
    used_eps = int(len(summary_rows))
    discarded = int(total_eps - used_eps)

    rewards = np.asarray([r["reward"] for r in summary_rows], dtype=np.float64)
    lengths = np.asarray([r["length"] for r in summary_rows], dtype=np.float64)
    goals = np.asarray([r["goals_scored"] for r in summary_rows], dtype=np.float64)
    conceded = np.asarray([r["goals_conceded"] for r in summary_rows], dtype=np.float64)

    summary: Dict[str, Any] = {
        "episodes_total": total_eps,
        "episodes_used_for_summary": used_eps,
        "episodes_discarded_short": discarded,
        "summary_discard_length_lte": int(summary_min_episode_length),
        "mean_env_reward": float(rewards.mean()) if rewards.size else float("nan"),
        "std_env_reward": float(rewards.std()) if rewards.size else float("nan"),
        "mean_episode_length": float(lengths.mean()) if lengths.size else float("nan"),
        "goals_scored_per_episode": float(goals.mean()) if goals.size else float("nan"),
        "goals_conceded_per_episode": float(conceded.mean()) if conceded.size else float("nan"),
        "goal_diff_per_episode": float((goals - conceded).mean()) if goals.size else float("nan"),
        "goal_episode_rate": float((goals > 0).mean()) if goals.size else float("nan"),
        "conceded_episode_rate": float((conceded > 0).mean()) if conceded.size else float("nan"),
        "goal_steps_rate": (
            float(counters["total_goal_steps"] / counters["total_steps"])
            if counters["total_steps"] > 0
            else float("nan")
        ),
        "conceded_steps_rate": (
            float(counters["total_conceded_steps"] / counters["total_steps"])
            if counters["total_steps"] > 0
            else float("nan")
        ),
        "elapsed_sec": float(counters["elapsed_sec"]),
        "num_envs": int(counters["num_envs"]),
    }
    for name in TERMINATION_KEYS:
        key = f"term_{name}"
        vals = np.asarray([r.get(key, 0) for r in summary_rows], dtype=np.float64)
        summary[f"term_rate_{name}"] = float(vals.mean()) if vals.size else float("nan")
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_cli_args(argv)
    _launch_simulation_app(args)
    _import_runtime_dependencies()
    if int(args.episodes) <= 0:
        raise ValueError("--episodes must be > 0")
    if int(args.num_envs) <= 0:
        raise ValueError("--num_envs must be > 0")
    if int(args.log_every_episodes) <= 0:
        raise ValueError("--log-every-episodes must be > 0")

    if args.rl_games_config_path is not None and args.config is None:
        args.config = args.rl_games_config_path

    checkpoint = resolve_checkpoint(args)
    policy_format = resolve_policy_format(args, checkpoint)
    opponent_checkpoint, opponent_policy_format = resolve_opponent_checkpoint_and_format(args)
    matchup_id = resolve_matchup_id(args, checkpoint=checkpoint, opponent_checkpoint=opponent_checkpoint)

    if args.config is not None and not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    output_dir = resolve_output_dir(args, checkpoint=checkpoint, opponent_checkpoint=opponent_checkpoint)
    summary_path = output_dir / "summary.json"
    if os.path.exists(summary_path) and not bool(args.force):
        print(f"[cache] existing evaluation found: {summary_path}")
        print(summary_path)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] matchup_id={matchup_id}")
    print(f"[setup] output_dir={output_dir}")

    agent_cfg = load_agent_cfg(args.task, args.config)
    if args.force_self_play is not None:
        agent_cfg.setdefault("params", {}).setdefault("config", {})["self_play"] = bool(args.force_self_play)
    if opponent_checkpoint is not None:
        agent_cfg.setdefault("params", {}).setdefault("config", {})["self_play"] = True
        print("[setup] fixed opponent requested; forcing self_play wrapper.")
    self_play_enabled = bool(agent_cfg.get("params", {}).get("config", {}).get("self_play", False))
    player_bypass_actuator = is_human_demo_bc_checkpoint(checkpoint)
    opponent_bypass_actuator = is_human_demo_bc_checkpoint(opponent_checkpoint)
    if opponent_checkpoint is None and self_play_enabled and player_bypass_actuator:
        opponent_bypass_actuator = True
    if player_bypass_actuator:
        print("[setup] bypassing actuator model on player side for human-demo BC checkpoint.")
    if opponent_bypass_actuator:
        print("[setup] bypassing actuator model on opponent side for human-demo BC checkpoint.")

    env = None
    env = build_env(
        task=args.task,
        num_envs=int(args.num_envs),
        device=args.device,
        disable_fabric=bool(args.disable_fabric),
        agent_cfg=agent_cfg,
        reward_profile=str(args.reward_profile),
        seed=int(args.seed),
        actuator_apply_player=not player_bypass_actuator,
        actuator_apply_opponent=not opponent_bypass_actuator,
    )

    runner: Optional[Runner] = None
    try:
        sb3_action_dim = infer_action_dim_from_env(env)
        print(f"[setup] loading policy: {checkpoint}")
        if policy_format == POLICY_FORMAT_RL_GAMES:
            runner, agent = build_runner_player(
                agent_cfg=agent_cfg,
                checkpoint=checkpoint,
                num_envs=int(args.num_envs),
                device=args.device,
            )
            expert_action_clip = get_rl_games_action_clip(task=args.task, config_path=args.config)
            apply_policy_action_clip(policy=agent, clip_value=expert_action_clip, label="policy")
        else:
            agent = build_sb3_player(
                checkpoint=checkpoint,
                device=args.device,
                action_dim=sb3_action_dim,
            )

        if bool(agent_cfg.get("params", {}).get("config", {}).get("self_play", False)):
            if opponent_checkpoint is not None:
                opponent_task = args.opponent_rl_games_task or args.task
                print(f"[setup] loading fixed opponent policy: {opponent_checkpoint}")
                opponent = build_policy_player(
                    policy_format=str(opponent_policy_format),
                    checkpoint=opponent_checkpoint,
                    device=args.device,
                    num_envs=int(args.num_envs),
                    action_dim=sb3_action_dim,
                    rl_games_task=str(opponent_task),
                    rl_games_config_path=args.opponent_rl_games_config_path,
                )
                if opponent_policy_format == POLICY_FORMAT_RL_GAMES:
                    opponent_action_clip = get_rl_games_action_clip(
                        task=str(opponent_task),
                        config_path=args.opponent_rl_games_config_path,
                    )
                    apply_policy_action_clip(
                        policy=opponent,
                        clip_value=opponent_action_clip,
                        label="opponent",
                    )
            else:
                if policy_format == POLICY_FORMAT_RL_GAMES:
                    if runner is None:
                        raise RuntimeError("RL-Games runner not initialized.")
                    opponent = runner.create_player()
                    opponent.device = torch.device(args.device)
                    opponent.model.to(args.device)
                    opponent.actions_low = agent.actions_low.to(args.device)
                    opponent.actions_high = agent.actions_high.to(args.device)
                    opponent.set_weights(agent.get_weights())
                else:
                    opponent = build_sb3_player(
                        checkpoint=checkpoint,
                        device=args.device,
                        action_dim=sb3_action_dim,
                    )
            self_play_wrapper = find_wrapper(env.env, KlaskAgentOpponentWrapper)
            self_play_wrapper.is_deterministic = bool(args.deterministic_opponent)
            self_play_wrapper.add_opponent(opponent)
            print(
                "[setup] self_play opponent deterministic="
                f"{bool(self_play_wrapper.is_deterministic)}"
            )

        print(
            f"[eval] collecting {int(args.episodes)} completed episodes "
            f"(num_envs={int(args.num_envs)})"
        )
        rows, counters = evaluate_matchup(
            env=env,
            agent=agent,
            episodes=int(args.episodes),
            deterministic_policy=bool(args.deterministic_policy),
            log_every_episodes=int(args.log_every_episodes),
        )
        summary = summarize_rows(
            rows=rows,
            summary_min_episode_length=int(args.summary_min_episode_length),
            counters=counters,
        )
        summary.update(
            {
                "created_at": datetime.datetime.now().isoformat(),
                "matchup_id": matchup_id,
                "episodes_requested": int(args.episodes),
                "task": str(args.task),
                "device": str(args.device),
                "seed": int(args.seed),
                "checkpoint": checkpoint,
                "policy_format": policy_format,
                "opponent_checkpoint": opponent_checkpoint,
                "opponent_policy_format": opponent_policy_format,
                "rl_games_config_path": args.config,
                "opponent_rl_games_config_path": args.opponent_rl_games_config_path,
                "deterministic_policy": bool(args.deterministic_policy),
                "deterministic_opponent": bool(args.deterministic_opponent),
                "reward_profile": str(args.reward_profile),
                "force_self_play": bool(args.force_self_play)
                if args.force_self_play is not None
                else None,
            }
        )
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[out] wrote {summary_path}")
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}")
        raise
    finally:
        try:
            if simulation_app is not None:
                simulation_app.close()
        except Exception:
            pass
