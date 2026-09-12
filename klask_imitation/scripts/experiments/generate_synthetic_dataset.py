"""Collect synthetic trajectories from RL-Games or SB3 checkpoints."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pickle
import re
import sys
import time
import traceback
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
        description="Collect synthetic trajectories from RL-Games or SB3 checkpoints.",
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate_synthetic_dataset.py --checkpoint <policy.zip> --policy-format sb3_ppo --target-steps 5000000\n"
            "  python generate_synthetic_dataset.py --checkpoint <policy.pth> --policy-format rl_games \\\n"
            "      --target-steps 80000 --opponent-policy-path <opp.zip>"
        ),
    )

    rollout_group = parser.add_argument_group("Rollout")
    rollout_group.add_argument("--video", action="store_true", default=False)
    rollout_group.add_argument("--video_length", type=int, default=200)
    rollout_group.add_argument("--disable_fabric", action="store_true", default=False)
    rollout_group.add_argument("--num_envs", type=int, default=256)
    rollout_group.add_argument("--task", type=str, default="Isaac-Klask-v0")
    rollout_group.add_argument("--target-steps", type=int, required=True)
    rollout_group.add_argument("--chunk-size", type=int, default=64)
    rollout_group.add_argument("--log-every-steps", type=int, default=1000)
    rollout_group.add_argument("--min-traj-len", type=int, default=1)
    rollout_group.add_argument("--seed", type=int, default=42)
    rollout_group.add_argument(
        "--deterministic-expert",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    rollout_group.add_argument(
        "--deterministic-opponent",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    policy_group = parser.add_argument_group("Policy")
    policy_group.add_argument("--checkpoint", type=str, default=None)
    policy_group.add_argument("--config", type=str, default=None)
    policy_group.add_argument("--policy-format", type=str, default=None)
    policy_group.add_argument("--expert-id", type=str, default="rl_games_expert")
    policy_group.add_argument("--mode", type=str, default="peg1_self_play")

    opponent_group = parser.add_argument_group("Opponent")
    opponent_group.add_argument("--opponent-policy-path", type=str, default=None)
    opponent_group.add_argument("--opponent-policy-format", type=str, default=None)
    opponent_group.add_argument("--opponent-rl-games-task", type=str, default=None)
    opponent_group.add_argument("--opponent-rl-games-config-path", type=str, default=None)
    opponent_group.add_argument(
        "--invert-opponent-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    opponent_group.add_argument(
        "--mirror-opponent-x",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    opponent_group.add_argument(
        "--mirror-opponent-y",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
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
        help="Dataset output directory. Defaults to artifacts/data_wall/datasets/<derived-name>.",
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
    if args.video:
        args.enable_cameras = True
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
    global DirectMARLEnv, multi_agent_to_single_agent
    global print_dict, retrieve_file_path
    global RlGamesGpuEnv, RlGamesVecEnvWrapper
    global load_cfg_from_registry, parse_env_cfg
    global ActionHistoryWrapper, CurriculumWrapper, KlaskAgentOpponentWrapper
    global KlaskCollisionAvoidanceWrapper, KlaskRandomOpponentWrapper
    global ObservationNoiseWrapper, RlGamesGpuEnvSelfPlay, find_wrapper
    global klask_utils_manager_based, set_terminations, ActuatorModelWrapper
    global KLASK_PARAMS

    import gymnasium as gym
    import numpy as np
    import torch
    import yaml

    from rl_games.common import env_configurations, vecenv
    from rl_games.common.player import BasePlayer
    from rl_games.torch_runner import Runner

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    from isaaclab_tasks.manager_based.klask import (
        ActionHistoryWrapper,
        CurriculumWrapper,
        KlaskAgentOpponentWrapper,
        KlaskCollisionAvoidanceWrapper,
        KlaskRandomOpponentWrapper,
        ObservationNoiseWrapper,
        RlGamesGpuEnvSelfPlay,
        find_wrapper,
    )
    import isaaclab_tasks.manager_based.klask.utils_manager_based as klask_utils_manager_based
    from isaaclab_tasks.manager_based.klask.utils_manager_based import set_terminations
    from isaaclab_tasks.manager_based.klask.actuator_model import ActuatorModelWrapper
    from my_isaaclab_tasks.manager_based.klask.assets.klask_cfg import KLASK_PARAMS

    _RUNTIME_IMPORTS_READY = True

POLICY_FORMAT_RL_GAMES = "rl_games"
POLICY_FORMAT_SB3_PPO = "sb3_ppo"
CORE_TERMINATION_MODES = (
    "player_in_goal",
    "opponent_in_goal",
    "goal_scored",
    "goal_conceded",
    "time_out",
)
TERMINATION_MODE_PRIORITY = (
    "goal_scored",
    "goal_conceded",
    "player_in_goal",
    "opponent_in_goal",
    "time_out",
)


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


def format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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
        raise ValueError(f"Termination term size mismatch: expected {num_envs}, got {flat.size}.")
    return flat


def _find_termination_manager(env: Any) -> Optional[Any]:
    """Best-effort lookup of termination manager through common wrapper attributes."""
    seen = set()
    queue = [env]
    while queue:
        cur = queue.pop(0)
        if cur is None:
            continue
        ident = id(cur)
        if ident in seen:
            continue
        seen.add(ident)
        term_mgr = getattr(cur, "termination_manager", None)
        if term_mgr is not None:
            return term_mgr
        for attr in ("unwrapped", "env", "venv", "_env", "_wrapped_env"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                queue.append(nxt)
    return None


def _default_info_dict(done: bool) -> Dict[str, Any]:
    payload = {
        "done": bool(done),
        "termination_mode": "none" if not done else "unknown",
        "termination_terms": [],
    }
    for name in CORE_TERMINATION_MODES:
        payload[f"term_{name}"] = False
    return payload


def _normalize_step_infos(step_info: Any, num_envs: int) -> List[Dict[str, Any]]:
    """Normalize env.step info payload into per-env list[dict]."""
    out = [{} for _ in range(num_envs)]
    if step_info is None:
        return out
    if isinstance(step_info, (list, tuple)):
        for i, item in enumerate(step_info[:num_envs]):
            if isinstance(item, dict):
                out[i] = dict(item)
        return out
    if isinstance(step_info, dict):
        for key, value in step_info.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                arr = value
            elif isinstance(value, (list, tuple)):
                arr = np.asarray(value, dtype=object)
            else:
                arr = None

            if arr is not None and arr.ndim >= 1 and arr.shape[0] == num_envs:
                for i in range(num_envs):
                    elem = arr[i]
                    if isinstance(elem, np.ndarray):
                        out[i][key] = elem.item() if elem.ndim == 0 else elem.tolist()
                    elif isinstance(elem, np.generic):
                        out[i][key] = elem.item()
                    else:
                        out[i][key] = elem
            else:
                scalar = value.item() if isinstance(value, np.generic) else value
                for i in range(num_envs):
                    out[i][key] = scalar
        return out
    return out


def _termination_flags_from_step(
    *,
    term_mgr: Optional[Any],
    term_names: List[str],
    step_info: Any,
    num_envs: int,
) -> Dict[str, np.ndarray]:
    """Read per-env termination flags from manager (preferred) or env.step info fallback."""
    flags: Dict[str, np.ndarray] = {}
    if term_mgr is not None:
        for name in term_names:
            try:
                flags[name] = _to_bool_vector(term_mgr.get_term(name), num_envs)
            except Exception:
                flags[name] = np.zeros(num_envs, dtype=bool)
        return flags

    infos = _normalize_step_infos(step_info, num_envs)
    for name in term_names:
        vec = np.zeros(num_envs, dtype=bool)
        for i, item in enumerate(infos):
            val = item.get(name, item.get(f"term_{name}", False))
            if name == "time_out":
                val = val or item.get("timeout", False) or item.get("time_out", False)
            vec[i] = bool(val)
        flags[name] = vec
    return flags


def _termination_mode_from_flags(done: bool, per_env_flags: Dict[str, bool]) -> str:
    if not bool(done):
        return "none"
    for name in TERMINATION_MODE_PRIORITY:
        if bool(per_env_flags.get(name, False)):
            return name
    active = [name for name, value in per_env_flags.items() if bool(value)]
    if active:
        return str(active[0])
    return "unknown"


def _slice_transition_infos(infos: Any, length: int) -> Optional[List[Dict[str, Any]]]:
    """Slice per-transition infos to match transition count."""
    if infos is None:
        return None
    try:
        seq = list(infos)
    except TypeError:
        return None

    out: List[Dict[str, Any]] = []
    for item in seq[: int(length)]:
        out.append(dict(item) if isinstance(item, dict) else {})

    if len(out) < int(length):
        out.extend({} for _ in range(int(length) - len(out)))
    return out


def trim_trajectories_to_target(
    trajectories: List[Dict[str, Any]],
    *,
    target_steps: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    total = 0
    for tr in trajectories:
        obs = np.asarray(tr["obs"], dtype=np.float32)
        acts = np.asarray(tr["acts"], dtype=np.float32)
        rews = np.asarray(tr.get("rews", np.zeros((len(acts),), dtype=np.float32)))
        t = len(acts)
        if t <= 0:
            continue
        remaining = int(target_steps - total)
        if remaining <= 0:
            break
        if t <= remaining:
            out.append(
                {
                    "obs": obs,
                    "acts": acts,
                    "rews": rews[:t],
                    "infos": _slice_transition_infos(tr.get("infos", None), t),
                    "terminal": bool(tr.get("terminal", True)),
                }
            )
            total += t
        else:
            out.append(
                {
                    "obs": obs[: remaining + 1],
                    "acts": acts[:remaining],
                    "rews": rews[:remaining],
                    "infos": _slice_transition_infos(tr.get("infos", None), remaining),
                    "terminal": False,
                }
            )
            total += remaining
            break
    if total != target_steps:
        raise RuntimeError(f"Failed to trim to {target_steps} steps (got {total}).")
    return out


def write_trajectory_chunks(
    *,
    trajectories: List[Dict[str, Any]],
    output_dir: str,
    chunk_size: int,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    out_paths: List[str] = []
    for start in range(0, len(trajectories), int(chunk_size)):
        chunk = trajectories[start : start + int(chunk_size)]
        path = os.path.join(output_dir, f"chunk_{start // int(chunk_size):05d}.pkl")
        with open(path, "wb") as fh:
            pickle.dump(chunk, fh, protocol=pickle.HIGHEST_PROTOCOL)
        out_paths.append(path)
    return out_paths


def load_agent_cfg(task: str, config_path: Optional[str]) -> Dict[str, Any]:
    # If config is provided, use it directly (same data used by play_klask command).
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


def resolve_policy_format(args: argparse.Namespace, checkpoint: str) -> str:
    raw_fmt = str(args.policy_format).strip().lower() if args.policy_format is not None else ""
    return resolve_policy_format_value(raw_fmt, checkpoint=checkpoint, flag_name="--policy-format")


def resolve_policy_format_value(
    raw_fmt: str,
    *,
    checkpoint: str,
    flag_name: str,
) -> str:
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
    print(f"[setup] inferred policy_format={inferred} from checkpoint path")
    return inferred


def resolve_opponent_checkpoint_and_format(args: argparse.Namespace) -> Tuple[Optional[str], Optional[str]]:
    if args.opponent_policy_path is None:
        return None, None
    if not os.path.exists(args.opponent_policy_path):
        raise FileNotFoundError(f"Opponent policy not found: {args.opponent_policy_path}")
    ckpt = retrieve_file_path(args.opponent_policy_path)
    raw_fmt = str(args.opponent_policy_format).strip().lower() if args.opponent_policy_format is not None else ""
    fmt = resolve_policy_format_value(raw_fmt, checkpoint=ckpt, flag_name="--opponent-policy-format")
    return ckpt, fmt


def resolve_output_dir(
    args: argparse.Namespace,
    *,
    checkpoint: str,
    opponent_checkpoint: Optional[str],
) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()

    policy_label = _checkpoint_label(checkpoint)
    opponent_label = (
        "self_play" if opponent_checkpoint is None else f"vs_{_checkpoint_label(opponent_checkpoint)}"
    )
    derived_name = f"{policy_label}_{opponent_label}_{int(args.target_steps)}"
    return (_artifacts_dir() / "data_wall" / "datasets" / derived_name).resolve()


def _clear_generated_dataset_outputs(output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    traj_dir = output_dir / "trajectories"
    if traj_dir.exists():
        for pkl_path in traj_dir.glob("*.pkl"):
            pkl_path.unlink()


def prepare_dataset_output_dir(output_dir: Path, *, force: bool) -> Tuple[Path, Path, Path]:
    traj_dir = output_dir / "trajectories"
    manifest_path = output_dir / "manifest.json"

    if manifest_path.exists() and not force:
        print(f"[cache] existing dataset found: {output_dir}")
        print(output_dir)
        raise SystemExit(0)

    if output_dir.exists():
        existing_items = [p for p in output_dir.iterdir()]
        has_known_outputs = manifest_path.exists() or traj_dir.exists()
        has_existing_chunks = traj_dir.exists() and any(traj_dir.glob("*.pkl"))
        if existing_items and not has_known_outputs and not force:
            raise FileExistsError(
                f"Output directory already exists and is not a known generated dataset: {output_dir}. "
                "Pass --force to reuse it or choose --output-dir explicitly."
            )
        if has_existing_chunks and not manifest_path.exists() and not force:
            raise FileExistsError(
                f"Output directory already contains trajectory chunks without a manifest: {output_dir}. "
                "Pass --force to overwrite them."
            )
        if force:
            _clear_generated_dataset_outputs(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, traj_dir, manifest_path


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
    checkpoint_name = os.path.basename(checkpoint)
    if checkpoint_name != "ppo_policy_bc_final.zip":
        return False
    cfg = _load_checkpoint_run_config(checkpoint)
    if str(cfg.get("demo_source_format", "")).strip().lower() == "human_raw":
        return True
    run_name = os.path.basename(os.path.dirname(os.path.abspath(checkpoint))).upper()
    return "HUMAN_DEMO" in run_name


class Sb3PpoPolicyAdapter:
    """Adapter exposing RL-Games-like methods used by rollout/self-play wrappers."""

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


def _replace_cfg(obj: Any, **kwargs) -> Any:
    """Replace configclass/dataclass fields, with mutable-object fallback."""
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
    """Avoid early manager shape-probes from crashing before articulation data exists."""
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
    """Disable default velocity-offset reads that require articulation data at init."""
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
    """Guard IsaacLab joint action init against missing articulation data."""
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


def build_env(
    *,
    task: str,
    num_envs: int,
    device: str,
    disable_fabric: bool,
    video: bool,
    video_length: int,
    output_dir: str,
    agent_cfg: Dict[str, Any],
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

    # Some Klask task variants define default joint positions at 0.0, which can be
    # outside USD joint limits for the Y sliders and cause Articulation validation to fail.
    # Force safe defaults before creating the environment.
    try:
        scene_cfg = getattr(env_cfg, "scene", None)
        klask_cfg = getattr(scene_cfg, "klask", None)
        init_state = getattr(klask_cfg, "init_state", None)
        joint_pos = dict(getattr(init_state, "joint_pos", {}) or {})
        wildcard_default = float(joint_pos.get(".*", 0.0))
        y1 = float(joint_pos.get("ground_to_slider_1", wildcard_default))
        y2 = float(joint_pos.get("ground_to_slider_2", wildcard_default))
        if abs(y1) < 1e-9 or abs(y2) < 1e-9:
            # Avoid ambiguous regex/value mapping like ".*" + explicit joint key.
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

    env_cfg = _patch_velocity_action_cfg_offsets(env_cfg)
    env_cfg = _patch_observation_terms_for_startup(env_cfg)

    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if video else None)
    if video:
        video_kwargs = {
            "video_folder": os.path.join(output_dir, "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": int(video_length),
            "disable_logger": True,
        }
        print("[INFO] Recording videos.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

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
    if KLASK_PARAMS["action_history"] > 0:
        env = ActionHistoryWrapper(env, history_length=KLASK_PARAMS["action_history"])

    obs_noise = float(agent_cfg.get("env", {}).get("obs_noise", 0.0))
    if obs_noise > 0.0:
        env = ObservationNoiseWrapper(env, obs_noise)

    self_play = bool(agent_cfg.get("params", {}).get("config", {}).get("self_play", False))
    if self_play:
        print("[setup] opponent_mode=self_play (policy-controlled opponent)")
        env = KlaskAgentOpponentWrapper(env)
        # Keep policy action space player-only (2 dims) so checkpoints trained with
        # 2-action outputs can be used for self-play (peg2 controlled internally).
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
    if "rewards" in agent_cfg:
        env = CurriculumWrapper(env, agent_cfg["rewards"], mode="test")

    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", np.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", np.inf))
    rl_env = RlGamesVecEnvWrapper(env, device, clip_obs=clip_obs, clip_actions=clip_actions)

    # Register exactly the same env name expected by RL-Games config ("rlgpu").
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
                lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
            )
        except Exception:
            pass
    env_configurations.configurations["rlgpu"] = {
        "vecenv_type": "IsaacRlgWrapper",
        "env_creator": lambda **kwargs: rl_env,
    }

    if "terminations" in agent_cfg:
        set_terminations(rl_env, agent_cfg["terminations"])

    return rl_env


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
        # Some checkpoints are saved with tensors on a specific CUDA index (e.g., cuda:2).
        # When fewer GPUs are visible, rl_games can fail during restore with a wrapped runtime error.
        print(
            f"[warn] restore() failed ({exc}). "
            f"Falling back to direct torch.load(map_location={device})."
        )
        try:
            ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        except Exception:
            raise exc
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
    """Infer action dimension from the wrapped environment action space."""
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
    """Read params.env.clip_actions from an RL-Games config if available."""
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
    """Monkey-patch policy.get_action to clamp continuous actions."""
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


def collect_rollout_trajectories(
    *,
    env: Any,
    agent: Any,
    target_steps: int,
    min_traj_len: int,
    deterministic_expert: bool,
    deterministic_opponent: bool,
    log_every_steps: int,
) -> List[Dict[str, Any]]:
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

    obs_buffers: List[List[np.ndarray]] = [[obs_arr[i].copy()] for i in range(num_envs)]
    act_buffers: List[List[np.ndarray]] = [[] for _ in range(num_envs)]
    info_buffers: List[List[Dict[str, Any]]] = [[] for _ in range(num_envs)]
    trajectories: List[Dict[str, Any]] = []
    short_trajectories: List[Dict[str, Any]] = []
    min_len = max(1, int(min_traj_len))
    short_count = 0
    short_steps = 0
    collected = 0
    start_t = time.perf_counter()
    last_log_t = start_t
    last_log_steps = 0
    interval = max(1, int(log_every_steps))

    term_mgr = _find_termination_manager(env)
    active_term_names: List[str] = []
    if term_mgr is not None:
        active_term_names = list(getattr(term_mgr, "active_terms", []))
    tracked_term_names = list(dict.fromkeys([*active_term_names, *CORE_TERMINATION_MODES]))
    if term_mgr is None:
        print("[warn] termination_manager unavailable; falling back to env.step info for term flags.")
    elif not active_term_names:
        print("[warn] termination_manager.active_terms is empty; tracking default core terms only.")

    def build_traj(
        obs_seq: List[np.ndarray],
        act_seq: List[np.ndarray],
        info_seq: List[Dict[str, Any]],
        *,
        terminal: bool,
    ) -> Dict[str, Any]:
        traj_len = int(len(act_seq))
        infos = _slice_transition_infos(info_seq, traj_len)
        return {
            "obs": np.asarray(obs_seq, dtype=np.float32),
            "acts": np.asarray(act_seq, dtype=np.float32),
            "rews": np.asarray([0.0] * traj_len, dtype=np.float32),
            "infos": infos,
            "terminal": bool(terminal),
        }

    while simulation_app.is_running() and collected < int(target_steps):
        with torch.inference_mode():
            # Keep a copy of the pre-step observation to avoid mixing in
            # auto-reset observations for terminal transitions.
            curr_arr = ensure_2d(obs)

            obs_t = agent.obs_to_torch(obs)
            act_peg1 = agent.get_action(obs_t, is_deterministic=bool(deterministic_expert))
            act_peg1 = torch.clamp(act_peg1, -1.0, 1.0)
            stored_action = act_peg1

            if self_play_wrapper is not None and self_play_wrapper.opponent is not None:
                opp = self_play_wrapper.opponent
                opp_obs_t = opp.obs_to_torch(self_play_wrapper.opponent_obs)
                opp_action = opp.get_action(opp_obs_t, is_deterministic=bool(deterministic_opponent))
                stored_action = torch.cat([act_peg1, -opp_action], dim=1)
            stored_action = torch.clamp(stored_action, -1.0, 1.0)

            next_obs, rew, dones, step_info = env.step(act_peg1)
            if isinstance(next_obs, dict):
                next_obs = next_obs["obs"]

            done_mask = dones if torch.is_tensor(dones) else torch.as_tensor(dones, dtype=torch.bool, device=agent.device)
            done_np = to_numpy(done_mask).astype(bool).reshape(-1)
            next_arr = ensure_2d(next_obs)
            act_arr = ensure_2d(stored_action)
            rew_arr = ensure_2d(rew).reshape(-1)
            term_flags = _termination_flags_from_step(
                term_mgr=term_mgr,
                term_names=tracked_term_names,
                step_info=step_info,
                num_envs=num_envs,
            )

            for i in range(num_envs):
                act_buffers[i].append(act_arr[i].copy())
                per_env_flags = {name: bool(term_flags.get(name, np.zeros(num_envs, dtype=bool))[i]) for name in tracked_term_names}
                done_i = bool(done_np[i])
                active_terms = [name for name, value in per_env_flags.items() if value]
                step_payload = _default_info_dict(done_i)
                step_payload["env_reward"] = float(rew_arr[i])
                step_payload["termination_terms"] = active_terms if done_i else []
                step_payload["termination_mode"] = _termination_mode_from_flags(done_i, per_env_flags)
                for name in CORE_TERMINATION_MODES:
                    step_payload[f"term_{name}"] = bool(per_env_flags.get(name, False))
                info_buffers[i].append(step_payload)
                collected += 1
                if done_np[i]:
                    # ManagerBasedRLEnv performs auto-reset before returning observations.
                    # Therefore `next_arr[i]` is already from the next episode on done envs.
                    # Use pre-step observation as terminal successor to avoid artificial jumps.
                    obs_buffers[i].append(curr_arr[i].copy())
                    traj_len = int(len(act_buffers[i]))
                    if traj_len > 0:
                        tr = build_traj(obs_buffers[i], act_buffers[i], info_buffers[i], terminal=True)
                        if traj_len >= min_len:
                            trajectories.append(tr)
                        else:
                            short_trajectories.append(tr)
                            short_count += 1
                            short_steps += traj_len
                    obs_buffers[i] = [next_arr[i].copy()]
                    act_buffers[i] = []
                    info_buffers[i] = []
                else:
                    obs_buffers[i].append(next_arr[i].copy())

            now = time.perf_counter()
            should_log = (
                (collected - last_log_steps) >= interval
                or collected >= int(target_steps)
                or (now - last_log_t) >= 30.0
            )
            if should_log:
                elapsed = max(1e-6, now - start_t)
                overall_sps = collected / elapsed
                remaining = max(0, int(target_steps) - collected)
                eta_sec = remaining / max(1e-6, overall_sps)
                pct = 100.0 * collected / max(1, int(target_steps))
                print(
                    "[progress] "
                    f"steps={collected:,}/{int(target_steps):,} "
                    f"({pct:5.1f}%) "
                    f"sps={overall_sps:,.1f} "
                    f"eta={format_hms(eta_sec)} "
                    f"elapsed={format_hms(elapsed)}"
                )
                last_log_t = now
                last_log_steps = collected

            if getattr(agent, "is_rnn", False) and getattr(agent, "states", None) is not None:
                for s in agent.states:
                    s[:, done_mask, :] = 0.0
            if self_play_wrapper is not None and self_play_wrapper.opponent is not None:
                opp = self_play_wrapper.opponent
                if getattr(opp, "is_rnn", False) and getattr(opp, "states", None) is not None:
                    for s in opp.states:
                        s[:, done_mask, :] = 0.0

            obs = next_obs

    for i in range(num_envs):
        traj_len = int(len(act_buffers[i]))
        if traj_len > 0:
            tr = build_traj(obs_buffers[i], act_buffers[i], info_buffers[i], terminal=False)
            if traj_len >= min_len:
                trajectories.append(tr)
            else:
                short_trajectories.append(tr)
                short_count += 1
                short_steps += traj_len

    kept_steps = int(sum(len(t["acts"]) for t in trajectories))
    if kept_steps < int(target_steps) and short_trajectories:
        print(
            "[warn] not enough steps after filtering with "
            f"min_traj_len={min_len}; reincluding {len(short_trajectories)} "
            f"short trajectories ({short_steps} steps)."
        )
        trajectories.extend(short_trajectories)

    try:
        return trim_trajectories_to_target(trajectories, target_steps=int(target_steps))
    except RuntimeError as exc:
        kept_after = int(sum(len(t["acts"]) for t in trajectories))
        raise RuntimeError(
            f"{exc} Collected {collected} steps, kept {kept_after} after min_traj_len={min_len} filtering."
        ) from exc


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_cli_args(argv)
    _launch_simulation_app(args)
    _import_runtime_dependencies()
    if args.target_steps <= 0:
        raise ValueError("--target-steps must be > 0")
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be > 0")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if args.min_traj_len <= 0:
        raise ValueError("--min-traj-len must be > 0")
    if args.log_every_steps <= 0:
        raise ValueError("--log-every-steps must be > 0")

    if args.rl_games_config_path is not None and args.config is None:
        args.config = args.rl_games_config_path

    checkpoint = resolve_checkpoint(args)
    policy_format = resolve_policy_format(args, checkpoint)
    opponent_checkpoint, opponent_policy_format = resolve_opponent_checkpoint_and_format(args)
    if args.config is not None and not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    output_dir, traj_dir, manifest_path = prepare_dataset_output_dir(
        resolve_output_dir(args, checkpoint=checkpoint, opponent_checkpoint=opponent_checkpoint),
        force=bool(args.force),
    )
    print(f"[setup] output_dir={output_dir}")

    agent_cfg = load_agent_cfg(args.task, args.config)
    if args.force_self_play is not None:
        agent_cfg.setdefault("params", {}).setdefault("config", {})["self_play"] = bool(args.force_self_play)
        print(f"[setup] force_self_play={bool(args.force_self_play)}")
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
            video=bool(args.video),
            video_length=int(args.video_length),
            output_dir=str(output_dir),
            agent_cfg=agent_cfg,
            actuator_apply_player=not player_bypass_actuator,
            actuator_apply_opponent=not opponent_bypass_actuator,
        )

    try:
        print(f"[setup] loading expert policy: {checkpoint}")
        runner: Optional[Runner] = None
        sb3_action_dim = infer_action_dim_from_env(env)
        env_clip_actions = None
        try:
            env_clip_actions = float(agent_cfg.get("params", {}).get("env", {}).get("clip_actions", np.nan))
            if not np.isfinite(env_clip_actions):
                env_clip_actions = None
        except Exception:
            env_clip_actions = None
        if policy_format == POLICY_FORMAT_SB3_PPO:
            print(f"[setup] sb3 expected env action_dim={sb3_action_dim}")
        if policy_format == POLICY_FORMAT_RL_GAMES:
            runner, agent = build_runner_player(
                agent_cfg=agent_cfg,
                checkpoint=checkpoint,
                num_envs=int(args.num_envs),
                device=args.device,
            )
        else:
            agent = build_sb3_player(
                checkpoint=checkpoint,
                device=args.device,
                action_dim=sb3_action_dim,
            )
        if policy_format == POLICY_FORMAT_RL_GAMES:
            expert_action_clip = get_rl_games_action_clip(task=args.task, config_path=args.config)
            apply_policy_action_clip(policy=agent, clip_value=expert_action_clip, label="expert")

        if bool(agent_cfg.get("params", {}).get("config", {}).get("self_play", False)):
            if opponent_checkpoint is not None:
                opponent_task = args.opponent_rl_games_task or args.task
                opponent_cfg_path = args.opponent_rl_games_config_path
                print(f"[setup] loading fixed opponent policy: {opponent_checkpoint}")
                opponent = build_policy_player(
                    policy_format=str(opponent_policy_format),
                    checkpoint=opponent_checkpoint,
                    device=args.device,
                    num_envs=int(args.num_envs),
                    action_dim=sb3_action_dim,
                    rl_games_task=str(opponent_task),
                    rl_games_config_path=opponent_cfg_path,
                )
                if opponent_policy_format == POLICY_FORMAT_RL_GAMES:
                    opponent_action_clip = get_rl_games_action_clip(
                        task=str(opponent_task),
                        config_path=opponent_cfg_path,
                    )
                    apply_policy_action_clip(
                        policy=opponent,
                        clip_value=opponent_action_clip,
                        label="opponent",
                    )
                    if (
                        env_clip_actions is not None
                        and opponent_action_clip is not None
                        and abs(float(env_clip_actions) - float(opponent_action_clip)) > 1e-8
                    ):
                        print(
                            "[warn] clip_actions mismatch: "
                            f"env/main={env_clip_actions} vs opponent_cfg={opponent_action_clip}. "
                            "Using opponent action clipping to keep fixed-opponent behavior stable."
                        )
            else:
                if policy_format == POLICY_FORMAT_RL_GAMES:
                    if runner is None:
                        raise RuntimeError("RL-Games runner was not initialized.")
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
            f"[rollout] collecting {int(args.target_steps):,} steps "
            f"(num_envs={int(args.num_envs)})"
        )
        trajectories = collect_rollout_trajectories(
            env=env,
            agent=agent,
            target_steps=int(args.target_steps),
            min_traj_len=int(args.min_traj_len),
            deterministic_expert=bool(args.deterministic_expert),
            deterministic_opponent=bool(args.deterministic_opponent),
            log_every_steps=int(args.log_every_steps),
        )

        chunk_paths = write_trajectory_chunks(
            trajectories=trajectories,
            output_dir=str(traj_dir),
            chunk_size=int(args.chunk_size),
        )
        total_steps = int(sum(len(t["acts"]) for t in trajectories))
        manifest = {
            "created_at": datetime.datetime.now().isoformat(),
            "dataset_dir": str(output_dir),
            "trajectory_dir": str(traj_dir),
            "num_files": int(len(chunk_paths)),
            "num_trajectories": int(len(trajectories)),
            "actual_steps": total_steps,
            "target_steps": int(args.target_steps),
            "expert_id": str(args.expert_id),
            "checkpoint": checkpoint,
            "policy_format": policy_format,
            "opponent_checkpoint": opponent_checkpoint,
            "opponent_policy_format": opponent_policy_format,
            "config": args.config,
            "task": args.task,
            "num_envs": int(args.num_envs),
            "device": args.device,
            "self_play": bool(agent_cfg.get("params", {}).get("config", {}).get("self_play", False)),
            "demo_source_format": "sim_native",
        }
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)

        print(
            f"[done] steps={total_steps:,}, trajectories={len(trajectories):,}, "
            f"files={len(chunk_paths)}"
        )
        print(output_dir)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        try:
            env.close()
        except Exception:
            pass
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
