"""Standalone BC baseline training entrypoint with release-facing config support."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

from config_utils import default_config_path, merge_config_file, resolve_repo_path


DEMO_SOURCE_HUMAN_RAW = "human_raw"
DEMO_SOURCE_SIM_NATIVE = "sim_native"


DEFAULT_CONFIG_PATH = default_config_path("bc_baseline.json")

DEFAULTS: Dict[str, Any] = {
    "demo_base": "data/human/trajectories",
    "demo_source_format": DEMO_SOURCE_HUMAN_RAW,
    "save_root": "trained_models",
    "run_tag": "BC1",
    "seed": 42,
    "device": "cuda:0",
    "num_envs": 1,
    "batch_size": 128,
    "n_epochs": 20,
    "learning_rate": 6.72e-4,
    "l2_weight": 1.794e-4,
    "ent_weight": 3.1314e-6,
    "net_arch": [64, 64],
    "augment": True,
    "min_len": 2,
    "strip_sim_action_history": True,
    "zero_mask_velocity_obs_sim": False,
    "zero_mask_velocity_obs_human": False,
    "val_split_fraction": 0.2,
    "val_eval_batch_size": 4096,
    "wandb_project": "klask_bc",
    "wandb_entity": None,
    "wandb_mode": "disabled",
    "enable_wandb_bc_loss_logging": True,
    "use_optuna": False,
    "optuna_n_trials": 30,
    "optuna_timeout_sec": 3600,
    "optuna_n_startup_trials": 5,
    "optuna_n_warmup_steps": 3,
    "ppo_learning_rate": 1e-6,
    "ppo_n_steps": 128,
    "ppo_batch_size": 128,
    "ppo_ent_coef": 0.02,
    "ppo_clip_range": 0.2,
    "policy_log_std_init": 0.0,
    "policy_ortho_init": False,
}

BASE_OBS_DIM = 12
MASK_ACTION_LIKE_OBS_START = 8
MASK_ACTION_LIKE_OBS_END = 12


def _resolve_config_paths(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    out["demo_base"] = resolve_repo_path(out["demo_base"])
    out["save_root"] = resolve_repo_path(out["save_root"])
    return out


def build_arg_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the standalone KlaskTron BC baseline from a curated config."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--print-config", action="store_true", help="Print resolved config and exit.")
    parser.add_argument("--demo-base", type=str, default=defaults["demo_base"])
    parser.add_argument(
        "--demo-source-format",
        choices=[DEMO_SOURCE_HUMAN_RAW, DEMO_SOURCE_SIM_NATIVE],
        default=defaults["demo_source_format"],
    )
    parser.add_argument("--save-root", type=str, default=defaults["save_root"])
    parser.add_argument("--run-tag", type=str, default=defaults["run_tag"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--device", type=str, default=defaults["device"])
    parser.add_argument("--num-envs", type=int, default=defaults["num_envs"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--n-epochs", type=int, default=defaults["n_epochs"])
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--l2-weight", type=float, default=defaults["l2_weight"])
    parser.add_argument("--ent-weight", type=float, default=defaults["ent_weight"])
    parser.add_argument("--net-arch", nargs="+", type=int, default=list(defaults["net_arch"]))
    parser.add_argument("--min-len", type=int, default=defaults["min_len"])
    parser.add_argument(
        "--strip-sim-action-history",
        action=argparse.BooleanOptionalAction,
        default=defaults["strip_sim_action_history"],
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=defaults["augment"],
    )
    parser.add_argument(
        "--zero-mask-velocity-obs-sim",
        action=argparse.BooleanOptionalAction,
        default=defaults["zero_mask_velocity_obs_sim"],
    )
    parser.add_argument(
        "--zero-mask-velocity-obs-human",
        action=argparse.BooleanOptionalAction,
        default=defaults["zero_mask_velocity_obs_human"],
    )
    parser.add_argument(
        "--use-optuna",
        action=argparse.BooleanOptionalAction,
        default=defaults["use_optuna"],
        help="Run Optuna search before the final BC training pass.",
    )
    parser.add_argument("--optuna-n-trials", type=int, default=defaults["optuna_n_trials"])
    parser.add_argument("--wandb-project", type=str, default=defaults["wandb_project"])
    parser.add_argument("--wandb-mode", type=str, default=defaults["wandb_mode"])
    return parser


def build_runtime_config() -> tuple[argparse.Namespace, Dict[str, Any]]:
    """Resolve config file defaults, then apply explicit CLI overrides."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args()

    merged = merge_config_file(DEFAULTS, pre_args.config, label="BC", allow_unknown_keys=False)
    merged = _resolve_config_paths({k: v for k, v in merged.items() if not k.startswith("_")})
    args = build_arg_parser(merged).parse_args()

    runtime = dict(merged)
    runtime["demo_base"] = resolve_repo_path(args.demo_base)
    runtime["demo_source_format"] = args.demo_source_format
    runtime["save_root"] = resolve_repo_path(args.save_root)
    runtime["run_tag"] = args.run_tag
    runtime["seed"] = int(args.seed)
    runtime["device"] = args.device
    runtime["num_envs"] = int(args.num_envs)
    runtime["batch_size"] = int(args.batch_size)
    runtime["n_epochs"] = int(args.n_epochs)
    runtime["learning_rate"] = float(args.learning_rate)
    runtime["l2_weight"] = float(args.l2_weight)
    runtime["ent_weight"] = float(args.ent_weight)
    runtime["net_arch"] = [int(v) for v in args.net_arch]
    runtime["min_len"] = int(args.min_len)
    runtime["strip_sim_action_history"] = bool(args.strip_sim_action_history)
    runtime["augment"] = bool(args.augment)
    runtime["zero_mask_velocity_obs_sim"] = bool(args.zero_mask_velocity_obs_sim)
    runtime["zero_mask_velocity_obs_human"] = bool(args.zero_mask_velocity_obs_human)
    runtime["use_optuna"] = bool(args.use_optuna)
    runtime["optuna_n_trials"] = int(args.optuna_n_trials)
    runtime["wandb_project"] = args.wandb_project
    runtime["wandb_mode"] = args.wandb_mode
    if args.print_config:
        print(json.dumps(runtime, indent=2))
        raise SystemExit(0)
    return args, runtime


ARGS, CONFIG = build_runtime_config()

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import numpy as np
import torch
from imitation.algorithms import bc
from imitation.data.types import Trajectory
from optuna.pruners import MedianPruner
from stable_baselines3 import PPO
import optuna

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from my_isaaclab_tasks.manager_based.klask.env_cfg import EnvCfg
from demo_dataset import augment_with_axis_mirrors, load_dataset_with_optional_augmentation

try:
    import wandb
except ImportError:
    wandb = None


def resolve_device(requested_device: str) -> str:
    """Resolve the requested torch device against visible CUDA devices."""
    req = str(requested_device).strip().lower()
    if req == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda:0"
        return "cpu"

    if not req.startswith("cuda"):
        return req

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        print(f"[warn] requested {requested_device}, but CUDA unavailable. Falling back to cpu.")
        return "cpu"

    idx = 0
    if ":" in req:
        try:
            idx = int(req.split(":", 1)[1])
        except ValueError:
            idx = 0
    idx = min(max(idx, 0), torch.cuda.device_count() - 1)
    return f"cuda:{idx}"


def make_env(num_envs: int, seed: int) -> Sb3VecEnvWrapper:
    """Construct the Isaac Klask environment wrapped for SB3."""
    cfg = EnvCfg()
    cfg.scene.num_envs = int(num_envs)
    cfg.seed = int(seed)
    base_env = gym.make("Isaac-Klask-v0", cfg=cfg)
    return Sb3VecEnvWrapper(base_env)


def strip_sim_native_action_history(
    trajectories: List[Trajectory], keep_obs_dim: int = BASE_OBS_DIM
) -> Tuple[List[Trajectory], int, List[int]]:
    """Drop appended action-history dims from sim-native trajectories."""
    out: List[Trajectory] = []
    stripped = 0
    seen_dims = set()

    for tr in trajectories:
        if isinstance(tr.obs, dict):
            out.append(tr)
            continue

        obs_arr = np.asarray(tr.obs, dtype=np.float32)
        if obs_arr.ndim != 2:
            out.append(tr)
            continue

        seen_dims.add(int(obs_arr.shape[1]))
        if obs_arr.shape[1] > int(keep_obs_dim):
            obs_arr = obs_arr[:, : int(keep_obs_dim)]
            stripped += 1

        out.append(
            Trajectory(
                obs=obs_arr,
                acts=np.asarray(tr.acts, dtype=np.float32),
                infos=tr.infos,
                terminal=tr.terminal,
            )
        )

    return out, stripped, sorted(seen_dims)


def zero_mask_observation_channels(
    trajectories: List[Trajectory],
    *,
    start: int,
    end: int,
) -> List[Trajectory]:
    """Set observation channels [start:end] to zero for all non-dict trajectories."""
    out: List[Trajectory] = []
    for tr in trajectories:
        if isinstance(tr.obs, dict):
            out.append(tr)
            continue
        obs_arr = np.asarray(tr.obs, dtype=np.float32).copy()
        if obs_arr.ndim == 2 and obs_arr.shape[1] >= int(end):
            obs_arr[:, int(start) : int(end)] = 0.0
        out.append(
            Trajectory(
                obs=obs_arr,
                acts=np.asarray(tr.acts, dtype=np.float32),
                infos=tr.infos,
                terminal=tr.terminal,
            )
        )
    return out


def split_train_val_trajectories(
    trajectories: List[Trajectory],
    group_ids: Sequence[Hashable],
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[List[Trajectory], List[Trajectory]]:
    """Split trajectories into train/validation sets by group id."""
    n = len(trajectories)
    if n <= 1:
        return trajectories, trajectories

    if len(group_ids) != n:
        raise ValueError(
            f"group_ids length {len(group_ids)} does not match trajectories length {n}"
        )

    group_to_indices: Dict[Hashable, List[int]] = {}
    for idx, gid in enumerate(group_ids):
        group_to_indices.setdefault(gid, []).append(idx)

    groups = list(group_to_indices.keys())
    n_groups = len(groups)
    if n_groups <= 1:
        return trajectories, trajectories

    n_val_groups = max(1, int(round(n_groups * val_fraction)))
    n_val_groups = min(n_val_groups, n_groups - 1)
    perm = np.random.default_rng(seed).permutation(n_groups)
    val_group_set = {groups[i] for i in perm[:n_val_groups]}

    train: List[Trajectory] = []
    val: List[Trajectory] = []
    for idx, gid in enumerate(group_ids):
        if gid in val_group_set:
            val.append(trajectories[idx])
        else:
            train.append(trajectories[idx])
    return train, val


def flatten_trajectories(trajs: List[Trajectory]) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten trajectories into aligned obs/act arrays."""
    obs_parts = []
    act_parts = []
    for tr in trajs:
        obs_parts.append(np.asarray(tr.obs[:-1], dtype=np.float32))
        act_parts.append(np.asarray(tr.acts, dtype=np.float32))
    return np.concatenate(obs_parts, axis=0), np.concatenate(act_parts, axis=0)


def compute_bc_loss_chunked(
    bc_trainer: bc.BC,
    obs_np: np.ndarray,
    acts_np: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    """Compute mean BC loss on a dataset using chunked forward passes."""
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for start in range(0, obs_np.shape[0], batch_size):
            end = min(start + batch_size, obs_np.shape[0])
            obs_th = torch.as_tensor(obs_np[start:end], dtype=torch.float32, device=device)
            acts_th = torch.as_tensor(acts_np[start:end], dtype=torch.float32, device=device)
            stats = bc_trainer.loss_calculator(bc_trainer.policy, obs_th, acts_th)
            batch_loss = stats.loss.item() if hasattr(stats.loss, "item") else float(stats.loss)
            count = end - start
            total_loss += float(batch_loss) * count
            total_count += count
    return total_loss / max(total_count, 1)


def create_bc_objective(
    venv: Sb3VecEnvWrapper,
    expert_trajs: List[Trajectory],
    device: str,
    seed: int,
) -> Any:
    """Build the Optuna objective for BC hyperparameter search."""

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        n_epochs = trial.suggest_int("n_epochs", 4, 20)
        l2_weight = trial.suggest_float("l2_weight", 1e-6, 1e-2, log=True)
        ent_weight = trial.suggest_float("ent_weight", 1e-6, 1e-2, log=True)
        net_arch_choice = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
        net_arch_map = {
            "small": [64, 64],
            "medium": [128, 128],
            "large": [256, 256],
        }
        net_arch = net_arch_map[net_arch_choice]

        policy_name = (
            "MultiInputPolicy"
            if isinstance(venv.observation_space, gym.spaces.Dict)
            else "MlpPolicy"
        )
        learner = PPO(
            policy_name,
            venv,
            learning_rate=float(CONFIG["ppo_learning_rate"]),
            n_steps=int(CONFIG["ppo_n_steps"]),
            batch_size=int(CONFIG["ppo_batch_size"]),
            ent_coef=float(CONFIG["ppo_ent_coef"]),
            clip_range=float(CONFIG["ppo_clip_range"]),
            policy_kwargs=dict(
                log_std_init=float(CONFIG["policy_log_std_init"]),
                ortho_init=bool(CONFIG["policy_ortho_init"]),
                net_arch=net_arch,
            ),
            verbose=0,
            device=device,
            seed=seed,
        )

        bc_trainer = bc.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            demonstrations=expert_trajs,
            policy=learner.policy,
            device=device,
            rng=np.random.default_rng(seed),
            batch_size=batch_size,
            optimizer_kwargs={"lr": lr},
            l2_weight=l2_weight,
            ent_weight=ent_weight,
        )

        n_steps_total = sum(len(tr.acts) for tr in expert_trajs)
        batches_per_epoch = max(1, n_steps_total // batch_size)
        bc_trainer.train(n_batches=n_epochs * batches_per_epoch)

        obs_np, acts_np = flatten_trajectories(expert_trajs)
        return compute_bc_loss_chunked(
            bc_trainer,
            obs_np,
            acts_np,
            torch.device(device),
            batch_size=int(CONFIG["val_eval_batch_size"]),
        )

    return objective


def main() -> None:
    np.random.seed(int(CONFIG["seed"]))
    torch.manual_seed(int(CONFIG["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(CONFIG["seed"]))

    device = resolve_device(str(CONFIG["device"]))
    venv = make_env(num_envs=int(CONFIG["num_envs"]), seed=int(CONFIG["seed"]))
    wandb_run = None

    try:
        print("[dataset] loading demonstrations...")
        raw_trajectories, group_ids, dataset_stats = load_dataset_with_optional_augmentation(
            demo_base=str(CONFIG["demo_base"]),
            action_space=venv.action_space,
            source_format=str(CONFIG["demo_source_format"]),
            min_len=int(CONFIG["min_len"]),
            augment_mirrors=False,
        )
        print(
            "[dataset] files={n_files} raw_traj={n_traj_raw} raw_steps={n_steps_raw}".format(
                **dataset_stats.to_dict()
            )
        )

        demo_obs_dim: Optional[int] = None
        if (
            CONFIG["demo_source_format"] == DEMO_SOURCE_SIM_NATIVE
            and bool(CONFIG["strip_sim_action_history"])
        ):
            raw_trajectories, stripped_count, seen_dims = strip_sim_native_action_history(
                raw_trajectories,
                keep_obs_dim=BASE_OBS_DIM,
            )
            print(
                "[dataset] strip_sim_action_history=True "
                f"seen_obs_dims={seen_dims} stripped_traj={stripped_count}"
            )
        if (
            CONFIG["demo_source_format"] == DEMO_SOURCE_SIM_NATIVE
            and bool(CONFIG["zero_mask_velocity_obs_sim"])
        ):
            raw_trajectories = zero_mask_observation_channels(
                raw_trajectories,
                start=MASK_ACTION_LIKE_OBS_START,
                end=MASK_ACTION_LIKE_OBS_END,
            )
            print(
                "[dataset] zero-masked sim_native observation channels "
                f"[{MASK_ACTION_LIKE_OBS_START}:{MASK_ACTION_LIKE_OBS_END}]."
            )

        if CONFIG["demo_source_format"] == DEMO_SOURCE_HUMAN_RAW and bool(
            CONFIG["zero_mask_velocity_obs_human"]
        ):
            raw_trajectories = zero_mask_observation_channels(
                raw_trajectories,
                start=MASK_ACTION_LIKE_OBS_START,
                end=MASK_ACTION_LIKE_OBS_END,
            )
            print(
                "[dataset] zero-masked human observation channels "
                f"[{MASK_ACTION_LIKE_OBS_START}:{MASK_ACTION_LIKE_OBS_END}]."
            )
        elif CONFIG["demo_source_format"] == DEMO_SOURCE_HUMAN_RAW:
            print(
                "[dataset] keeping original human observation channels "
                f"[{MASK_ACTION_LIKE_OBS_START}:{MASK_ACTION_LIKE_OBS_END}] (no masking)."
            )

        train_raw_trajs, val_trajs = split_train_val_trajectories(
            raw_trajectories,
            group_ids=group_ids,
            val_fraction=float(CONFIG["val_split_fraction"]),
            seed=int(CONFIG["seed"]),
        )
        if bool(CONFIG["augment"]):
            train_trajs, train_aug_n = augment_with_axis_mirrors(train_raw_trajs)
        else:
            train_trajs = list(train_raw_trajs)
            train_aug_n = 0

        train_raw_steps = sum(len(tr.acts) for tr in train_raw_trajs)
        train_steps = sum(len(tr.acts) for tr in train_trajs)
        val_steps = sum(len(tr.acts) for tr in val_trajs)
        print(
            f"[BC split] train(raw): {len(train_raw_trajs)} traj, {train_raw_steps} steps | "
            f"train(used): {len(train_trajs)} traj, {train_steps} steps | "
            f"val(raw holdout): {len(val_trajs)} traj, {val_steps} steps"
        )
        if bool(CONFIG["augment"]):
            print(
                f"[BC split] Augmentation applied to train only: +{train_aug_n} mirrored trajectories "
                "(x, y, xy). Validation is unaugmented."
            )
        else:
            print("[BC split] Augmentation disabled. Validation is unaugmented.")

        if train_trajs and not isinstance(train_trajs[0].obs, dict):
            demo_obs_dim = int(np.asarray(train_trajs[0].obs).shape[-1])
            env_obs_dim = int(np.prod(venv.observation_space.shape))
            if demo_obs_dim != env_obs_dim:
                raise ValueError(
                    "Observation dimension mismatch after preprocessing: "
                    f"demo_obs_dim={demo_obs_dim}, env_obs_dim={env_obs_dim}."
                )

        if bool(CONFIG["use_optuna"]):
            print(
                f"[optuna] running {int(CONFIG['optuna_n_trials'])} BC trials "
                f"with timeout={int(CONFIG['optuna_timeout_sec'])}s"
            )
            study = optuna.create_study(
                direction="minimize",
                pruner=MedianPruner(
                    n_startup_trials=int(CONFIG["optuna_n_startup_trials"]),
                    n_warmup_steps=int(CONFIG["optuna_n_warmup_steps"]),
                ),
            )
            study.optimize(
                create_bc_objective(
                    venv,
                    train_trajs,
                    device=device,
                    seed=int(CONFIG["seed"]),
                ),
                n_trials=int(CONFIG["optuna_n_trials"]),
                timeout=int(CONFIG["optuna_timeout_sec"]),
            )
            selected_hyperparams = {
                "batch_size": int(study.best_params["batch_size"]),
                "n_epochs": int(study.best_params["n_epochs"]),
                "learning_rate": float(study.best_params["learning_rate"]),
                "l2_weight": float(study.best_params["l2_weight"]),
                "ent_weight": float(study.best_params["ent_weight"]),
                "net_arch": {
                    "small": [64, 64],
                    "medium": [128, 128],
                    "large": [256, 256],
                }[study.best_params["net_arch"]],
            }
        else:
            selected_hyperparams = {
                "batch_size": int(CONFIG["batch_size"]),
                "n_epochs": int(CONFIG["n_epochs"]),
                "learning_rate": float(CONFIG["learning_rate"]),
                "l2_weight": float(CONFIG["l2_weight"]),
                "ent_weight": float(CONFIG["ent_weight"]),
                "net_arch": list(CONFIG["net_arch"]),
            }

        policy_name = (
            "MultiInputPolicy"
            if isinstance(venv.observation_space, gym.spaces.Dict)
            else "MlpPolicy"
        )
        learner = PPO(
            policy_name,
            venv,
            learning_rate=float(CONFIG["ppo_learning_rate"]),
            n_steps=int(CONFIG["ppo_n_steps"]),
            batch_size=int(CONFIG["ppo_batch_size"]),
            ent_coef=float(CONFIG["ppo_ent_coef"]),
            clip_range=float(CONFIG["ppo_clip_range"]),
            policy_kwargs=dict(
                log_std_init=float(CONFIG["policy_log_std_init"]),
                ortho_init=bool(CONFIG["policy_ortho_init"]),
                net_arch=list(selected_hyperparams["net_arch"]),
            ),
            verbose=2,
            device=device,
            seed=int(CONFIG["seed"]),
        )

        bc_trainer = bc.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            demonstrations=train_trajs,
            policy=learner.policy,
            device=device,
            rng=np.random.default_rng(int(CONFIG["seed"])),
            batch_size=int(selected_hyperparams["batch_size"]),
            optimizer_kwargs={"lr": float(selected_hyperparams["learning_rate"])},
            l2_weight=float(selected_hyperparams["l2_weight"]),
            ent_weight=float(selected_hyperparams["ent_weight"]),
        )

        # Wrap loss calculator to ensure tensors are on the correct device.
        # Some imitation/SB3 version combinations fail to move demo batches to GPU.
        _original_loss_calc = bc_trainer.loss_calculator
        _policy_device = torch.device(device)

        def _device_safe_loss_calc(policy, obs, acts):
            if torch.is_tensor(obs):
                obs = obs.to(_policy_device)
            acts = acts.to(_policy_device)
            return _original_loss_calc(policy, obs, acts)

        bc_trainer.loss_calculator = _device_safe_loss_calc

        train_obs_np, train_acts_np = flatten_trajectories(train_trajs)
        val_obs_np, val_acts_np = flatten_trajectories(val_trajs)

        if bool(CONFIG["enable_wandb_bc_loss_logging"]):
            if wandb is None:
                print("[warn] wandb is not installed; skipping BC train/val loss logging.")
            elif str(CONFIG["wandb_mode"]) == "disabled":
                print("[info] wandb logging disabled by config.")
            else:
                wandb_run = wandb.init(
                    project=str(CONFIG["wandb_project"]),
                    entity=CONFIG["wandb_entity"],
                    mode=str(CONFIG["wandb_mode"]),
                    name=f"klask_bc_loss_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "runtime_config": CONFIG,
                        "selected_hyperparams": selected_hyperparams,
                    },
                )

        print(
            "[train] starting BC: "
            f"epochs={selected_hyperparams['n_epochs']}, "
            f"batch_size={selected_hyperparams['batch_size']}, "
            f"lr={selected_hyperparams['learning_rate']}, "
            f"l2={selected_hyperparams['l2_weight']}, "
            f"ent={selected_hyperparams['ent_weight']}"
        )

        metrics = {
            "train_loss": None,
            "val_loss": None,
        }

        def _log_train_val_losses(epoch_idx: int) -> None:
            train_loss = compute_bc_loss_chunked(
                bc_trainer,
                train_obs_np,
                train_acts_np,
                torch.device(device),
                batch_size=int(CONFIG["val_eval_batch_size"]),
            )
            val_loss = compute_bc_loss_chunked(
                bc_trainer,
                val_obs_np,
                val_acts_np,
                torch.device(device),
                batch_size=int(CONFIG["val_eval_batch_size"]),
            )
            metrics["train_loss"] = float(train_loss)
            metrics["val_loss"] = float(val_loss)
            print(
                f"[BC loss] epoch={epoch_idx:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
            )
            if wandb_run is not None:
                wandb.log(
                    {
                        "bc/epoch": int(epoch_idx),
                        "bc/train_loss": float(train_loss),
                        "bc/val_loss": float(val_loss),
                    },
                    step=int(epoch_idx),
                )

        epoch_counter = {"idx": 0}

        def _on_epoch_end() -> None:
            epoch_counter["idx"] += 1
            _log_train_val_losses(epoch_counter["idx"])

        t0 = time.time()
        _log_train_val_losses(epoch_counter["idx"])
        bc_trainer.train(
            n_epochs=int(selected_hyperparams["n_epochs"]),
            on_epoch_end=_on_epoch_end,
        )
        elapsed = time.time() - t0
        if wandb_run is not None:
            wandb_run.finish()
            wandb_run = None
        print(f"[train] done in {elapsed:.1f}s")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"klask_bc_{timestamp}_{CONFIG['run_tag']}"
        run_dir = f"{CONFIG['save_root']}/{run_name}"
        policy_path = f"{run_dir}/ppo_policy_bc_final.zip"

        os.makedirs(run_dir, exist_ok=True)
        learner.save(policy_path)

        config_payload = {
            "config": CONFIG,
            "config_file": ARGS.config,
            "selected_hyperparams": selected_hyperparams,
            "dataset_stats": dataset_stats.to_dict(),
            "run_name": run_name,
            "run_dir": run_dir,
            "created_at": datetime.datetime.now().isoformat(),
            "training_elapsed_sec": float(elapsed),
            "final_policy_path": policy_path,
            "effective_demo_obs_dim": int(demo_obs_dim) if demo_obs_dim is not None else None,
        }
        with open(f"{run_dir}/config.json", "w", encoding="utf-8") as fh:
            json.dump(config_payload, fh, indent=2)

        with open(f"{run_dir}/train_stats.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "training_elapsed_sec": float(elapsed),
                    "train_raw_steps": int(train_raw_steps),
                    "dataset_steps_used": int(train_steps),
                    "val_steps": int(val_steps),
                    "final_train_loss": metrics["train_loss"],
                    "final_val_loss": metrics["val_loss"],
                },
                fh,
                indent=2,
            )

        print(f"[out] {policy_path}")
        print(f"[out] {run_dir}/config.json")
        print(f"[out] {run_dir}/train_stats.json")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        venv.close()


if __name__ == "__main__":
    # Isaac Sim's shutdown path swallows the process exit status, so a crash
    # here would otherwise be reported as success to any caller (CI, the
    # experiment launchers, shell scripts). Flush and exit explicitly.
    import os as _os
    import sys as _sys
    import traceback as _tb

    try:
        main()
    except BaseException:
        _tb.print_exc()
        _sys.stdout.flush()
        _sys.stderr.flush()
        _os._exit(1)
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(0)
