"""Configurable BC training entrypoint for data-scaling experiments."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import numpy as np
import torch
from imitation.algorithms import bc
from imitation.data.types import Trajectory
from stable_baselines3 import PPO

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from my_isaaclab_tasks.manager_based.klask.env_cfg import EnvCfg

from demo_dataset import (
    DEMO_SOURCE_HUMAN_RAW,
    DEMO_SOURCE_SIM_NATIVE,
    augment_with_axis_mirrors,
    load_dataset_with_optional_augmentation,
)

try:
    import wandb
except ImportError:
    wandb = None


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = Path(os.environ.get("KLASKTRON_DATA_DIR", _REPO_ROOT / "data"))
_TRAINED_MODELS_DIR = Path(
    os.environ.get("KLASKTRON_TRAINED_MODELS_DIR", _REPO_ROOT / "trained_models")
)

DEFAULTS: Dict[str, Any] = {
    "demo_base": str(_DATA_DIR / "expert_trajectories_cleaned"),
    "demo_source_format": DEMO_SOURCE_HUMAN_RAW,
    "save_root": str(_TRAINED_MODELS_DIR),
    "run_tag": "BC_SWEEP",
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
    "zero_mask_velocity_obs_human": True,
}
BASE_OBS_DIM = 12
WANDB_PROJECT = "klask_bc"
WANDB_ENTITY = None
WANDB_MODE = "online"  # "online" | "offline" | "disabled"
ENABLE_WANDB_BC_LOSS_LOGGING = True
VAL_SPLIT_FRACTION = 0.2
VAL_EVAL_BATCH_SIZE = 4096
MASK_ACTION_LIKE_OBS_START = 8
MASK_ACTION_LIKE_OBS_END = 12


def resolve_device(requested_device: str) -> str:
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
    cfg = EnvCfg()
    cfg.scene.num_envs = int(num_envs)
    cfg.seed = int(seed)
    base_env = gym.make("Isaac-Klask-v0", cfg=cfg)
    return Sb3VecEnvWrapper(base_env)


def strip_sim_native_action_history(
    trajectories: List[Trajectory], keep_obs_dim: int = BASE_OBS_DIM
) -> Tuple[List[Trajectory], int, List[int]]:
    """Drop appended action-history dims and keep only the recent physical state."""
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


def resolve_augment_setting(
    source_format: str,
    cli_augment_value: Optional[bool],
) -> bool:
    """Choose augmentation default by source unless explicitly overridden."""
    if cli_augment_value is not None:
        return bool(cli_augment_value)
    if source_format == DEMO_SOURCE_SIM_NATIVE:
        return False
    return True


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
    batch_size: int = 4096,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BC policy on configurable dataset.")
    parser.add_argument("--demo-base", type=str, default=DEFAULTS["demo_base"])
    parser.add_argument(
        "--demo-source-format",
        choices=[DEMO_SOURCE_HUMAN_RAW, DEMO_SOURCE_SIM_NATIVE],
        default=DEFAULTS["demo_source_format"],
    )
    parser.add_argument("--save-root", type=str, default=DEFAULTS["save_root"])
    parser.add_argument("--run-tag", type=str, default=DEFAULTS["run_tag"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--device", type=str, default=DEFAULTS["device"])
    parser.add_argument("--num-envs", type=int, default=DEFAULTS["num_envs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--n-epochs", type=int, default=DEFAULTS["n_epochs"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--l2-weight", type=float, default=DEFAULTS["l2_weight"])
    parser.add_argument("--ent-weight", type=float, default=DEFAULTS["ent_weight"])
    parser.add_argument("--min-len", type=int, default=DEFAULTS["min_len"])
    parser.add_argument(
        "--strip-sim-action-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For sim_native demos, keep only the first 12 observation dims "
            "(ball/peg positions + velocities) and drop appended action history."
        ),
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable x/y/xy mirrored data augmentation. If unset: enabled for human_raw, "
            "disabled for sim_native."
        ),
    )
    parser.add_argument(
        "--zero-mask-velocity-obs-sim",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For sim_native demos, optionally zero observation channels [8:12] "
            "(peg velocity states) to match no-velocity-input ablations."
        ),
    )
    parser.add_argument(
        "--zero-mask-velocity-obs-human",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["zero_mask_velocity_obs_human"],
        help=(
            "For human_raw demos, optionally zero observation channels [8:12] "
            "(peg velocity states) to avoid BC target leakage."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    augment_enabled = resolve_augment_setting(args.demo_source_format, args.augment)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    venv = make_env(num_envs=args.num_envs, seed=args.seed)
    wandb_run = None

    try:
        print("[dataset] loading demonstrations...")
        if args.augment is None:
            print(
                f"[dataset] --augment not set; defaulting to augment={augment_enabled} "
                f"for source={args.demo_source_format}."
            )
        else:
            print(f"[dataset] using explicit augment={augment_enabled}.")
        raw_trajectories, group_ids, dataset_stats = load_dataset_with_optional_augmentation(
            demo_base=args.demo_base,
            action_space=venv.action_space,
            source_format=args.demo_source_format,
            min_len=args.min_len,
            augment_mirrors=False,
        )
        print(
            "[dataset] files={n_files} raw_traj={n_traj_raw} raw_steps={n_steps_raw}".format(
                **dataset_stats.to_dict()
            )
        )

        demo_obs_dim = None
        if (
            args.demo_source_format == DEMO_SOURCE_SIM_NATIVE
            and bool(args.strip_sim_action_history)
        ):
            raw_trajectories, stripped_count, seen_dims = strip_sim_native_action_history(
                raw_trajectories, keep_obs_dim=BASE_OBS_DIM
            )
            print(
                "[dataset] strip_sim_action_history=True "
                f"seen_obs_dims={seen_dims} stripped_traj={stripped_count}"
            )
        if (
            args.demo_source_format == DEMO_SOURCE_SIM_NATIVE
            and bool(args.zero_mask_velocity_obs_sim)
        ):
            raw_trajectories = zero_mask_observation_channels(
                raw_trajectories,
                start=MASK_ACTION_LIKE_OBS_START,
                end=MASK_ACTION_LIKE_OBS_END,
            )
            print(
                "[dataset] zero-masked sim_native observation channels "
                f"[{MASK_ACTION_LIKE_OBS_START}:{MASK_ACTION_LIKE_OBS_END}] "
                "for velocity-state ablation."
            )

        if args.demo_source_format == DEMO_SOURCE_HUMAN_RAW and bool(
            args.zero_mask_velocity_obs_human
        ):
            raw_trajectories = zero_mask_observation_channels(
                raw_trajectories,
                start=MASK_ACTION_LIKE_OBS_START,
                end=MASK_ACTION_LIKE_OBS_END,
            )
            print(
                "[dataset] zero-masked human observation channels "
                f"[{MASK_ACTION_LIKE_OBS_START}:{MASK_ACTION_LIKE_OBS_END}] to avoid BC target leakage."
            )
        elif args.demo_source_format == DEMO_SOURCE_HUMAN_RAW:
            print(
                "[dataset] keeping original human observation channels "
                f"[{MASK_ACTION_LIKE_OBS_START}:{MASK_ACTION_LIKE_OBS_END}] (no masking)."
            )

        train_raw_trajs, val_trajs = split_train_val_trajectories(
            raw_trajectories,
            group_ids=group_ids,
            val_fraction=VAL_SPLIT_FRACTION,
            seed=args.seed,
        )
        if augment_enabled:
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
        if augment_enabled:
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
                    f"Observation dimension mismatch after preprocessing: "
                    f"demo_obs_dim={demo_obs_dim}, env_obs_dim={env_obs_dim}."
                )

        policy_name = (
            "MultiInputPolicy"
            if isinstance(venv.observation_space, gym.spaces.Dict)
            else "MlpPolicy"
        )
        learner = PPO(
            policy_name,
            venv,
            learning_rate=1e-6,  # optimizer is controlled by BC trainer.
            n_steps=128,
            batch_size=128,
            ent_coef=0.02,
            clip_range=0.2,
            policy_kwargs=dict(
                log_std_init=0.0,
                ortho_init=False,
                net_arch=list(DEFAULTS["net_arch"]),
            ),
            verbose=2,
            device=device,
            seed=args.seed,
        )

        rng = np.random.default_rng(args.seed)
        bc_trainer = bc.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            demonstrations=train_trajs,
            policy=learner.policy,
            device=device,
            rng=rng,
            batch_size=args.batch_size,
            optimizer_kwargs={"lr": args.learning_rate},
            l2_weight=args.l2_weight,
            ent_weight=args.ent_weight,
        )

        train_obs_np, train_acts_np = flatten_trajectories(train_trajs)
        val_obs_np, val_acts_np = flatten_trajectories(val_trajs)

        if ENABLE_WANDB_BC_LOSS_LOGGING:
            if wandb is None:
                print("[warn] wandb is not installed; skipping BC train/val loss logging to wandb.")
            elif WANDB_MODE == "disabled":
                print("[info] WANDB_MODE=disabled; skipping wandb logging.")
            else:
                wandb_run = wandb.init(
                    project=WANDB_PROJECT,
                    entity=WANDB_ENTITY,
                    mode=WANDB_MODE,
                    name=f"klask_bc_loss_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "batch_size": int(args.batch_size),
                        "n_epochs": int(args.n_epochs),
                        "learning_rate": float(args.learning_rate),
                        "l2_weight": float(args.l2_weight),
                        "ent_weight": float(args.ent_weight),
                        "net_arch": list(DEFAULTS["net_arch"]),
                        "train_raw_steps": int(train_raw_steps),
                        "train_steps": int(train_steps),
                        "val_steps": int(val_steps),
                        "val_split_fraction": float(VAL_SPLIT_FRACTION),
                        "train_augmented": bool(augment_enabled),
                        "zero_mask_velocity_obs_sim": bool(args.zero_mask_velocity_obs_sim),
                        "zero_mask_velocity_obs_human": bool(args.zero_mask_velocity_obs_human),
                    },
                )

        print(
            f"[train] starting BC: epochs={args.n_epochs}, batch_size={args.batch_size}, "
            f"lr={args.learning_rate}, l2={args.l2_weight}, ent={args.ent_weight}"
        )
        epoch_counter = {"idx": 0}

        def _log_train_val_losses(epoch_idx: int) -> None:
            train_loss = compute_bc_loss_chunked(
                bc_trainer,
                train_obs_np,
                train_acts_np,
                torch.device(device),
                batch_size=VAL_EVAL_BATCH_SIZE,
            )
            val_loss = compute_bc_loss_chunked(
                bc_trainer,
                val_obs_np,
                val_acts_np,
                torch.device(device),
                batch_size=VAL_EVAL_BATCH_SIZE,
            )
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

        def _on_epoch_end() -> None:
            epoch_counter["idx"] += 1
            _log_train_val_losses(epoch_counter["idx"])

        t0 = time.time()
        _log_train_val_losses(epoch_counter["idx"])  # pre-training baseline
        bc_trainer.train(n_epochs=int(args.n_epochs), on_epoch_end=_on_epoch_end)
        elapsed = time.time() - t0
        if wandb_run is not None:
            wandb_run.finish()
            wandb_run = None
        print(f"[train] done in {elapsed:.1f}s")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"klask_bc_{timestamp}_{args.run_tag}"
        run_dir = os.path.join(args.save_root, run_name)
        os.makedirs(run_dir, exist_ok=True)

        policy_path = os.path.join(run_dir, "ppo_policy_bc_final.zip")
        learner.save(policy_path)

        config = {
            "config": {
                "demo_base": args.demo_base,
                "demo_source_format": args.demo_source_format,
                "save_root": args.save_root,
                "run_tag": args.run_tag,
                "seed": int(args.seed),
                "device": str(device),
                "num_envs": int(args.num_envs),
                "batch_size": int(args.batch_size),
                "n_epochs": int(args.n_epochs),
                "learning_rate": float(args.learning_rate),
                "l2_weight": float(args.l2_weight),
                "ent_weight": float(args.ent_weight),
                "augment": bool(augment_enabled),
                "net_arch": list(DEFAULTS["net_arch"]),
                "min_len": int(args.min_len),
                "strip_sim_action_history": bool(args.strip_sim_action_history),
                "effective_demo_obs_dim": int(demo_obs_dim) if demo_obs_dim is not None else None,
                "val_split_fraction": float(VAL_SPLIT_FRACTION),
                "train_raw_traj": int(len(train_raw_trajs)),
                "train_traj_after_aug": int(len(train_trajs)),
                "val_traj": int(len(val_trajs)),
                "zero_mask_velocity_obs_sim": bool(args.zero_mask_velocity_obs_sim),
                "zero_mask_velocity_obs_human": bool(args.zero_mask_velocity_obs_human),
            },
            "dataset_stats": dataset_stats.to_dict(),
            "run_name": run_name,
            "run_dir": run_dir,
            "created_at": datetime.datetime.now().isoformat(),
            "training_elapsed_sec": float(elapsed),
            "final_policy_path": policy_path,
        }
        with open(os.path.join(run_dir, "config.json"), "w") as fh:
            json.dump(config, fh, indent=2)

        with open(os.path.join(run_dir, "train_stats.json"), "w") as fh:
            json.dump(
                {
                    "training_elapsed_sec": float(elapsed),
                    "train_raw_steps": int(train_raw_steps),
                    "dataset_steps_used": int(train_steps),
                    "val_steps": int(val_steps),
                },
                fh,
                indent=2,
            )

        print(f"[out] {policy_path}")
        print(f"[out] {os.path.join(run_dir, 'config.json')}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        venv.close()


if __name__ == "__main__":
    main()
