"""Configurable GAIL training entrypoint for data-scaling experiments."""

from __future__ import annotations

import argparse
import collections
import copy
import contextlib
import dataclasses
import datetime
import json
import os
import sys
import time
import importlib.util
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

from isaaclab.app import AppLauncher


def _extract_device_arg(argv: list[str], default: str = "cuda:0") -> str:
    """Read --device from raw CLI args before AppLauncher is initialized."""
    device = str(default)
    i = 0
    while i < len(argv):
        token = str(argv[i]).strip()
        if token == "--device":
            if i + 1 < len(argv):
                device = str(argv[i + 1]).strip()
                i += 2
                continue
        elif token.startswith("--device="):
            value = token.split("=", 1)[1].strip()
            if value:
                device = value
        i += 1
    return device


_APP_LAUNCH_DEVICE = _extract_device_arg(sys.argv[1:], default="cuda:0")
app = AppLauncher(headless=True, device=_APP_LAUNCH_DEVICE).app

import gymnasium as gym
import numpy as np
import torch
import yaml
from imitation.algorithms.adversarial import common as adv_common
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data import types
from imitation.data.types import Trajectory
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util import util as imit_util
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecEnvWrapper

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from my_isaaclab_tasks.manager_based.klask.actuator_model import ActuatorModelWrapper
from my_isaaclab_tasks.manager_based.klask.env_cfg import EnvCfg
from my_isaaclab_tasks.manager_based.klask.config.parameters import KLASK_PARAMS

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

from demo_dataset import (
    DEMO_SOURCE_HUMAN_RAW,
    DEMO_SOURCE_SIM_NATIVE,
    DatasetLoadStats,
    augment_with_axis_mirrors,
    collect_pkl_paths,
    load_trajectories,
)

try:
    import wandb
except ImportError:
    wandb = None


DEFAULTS: Dict[str, Any] = {
    "demo_base": "data/expert_trajectories_cleaned",
    "demo_source_format": DEMO_SOURCE_HUMAN_RAW,
    "save_root": "trained_models",
    "run_tag": "GAIL_SWEEP",
    "seed": 42,
    "device": "cuda:0",
    "num_envs": 4096,
    "total_timesteps": 5_000_000,
    "learning_rate": 2.15e-4,
    "batch_size": 2048,
    "n_steps": 64,
    "gamma": 0.9903,
    "gae_lambda": 0.9685,
    "ent_coef": 0.0116,
    "clip_range": 0.1477,
    "vf_coef": 0.355037,
    "n_epochs": 5,
    "n_disc_updates": 2,
    "adaptive_disc_updates": True,
    "disc_acc_low": 0.6,
    "disc_acc_high": 0.8,
    "min_disc_updates": 1,
    "max_disc_updates": 4,
    "balance_disc_expert_loss": True,
    "disc_balance_gap_target": 0.05,
    "disc_expert_loss_gain": 3.0,
    "disc_expert_loss_max_weight": 3.0,
    "freeze_disc_on_high_gen_acc": True,
    "disc_gen_acc_freeze_threshold": 0.78,
    "disc_gen_exp_gap_freeze_threshold": 0.12,
    "disc_freeze_rounds": 1,
    "strict_disc_freeze": False,
    "freeze_disc_until_gen_acc_below_threshold": False,
    "disc_gen_acc_unfreeze_threshold": None,
    "disc_weight_decay": 2.8e-4,
    "disc_learning_rate": 2.17e-4,
    "disc_label_smoothing": 6.95e-4,
    "disc_reward_hid_sizes": [32],
    "disc_potential_hid_sizes": [32, 32],
    "event_aware_disc_weighting": True,
    "event_expert_goal_weight": 1.8,
    "event_expert_bad_weight": 0.6,
    "event_expert_timeout_weight": 0.8,
    "event_generated_goal_weight": 0.7,
    "event_generated_bad_weight": 1.8,
    "event_generated_timeout_weight": 1.2,
    "event_preterminal_window_steps": 8,
    "event_preterminal_decay": 1.0,
    "augment": True,
    "min_len": 2,
    "wandb_project": "klask-gail",
    "wandb_mode": "disabled",
    "wandb_train_log_freq_timesteps": 10_000,
    "bc_bootstrap_runs_root": "trained_models/data_wall_experiments/bc_runs",
    "init_generator_from_bc_path": None,
    "init_generator_from_bc_auto": False,
    "init_generator_from_bc_mode": "actor_only",
    "bc_anchor_during_gail": True,
    "bc_anchor_batch_size": 2048,
    "bc_anchor_updates_per_round": 1,
    "bc_anchor_learning_rate": 5e-5,
    "bc_anchor_ent_weight": 0.0,
    "bc_anchor_l2_weight": 0.0,
    "bc_anchor_decay_fraction": 1.0,
    "save_intermediate_checkpoints": True,
    "checkpoint_interval_timesteps": 500_000,
    "reward_net_type": "klask_relative_shaped",
    # Deployment-faithful defaults: the physical robot clips commands at 0.2 m/s
    # and the learned actuator model stands between command and realised velocity.
    # Pass --action-limit 1.0 --no-actuator-model to reproduce the paper.
    "action_limit": 0.2,
    "actuator_model": True,
    "strip_sim_action_history": True,
    "masked_velocities": False,
    "zero_mask_velocity_obs_human": False,
    "zero_mask_velocity_obs_sim": False,
    "val_split_fraction": 0.1,
    "val_split_level": "file",
    "peg1_only_fixed_opponent": False,
    "fixed_opponent_policy_path": None,
    "fixed_opponent_policy_format": None,
    "fixed_opponent_rl_games_config_path": None,
    "fixed_opponent_deterministic": True,
    "invert_opponent_actions": True,
}
BASE_OBS_DIM = 12
MASK_ACTION_LIKE_OBS_START = 8
MASK_ACTION_LIKE_OBS_END = 12
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
BC_BOOTSTRAP_MODE_ACTOR_ONLY = "actor_only"
BC_BOOTSTRAP_MODE_FULL_POLICY = "full_policy"
BC_BOOTSTRAP_VALUE_KEY_PREFIXES = (
    "mlp_extractor.value_net.",
    "value_net.",
    "vf_features_extractor.",
)
POLICY_FORMAT_SB3_PPO = "sb3_ppo"
POLICY_FORMAT_RL_GAMES = "rl_games"
REWARD_NET_TYPE_BASIC_SHAPED = "basic_shaped"
REWARD_NET_TYPE_KLASK_RELATIVE_SHAPED = "klask_relative_shaped"
GOAL_PLAYER = KLASK_PARAMS["scene"]["player_goal"]
GOAL_OPPONENT = KLASK_PARAMS["scene"]["opponent_goal"]


@dataclasses.dataclass
class BCBootstrapInfo:
    enabled: bool
    init_source: Optional[str]
    mode: Optional[str]
    policy_path: Optional[str]
    run_dir: Optional[str]
    run_tag: Optional[str]
    demo_base: Optional[str]
    demo_source_format: Optional[str]
    masked_velocities: Optional[bool]
    strip_sim_action_history: Optional[bool]
    zero_mask_velocity_obs_human: Optional[bool]
    zero_mask_velocity_obs_sim: Optional[bool]
    val_split_fraction: Optional[float]
    seed: Optional[int]
    created_at: Optional[str]
    split_fraction_matches_current: Optional[bool] = None
    seed_matches_current: Optional[bool] = None
    loaded_key_count: int = 0
    source_key_count: int = 0
    skipped_value_key_count: int = 0
    skipped_shape_mismatch_key_count: int = 0
    skipped_missing_target_key_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _normalize_infos_for_steps(infos: Any, expected_len: int) -> List[Dict[str, Any]]:
    """Convert stored infos to list[dict] with stable step length."""
    n = max(0, int(expected_len))
    if infos is None:
        return [{} for _ in range(n)]
    if isinstance(infos, np.ndarray):
        seq = infos.tolist()
    elif isinstance(infos, (list, tuple)):
        seq = list(infos)
    else:
        return [{} for _ in range(n)]

    out: List[Dict[str, Any]] = []
    for item in seq[:n]:
        out.append(dict(item) if isinstance(item, dict) else {})
    if len(out) < n:
        out.extend({} for _ in range(n - len(out)))
    return out


def _info_has_term(info: Mapping[str, Any], term: str) -> bool:
    if not isinstance(info, Mapping):
        return False
    if bool(info.get(f"term_{term}", False)):
        return True
    if bool(info.get(term, False)):
        return True
    if term == "time_out" and bool(info.get("timeout", False)):
        return True
    mode = info.get("termination_mode", None)
    if isinstance(mode, str) and mode == term:
        return True
    terms = info.get("termination_terms", None)
    if isinstance(terms, (list, tuple, set)):
        return term in terms
    return False


def annotate_preterminal_event_windows(
    trajectories: Sequence[Trajectory],
    *,
    window_steps: int,
) -> List[Trajectory]:
    """Annotate the final `window_steps` before terminal events in expert infos."""
    if int(window_steps) <= 0:
        return list(trajectories)

    window = int(window_steps)
    out: List[Trajectory] = []
    for tr in trajectories:
        n_steps = len(tr.acts)
        infos = _normalize_infos_for_steps(tr.infos, n_steps)
        for step_idx, info in enumerate(infos):
            active_terms = [term for term in CORE_TERMINATION_MODES if _info_has_term(info, term)]
            if not active_terms:
                continue
            start = max(0, step_idx - window + 1)
            for prev_idx in range(start, step_idx + 1):
                distance = int(step_idx - prev_idx)
                prev_info = infos[prev_idx]
                for term in active_terms:
                    prev_info[f"preterm_{term}"] = True
                    key = f"preterm_steps_to_{term}"
                    existing = prev_info.get(key)
                    if existing is None:
                        prev_info[key] = distance
                    else:
                        try:
                            prev_info[key] = min(int(existing), distance)
                        except Exception:
                            prev_info[key] = distance
        out.append(
            Trajectory(
                obs=tr.obs,
                acts=np.asarray(tr.acts, dtype=np.float32),
                infos=np.asarray(infos, dtype=object),
                terminal=tr.terminal,
            )
        )
    return out


class BCAnchorDuringGAILCallback(BaseCallback):
    """Interleave small BC actor updates with PPO to keep the policy near demos."""

    def __init__(
        self,
        *,
        demonstrations: Sequence[Trajectory],
        total_timesteps: int,
        batch_size: int,
        updates_per_round: int,
        learning_rate: float,
        ent_weight: float = 0.0,
        l2_weight: float = 0.0,
        decay_fraction: float = 1.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        transitions = rollout.flatten_trajectories(list(demonstrations))
        self._obs = transitions.obs
        self._acts = np.asarray(transitions.acts, dtype=np.float32)
        self._n = int(len(self._acts))
        self._total_timesteps = max(1, int(total_timesteps))
        self._batch_size = max(1, int(batch_size))
        self._updates_per_round = max(1, int(updates_per_round))
        self._learning_rate = float(learning_rate)
        self._ent_weight = float(ent_weight)
        self._l2_weight = float(l2_weight)
        self._decay_fraction = max(1e-6, float(decay_fraction))
        self._optimizer: Optional[torch.optim.Optimizer] = None

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self) -> None:
        self._optimizer = torch.optim.Adam(
            self.model.policy.parameters(),
            lr=float(self._learning_rate),
        )

    def _current_scale(self) -> float:
        decay_horizon = max(1.0, float(self._total_timesteps) * float(self._decay_fraction))
        progress = min(1.0, float(self.num_timesteps) / decay_horizon)
        return max(0.0, 1.0 - progress)

    def _sample_obs(self, indices: np.ndarray) -> Any:
        return self._obs[indices]

    def _run_single_update(self, scale: float) -> Dict[str, float]:
        assert self._optimizer is not None
        batch_size = min(int(self._batch_size), int(self._n))
        indices = np.random.randint(0, self._n, size=batch_size)
        batch_obs = types.maybe_unwrap_dictobs(self._sample_obs(indices))
        batch_acts = self._acts[indices]

        tensor_obs = types.map_maybe_dict(imit_util.safe_to_tensor, batch_obs)
        acts_th = imit_util.safe_to_tensor(batch_acts)
        if isinstance(tensor_obs, dict):
            tensor_obs = {k: v.to(self.model.device) for k, v in tensor_obs.items()}
        else:
            tensor_obs = tensor_obs.to(self.model.device)
        acts_th = acts_th.to(self.model.device)

        self.model.policy.train()
        _, log_prob, entropy = self.model.policy.evaluate_actions(
            tensor_obs,  # type: ignore[arg-type]
            acts_th,
        )
        prob_true_act = torch.exp(log_prob).mean()
        log_prob_mean = log_prob.mean()
        entropy_mean = (
            entropy.mean()
            if entropy is not None
            else torch.zeros(1, device=acts_th.device)
        )
        l2_norm = sum(torch.sum(torch.square(w)) for w in self.model.policy.parameters()) / 2
        loss = (
            (-log_prob_mean)
            - float(self._ent_weight) * entropy_mean
            + float(self._l2_weight) * l2_norm
        ) * float(scale)

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._optimizer.step()
        return {
            "bc_anchor_loss": float(loss.detach().item()),
            "bc_anchor_neglogp": float((-log_prob_mean).detach().item()),
            "bc_anchor_entropy": float(entropy_mean.detach().item()),
            "bc_anchor_prob_true_act": float(prob_true_act.detach().item()),
            "bc_anchor_l2_norm": float(l2_norm.detach().item()),
        }

    def _on_rollout_end(self) -> None:
        if self._optimizer is None or self._n <= 0:
            return
        scale = self._current_scale()
        self.logger.record("gen/bc_anchor_active", 1.0 if scale > 0.0 else 0.0)
        self.logger.record("gen/bc_anchor_scale", float(scale))
        self.logger.record("gen/bc_anchor_updates_per_round", float(self._updates_per_round))
        if scale <= 0.0:
            return

        stats_accum: Dict[str, float] = {}
        for _ in range(self._updates_per_round):
            stats = self._run_single_update(scale)
            for key, value in stats.items():
                stats_accum[key] = float(stats_accum.get(key, 0.0) + value)
        for key, value in stats_accum.items():
            self.logger.record(f"gen/{key}", float(value / float(self._updates_per_round)))


class GAILWithLabelSmoothing(GAIL):
    """GAIL with label smoothing, adaptive updates, and class-imbalance controls."""

    def __init__(
        self,
        *args,
        label_smoothing: float = 0.0,
        adaptive_disc_updates: bool = True,
        disc_acc_low: float = 0.6,
        disc_acc_high: float = 0.8,
        min_disc_updates: int = 1,
        max_disc_updates: Optional[int] = None,
        balance_disc_expert_loss: bool = True,
        disc_balance_gap_target: float = 0.05,
        disc_expert_loss_gain: float = 3.0,
        disc_expert_loss_max_weight: float = 3.0,
        freeze_disc_on_high_gen_acc: bool = True,
        disc_gen_acc_freeze_threshold: float = 0.78,
        disc_gen_exp_gap_freeze_threshold: float = 0.12,
        disc_freeze_rounds: int = 1,
        strict_disc_freeze: bool = False,
        freeze_disc_until_gen_acc_below_threshold: bool = False,
        disc_gen_acc_unfreeze_threshold: Optional[float] = None,
        event_aware_disc_weighting: bool = True,
        event_expert_goal_weight: float = 1.8,
        event_expert_bad_weight: float = 0.6,
        event_expert_timeout_weight: float = 0.8,
        event_generated_goal_weight: float = 0.7,
        event_generated_bad_weight: float = 1.8,
        event_generated_timeout_weight: float = 1.2,
        event_preterminal_window_steps: int = 0,
        event_preterminal_decay: float = 1.0,
        **kwargs,
    ):
        self.label_smoothing = float(label_smoothing)
        self.adaptive_disc_updates = bool(adaptive_disc_updates)
        self.disc_acc_low = float(disc_acc_low)
        self.disc_acc_high = float(disc_acc_high)
        self.min_disc_updates = max(1, int(min_disc_updates))
        self.max_disc_updates = (
            None if max_disc_updates is None else max(self.min_disc_updates, int(max_disc_updates))
        )
        self.balance_disc_expert_loss = bool(balance_disc_expert_loss)
        self.disc_balance_gap_target = float(disc_balance_gap_target)
        self.disc_expert_loss_gain = max(0.0, float(disc_expert_loss_gain))
        self.disc_expert_loss_max_weight = max(1.0, float(disc_expert_loss_max_weight))
        self.freeze_disc_on_high_gen_acc = bool(freeze_disc_on_high_gen_acc)
        self.disc_gen_acc_freeze_threshold = float(disc_gen_acc_freeze_threshold)
        self.disc_gen_exp_gap_freeze_threshold = float(disc_gen_exp_gap_freeze_threshold)
        self.disc_freeze_rounds = max(1, int(disc_freeze_rounds))
        self.strict_disc_freeze = bool(strict_disc_freeze)
        self.freeze_disc_until_gen_acc_below_threshold = bool(
            freeze_disc_until_gen_acc_below_threshold
        )
        self.disc_gen_acc_unfreeze_threshold = (
            None
            if disc_gen_acc_unfreeze_threshold is None
            else float(disc_gen_acc_unfreeze_threshold)
        )
        self.event_aware_disc_weighting = bool(event_aware_disc_weighting)
        self.event_expert_goal_weight = max(1e-6, float(event_expert_goal_weight))
        self.event_expert_bad_weight = max(1e-6, float(event_expert_bad_weight))
        self.event_expert_timeout_weight = max(1e-6, float(event_expert_timeout_weight))
        self.event_generated_goal_weight = max(1e-6, float(event_generated_goal_weight))
        self.event_generated_bad_weight = max(1e-6, float(event_generated_bad_weight))
        self.event_generated_timeout_weight = max(1e-6, float(event_generated_timeout_weight))
        self.event_preterminal_window_steps = max(0, int(event_preterminal_window_steps))
        self.event_preterminal_decay = float(event_preterminal_decay)
        self._last_disc_acc_gen: Optional[float] = None
        self._last_disc_acc_expert: Optional[float] = None
        self._last_train_stats: Dict[str, float] = {}
        self._disc_updates_to_skip = 0
        self._disc_total_skipped_updates = 0
        self._disc_strict_frozen = False
        self._disc_until_recovered_frozen = False
        if self.disc_acc_low >= self.disc_acc_high:
            raise ValueError(
                "disc_acc_low must be strictly smaller than disc_acc_high, "
                f"got {self.disc_acc_low} >= {self.disc_acc_high}."
            )
        if self.disc_gen_exp_gap_freeze_threshold < 0:
            raise ValueError(
                "disc_gen_exp_gap_freeze_threshold must be >= 0, "
                f"got {self.disc_gen_exp_gap_freeze_threshold}."
            )
        if self.strict_disc_freeze and self.freeze_disc_until_gen_acc_below_threshold:
            raise ValueError(
                "strict_disc_freeze and freeze_disc_until_gen_acc_below_threshold "
                "cannot both be enabled."
            )
        if self.freeze_disc_until_gen_acc_below_threshold:
            if self.disc_gen_acc_unfreeze_threshold is None:
                self.disc_gen_acc_unfreeze_threshold = float(self.disc_gen_acc_freeze_threshold)
            if self.disc_gen_acc_unfreeze_threshold <= 0.0:
                raise ValueError(
                    "disc_gen_acc_unfreeze_threshold must be > 0 when "
                    "freeze_disc_until_gen_acc_below_threshold is enabled."
                )
            if self.disc_gen_acc_unfreeze_threshold > self.disc_gen_acc_freeze_threshold:
                raise ValueError(
                    "disc_gen_acc_unfreeze_threshold must be <= "
                    "disc_gen_acc_freeze_threshold."
                )
        if self.event_preterminal_decay <= 0.0:
            raise ValueError(
                "event_preterminal_decay must be > 0, "
                f"got {self.event_preterminal_decay}."
            )
        super().__init__(*args, **kwargs)
        if self.max_disc_updates is None:
            self.max_disc_updates = max(self.min_disc_updates, int(self.n_disc_updates_per_round))

    @staticmethod
    def _normalize_infos(infos: Any, expected_len: int) -> List[Dict[str, Any]]:
        return _normalize_infos_for_steps(infos, expected_len)

    @staticmethod
    def _info_has_term(info: Mapping[str, Any], term: str) -> bool:
        return _info_has_term(info, term)

    @staticmethod
    def _summarize_weight_distribution(weights: np.ndarray, *, suffix: str) -> Dict[str, float]:
        arr = np.asarray(weights, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return {
                f"weight_non1_frac_{suffix}": 0.0,
                f"weight_gt1_frac_{suffix}": 0.0,
                f"weight_lt1_frac_{suffix}": 0.0,
                f"weight_min_{suffix}": 1.0,
                f"weight_max_{suffix}": 1.0,
                f"weight_std_{suffix}": 0.0,
                f"weight_p05_{suffix}": 1.0,
                f"weight_p50_{suffix}": 1.0,
                f"weight_p95_{suffix}": 1.0,
                f"weight_unique_count_{suffix}": 1.0,
            }

        eps = 1e-6
        non1 = np.abs(arr - 1.0) > eps
        gt1 = arr > (1.0 + eps)
        lt1 = arr < (1.0 - eps)
        p05, p50, p95 = np.percentile(arr, [5.0, 50.0, 95.0]).astype(np.float32)
        unique_count = int(np.unique(np.round(arr, 6)).size)

        return {
            f"weight_non1_frac_{suffix}": float(np.mean(non1)),
            f"weight_gt1_frac_{suffix}": float(np.mean(gt1)),
            f"weight_lt1_frac_{suffix}": float(np.mean(lt1)),
            f"weight_min_{suffix}": float(np.min(arr)),
            f"weight_max_{suffix}": float(np.max(arr)),
            f"weight_std_{suffix}": float(np.std(arr)),
            f"weight_p05_{suffix}": float(p05),
            f"weight_p50_{suffix}": float(p50),
            f"weight_p95_{suffix}": float(p95),
            f"weight_unique_count_{suffix}": float(unique_count),
        }

    def _build_event_weight(
        self,
        *,
        expert_infos: Any,
        gen_infos: Any,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        def _preterm_mask(info_list: List[Dict[str, Any]], term: str) -> np.ndarray:
            return np.asarray(
                [bool(info.get(f"preterm_{term}", False)) for info in info_list],
                dtype=bool,
            )

        def _preterm_distance(info_list: List[Dict[str, Any]], term: str) -> np.ndarray:
            out = np.zeros((len(info_list),), dtype=np.float32)
            key = f"preterm_steps_to_{term}"
            for idx, info in enumerate(info_list):
                raw = info.get(key, 0)
                try:
                    out[idx] = max(0.0, float(raw))
                except (TypeError, ValueError):
                    out[idx] = 0.0
            return out

        def _preterm_weight(
            base_weight: float,
            mask: np.ndarray,
            distance: np.ndarray,
        ) -> np.ndarray:
            out = np.ones_like(distance, dtype=np.float32)
            if not np.any(mask):
                return out
            if abs(float(base_weight) - 1.0) < 1e-6:
                return out
            scale = np.power(float(self.event_preterminal_decay), distance[mask]).astype(np.float32)
            out_mask = np.ones_like(scale, dtype=np.float32)
            if float(base_weight) > 1.0:
                out_mask = 1.0 + (float(base_weight) - 1.0) * scale
            else:
                out_mask = 1.0 - (1.0 - float(base_weight)) * scale
            out[mask] = out_mask.astype(np.float32)
            return out

        n = int(self.demo_minibatch_size)
        expert_info_list = self._normalize_infos(expert_infos, n)
        gen_info_list = self._normalize_infos(gen_infos, n)

        expert_weights = np.ones((n,), dtype=np.float32)
        gen_weights = np.ones((n,), dtype=np.float32)

        expert_good = np.asarray(
            [self._info_has_term(info, "goal_scored") or self._info_has_term(info, "opponent_in_goal") for info in expert_info_list],
            dtype=bool,
        )
        expert_bad = np.asarray(
            [self._info_has_term(info, "goal_conceded") or self._info_has_term(info, "player_in_goal") for info in expert_info_list],
            dtype=bool,
        )
        expert_timeout = np.asarray(
            [self._info_has_term(info, "time_out") for info in expert_info_list],
            dtype=bool,
        )

        gen_good = np.asarray(
            [self._info_has_term(info, "goal_scored") or self._info_has_term(info, "opponent_in_goal") for info in gen_info_list],
            dtype=bool,
        )
        gen_bad = np.asarray(
            [self._info_has_term(info, "goal_conceded") or self._info_has_term(info, "player_in_goal") for info in gen_info_list],
            dtype=bool,
        )
        gen_timeout = np.asarray(
            [self._info_has_term(info, "time_out") for info in gen_info_list],
            dtype=bool,
        )

        expert_pre_good = np.asarray(
            [
                bool(info.get("preterm_goal_scored", False))
                or bool(info.get("preterm_opponent_in_goal", False))
                for info in expert_info_list
            ],
            dtype=bool,
        )
        expert_pre_bad = np.asarray(
            [
                bool(info.get("preterm_goal_conceded", False))
                or bool(info.get("preterm_player_in_goal", False))
                for info in expert_info_list
            ],
            dtype=bool,
        )
        expert_pre_timeout = _preterm_mask(expert_info_list, "time_out")
        gen_pre_good = np.asarray(
            [
                bool(info.get("preterm_goal_scored", False))
                or bool(info.get("preterm_opponent_in_goal", False))
                for info in gen_info_list
            ],
            dtype=bool,
        )
        gen_pre_bad = np.asarray(
            [
                bool(info.get("preterm_goal_conceded", False))
                or bool(info.get("preterm_player_in_goal", False))
                for info in gen_info_list
            ],
            dtype=bool,
        )
        gen_pre_timeout = _preterm_mask(gen_info_list, "time_out")
        expert_pre_good_distance = np.minimum(
            _preterm_distance(expert_info_list, "goal_scored"),
            _preterm_distance(expert_info_list, "opponent_in_goal"),
        )
        expert_pre_bad_distance = np.minimum(
            _preterm_distance(expert_info_list, "goal_conceded"),
            _preterm_distance(expert_info_list, "player_in_goal"),
        )
        expert_pre_timeout_distance = _preterm_distance(expert_info_list, "time_out")
        gen_pre_good_distance = np.minimum(
            _preterm_distance(gen_info_list, "goal_scored"),
            _preterm_distance(gen_info_list, "opponent_in_goal"),
        )
        gen_pre_bad_distance = np.minimum(
            _preterm_distance(gen_info_list, "goal_conceded"),
            _preterm_distance(gen_info_list, "player_in_goal"),
        )
        gen_pre_timeout_distance = _preterm_distance(gen_info_list, "time_out")

        expert_weights[expert_good] *= float(self.event_expert_goal_weight)
        expert_weights[expert_bad] *= float(self.event_expert_bad_weight)
        expert_weights[expert_timeout] *= float(self.event_expert_timeout_weight)

        # For generator-labeled samples: reduce penalty on good events, increase on bad/timeouts.
        gen_weights[gen_good] *= float(self.event_generated_goal_weight)
        gen_weights[gen_bad] *= float(self.event_generated_bad_weight)
        gen_weights[gen_timeout] *= float(self.event_generated_timeout_weight)
        if self.event_preterminal_window_steps > 0:
            expert_weights *= _preterm_weight(
                self.event_expert_goal_weight,
                expert_pre_good & ~expert_good,
                expert_pre_good_distance,
            )
            expert_weights *= _preterm_weight(
                self.event_expert_bad_weight,
                expert_pre_bad & ~expert_bad,
                expert_pre_bad_distance,
            )
            expert_weights *= _preterm_weight(
                self.event_expert_timeout_weight,
                expert_pre_timeout & ~expert_timeout,
                expert_pre_timeout_distance,
            )
            gen_weights *= _preterm_weight(
                self.event_generated_goal_weight,
                gen_pre_good & ~gen_good,
                gen_pre_good_distance,
            )
            gen_weights *= _preterm_weight(
                self.event_generated_bad_weight,
                gen_pre_bad & ~gen_bad,
                gen_pre_bad_distance,
            )
            gen_weights *= _preterm_weight(
                self.event_generated_timeout_weight,
                gen_pre_timeout & ~gen_timeout,
                gen_pre_timeout_distance,
            )

        weight = np.concatenate([expert_weights, gen_weights]).astype(np.float32)
        stats = {
            "info_coverage_expert": float(np.mean([1.0 if len(info) > 0 else 0.0 for info in expert_info_list])),
            "info_coverage_generated": float(np.mean([1.0 if len(info) > 0 else 0.0 for info in gen_info_list])),
            "rate_expert_good": float(expert_good.mean()) if n > 0 else 0.0,
            "rate_expert_bad": float(expert_bad.mean()) if n > 0 else 0.0,
            "rate_expert_timeout": float(expert_timeout.mean()) if n > 0 else 0.0,
            "rate_generated_good": float(gen_good.mean()) if n > 0 else 0.0,
            "rate_generated_bad": float(gen_bad.mean()) if n > 0 else 0.0,
            "rate_generated_timeout": float(gen_timeout.mean()) if n > 0 else 0.0,
            "rate_expert_preterm_good": float(expert_pre_good.mean()) if n > 0 else 0.0,
            "rate_expert_preterm_bad": float(expert_pre_bad.mean()) if n > 0 else 0.0,
            "rate_expert_preterm_timeout": float(expert_pre_timeout.mean()) if n > 0 else 0.0,
            "rate_generated_preterm_good": float(gen_pre_good.mean()) if n > 0 else 0.0,
            "rate_generated_preterm_bad": float(gen_pre_bad.mean()) if n > 0 else 0.0,
            "rate_generated_preterm_timeout": float(gen_pre_timeout.mean()) if n > 0 else 0.0,
            "weight_mean_expert": float(expert_weights.mean()) if n > 0 else 1.0,
            "weight_mean_generated": float(gen_weights.mean()) if n > 0 else 1.0,
        }
        stats.update(self._summarize_weight_distribution(expert_weights, suffix="expert"))
        stats.update(self._summarize_weight_distribution(gen_weights, suffix="generated"))
        return weight, stats

    def _make_disc_train_batches(
        self,
        *,
        gen_samples: Optional[Mapping] = None,
        expert_samples: Optional[Mapping] = None,
    ):
        """Override to preserve infos and attach event-aware sample weights."""
        batch_size = self.demo_batch_size
        if expert_samples is None:
            expert_samples = self._next_expert_batch()
        if gen_samples is None:
            if self._gen_replay_buffer.size() == 0:
                raise RuntimeError(
                    "No generator samples for training. Call `train_gen()` first.",
                )
            gen_samples_dataclass = self._gen_replay_buffer.sample(batch_size)
            gen_samples = types.dataclass_quick_asdict(gen_samples_dataclass)

        if not (len(gen_samples["obs"]) == len(expert_samples["obs"]) == batch_size):
            raise ValueError(
                "Need `gen_samples` and `expert_samples` of length `demo_batch_size` "
                f"(n_gen={len(gen_samples['obs'])} "
                f"n_expert={len(expert_samples['obs'])} "
                f"demo_batch_size={batch_size})",
            )

        expert_samples = dict(expert_samples)
        gen_samples = dict(gen_samples)
        for field in dataclasses.fields(types.Transitions):
            k = field.name
            if k == "infos":
                continue
            for d in [gen_samples, expert_samples]:
                if isinstance(d[k], torch.Tensor):
                    d[k] = d[k].detach().numpy()

        for start in range(0, batch_size, self.demo_minibatch_size):
            end = start + self.demo_minibatch_size
            expert_batch = {k: v[start:end] for k, v in expert_samples.items()}
            gen_batch = {k: v[start:end] for k, v in gen_samples.items()}

            obs = np.concatenate([expert_batch["obs"], gen_batch["obs"]])
            acts = np.concatenate([expert_batch["acts"], gen_batch["acts"]])
            next_obs = np.concatenate([expert_batch["next_obs"], gen_batch["next_obs"]])
            dones = np.concatenate([expert_batch["dones"], gen_batch["dones"]])
            labels_expert_is_one = np.concatenate(
                [
                    np.ones(self.demo_minibatch_size, dtype=int),
                    np.zeros(self.demo_minibatch_size, dtype=int),
                ],
            )

            with torch.no_grad():
                obs_th = torch.as_tensor(obs, device=self.gen_algo.device)
                acts_th = torch.as_tensor(acts, device=self.gen_algo.device)
                log_policy_act_prob = self._get_log_policy_act_prob(obs_th, acts_th)
                if log_policy_act_prob is not None:
                    assert len(log_policy_act_prob) == 2 * self.demo_minibatch_size
                    log_policy_act_prob = log_policy_act_prob.reshape((2 * self.demo_minibatch_size,))
                del obs_th, acts_th

            obs_th, acts_th, next_obs_th, dones_th = self.reward_train.preprocess(
                obs,
                acts,
                next_obs,
                dones,
            )
            batch_dict = {
                "state": obs_th,
                "action": acts_th,
                "next_state": next_obs_th,
                "done": dones_th,
                "labels_expert_is_one": self._torchify_array(labels_expert_is_one),
                "log_policy_act_prob": log_policy_act_prob,
            }

            if self.event_aware_disc_weighting:
                event_weight, event_stats = self._build_event_weight(
                    expert_infos=expert_batch.get("infos", None),
                    gen_infos=gen_batch.get("infos", None),
                )
                batch_dict["event_weight"] = self._torchify_array(event_weight)
                for key, value in event_stats.items():
                    batch_dict[f"event_{key}"] = float(value)

            yield batch_dict

    @staticmethod
    def _safe_scalar(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(out):
            return None
        return out

    def _compute_expert_loss_weight(self) -> float:
        if not self.balance_disc_expert_loss:
            return 1.0
        if self._last_disc_acc_gen is None or self._last_disc_acc_expert is None:
            return 1.0
        gap = (
            float(self._last_disc_acc_gen)
            - float(self._last_disc_acc_expert)
            - float(self.disc_balance_gap_target)
        )
        if not np.isfinite(gap) or gap <= 0.0:
            return 1.0
        weight = 1.0 + float(self.disc_expert_loss_gain) * gap
        return float(np.clip(weight, 1.0, self.disc_expert_loss_max_weight))

    def _compute_disc_batch_stats(
        self,
        *,
        expert_samples: Optional[Dict],
        gen_samples: Optional[Dict],
        expert_loss_weight: float,
        backprop: bool,
    ) -> Dict[str, float]:
        loss = None
        hard_labels = None
        disc_logits = None
        event_stat_sums: Dict[str, float] = {}
        event_stat_count = 0
        sample_weight_stat_sums: Dict[str, float] = {}
        sample_weight_stat_count = 0
        batch_iter = self._make_disc_train_batches(
            gen_samples=gen_samples,
            expert_samples=expert_samples,
        )
        grad_context = contextlib.nullcontext() if backprop else torch.no_grad()

        with grad_context:
            for batch in batch_iter:
                disc_logits = self.logits_expert_is_high(
                    batch["state"],
                    batch["action"],
                    batch["next_state"],
                    batch["done"],
                    batch["log_policy_act_prob"],
                )
                hard_labels = batch["labels_expert_is_one"].float()
                eps = max(0.0, min(0.5, self.label_smoothing))
                if eps > 0:
                    labels = hard_labels * (1.0 - eps) + (1.0 - hard_labels) * eps
                else:
                    labels = hard_labels
                sample_weight = torch.ones_like(labels)
                if expert_loss_weight > 1.0:
                    sample_weight = sample_weight * torch.where(
                        hard_labels > 0.5,
                        torch.full_like(labels, float(expert_loss_weight)),
                        torch.ones_like(labels),
                    )
                if self.event_aware_disc_weighting and "event_weight" in batch:
                    event_weight = batch["event_weight"]
                    if isinstance(event_weight, torch.Tensor):
                        event_weight = event_weight.to(device=labels.device, dtype=labels.dtype)
                    else:
                        event_weight = torch.as_tensor(
                            event_weight,
                            device=labels.device,
                            dtype=labels.dtype,
                        )
                    sample_weight = sample_weight * torch.clamp(event_weight, min=1e-6)
                    event_stat_count += 1
                    for key, value in batch.items():
                        if not str(key).startswith("event_"):
                            continue
                        if isinstance(value, torch.Tensor):
                            continue
                        parsed = self._safe_scalar(value)
                        if parsed is None:
                            continue
                        event_stat_sums[str(key)] = float(
                            event_stat_sums.get(str(key), 0.0) + parsed
                        )
                sw = sample_weight.detach()
                sw_eps = 1e-6
                sample_weight_stat_count += 1
                sample_weight_stat_sums["sample_weight_mean"] = (
                    float(sample_weight_stat_sums.get("sample_weight_mean", 0.0))
                    + float(sw.mean().item())
                )
                sample_weight_stat_sums["sample_weight_min"] = (
                    float(sample_weight_stat_sums.get("sample_weight_min", 0.0))
                    + float(sw.min().item())
                )
                sample_weight_stat_sums["sample_weight_max"] = (
                    float(sample_weight_stat_sums.get("sample_weight_max", 0.0))
                    + float(sw.max().item())
                )
                sample_weight_stat_sums["sample_weight_std"] = (
                    float(sample_weight_stat_sums.get("sample_weight_std", 0.0))
                    + float(sw.std(unbiased=False).item())
                )
                sample_weight_stat_sums["sample_weight_non1_frac"] = (
                    float(sample_weight_stat_sums.get("sample_weight_non1_frac", 0.0))
                    + float((torch.abs(sw - 1.0) > sw_eps).float().mean().item())
                )
                sample_weight_stat_sums["sample_weight_gt1_frac"] = (
                    float(sample_weight_stat_sums.get("sample_weight_gt1_frac", 0.0))
                    + float((sw > (1.0 + sw_eps)).float().mean().item())
                )
                sample_weight_stat_sums["sample_weight_lt1_frac"] = (
                    float(sample_weight_stat_sums.get("sample_weight_lt1_frac", 0.0))
                    + float((sw < (1.0 - sw_eps)).float().mean().item())
                )
                if torch.allclose(sample_weight, torch.ones_like(sample_weight)):
                    sample_weight = None
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    disc_logits,
                    labels,
                    weight=sample_weight,
                )

                assert len(batch["state"]) == 2 * self.demo_minibatch_size
                loss *= self.demo_minibatch_size / self.demo_batch_size
                if backprop:
                    loss.backward()

        train_stats: Dict[str, float] = {}
        if disc_logits is not None and hard_labels is not None and loss is not None:
            with torch.no_grad():
                train_stats = dict(
                    adv_common.compute_train_stats(
                        disc_logits,
                        hard_labels,
                        loss,
                    )
                )
        if event_stat_count > 0:
            for key, value in event_stat_sums.items():
                train_stats[f"disc_{key}"] = float(value / float(event_stat_count))
        if sample_weight_stat_count > 0:
            for key, value in sample_weight_stat_sums.items():
                train_stats[f"disc_{key}"] = float(value / float(sample_weight_stat_count))
        return train_stats

    def _update_cached_disc_stats(self, train_stats: Mapping[str, float]) -> None:
        cached: Dict[str, float] = {}
        for key, value in train_stats.items():
            parsed = self._safe_scalar(value)
            if parsed is not None:
                cached[str(key)] = parsed
        self._last_train_stats = cached
        parsed_gen = self._safe_scalar(train_stats.get("disc_acc_gen"))
        if parsed_gen is not None:
            self._last_disc_acc_gen = parsed_gen
        parsed_expert = self._safe_scalar(train_stats.get("disc_acc_expert"))
        if parsed_expert is not None:
            self._last_disc_acc_expert = parsed_expert

    def _record_disc_control_metrics(self, *, expert_loss_weight: float, frozen_step: float) -> None:
        self.logger.record(
            "disc/adaptive_n_disc_updates_per_round",
            float(self.n_disc_updates_per_round),
        )
        self.logger.record(
            "disc/expert_loss_weight",
            float(expert_loss_weight),
        )
        self.logger.record(
            "disc/frozen_step",
            float(frozen_step),
        )
        self.logger.record(
            "disc/freeze_updates_to_skip",
            float(self._disc_updates_to_skip),
        )
        self.logger.record(
            "disc/freeze_total_skipped_updates",
            float(self._disc_total_skipped_updates),
        )
        self.logger.record(
            "disc/strict_frozen_active",
            1.0 if self._disc_strict_frozen else 0.0,
        )
        self.logger.record(
            "disc/until_recovered_frozen_active",
            1.0 if self._disc_until_recovered_frozen else 0.0,
        )

    def _maybe_adjust_disc_updates(self, train_stats: Mapping[str, float]) -> None:
        if not self.adaptive_disc_updates:
            return
        disc_acc = self._safe_scalar(train_stats.get("disc_acc"))
        if disc_acc is None:
            return

        current_updates = int(self.n_disc_updates_per_round)
        new_updates = current_updates
        if disc_acc > self.disc_acc_high:
            new_updates = max(self.min_disc_updates, current_updates - 1)
        elif disc_acc < self.disc_acc_low:
            max_updates = (
                int(self.max_disc_updates)
                if self.max_disc_updates is not None
                else current_updates + 1
            )
            new_updates = min(max_updates, current_updates + 1)

        if new_updates != current_updates:
            self.n_disc_updates_per_round = int(new_updates)
            print(
                "[gail] adaptive disc updates: "
                f"{current_updates} -> {new_updates} (disc_acc={disc_acc:.3f})"
            )

    def _maybe_schedule_disc_freeze(self, train_stats: Mapping[str, float]) -> None:
        if not self.freeze_disc_on_high_gen_acc:
            return
        if self.strict_disc_freeze and self._disc_strict_frozen:
            return
        if (
            self.freeze_disc_until_gen_acc_below_threshold
            and self._disc_until_recovered_frozen
        ):
            return
        disc_acc_gen = self._safe_scalar(train_stats.get("disc_acc_gen"))
        if disc_acc_gen is None or disc_acc_gen < self.disc_gen_acc_freeze_threshold:
            return
        disc_acc_expert = self._safe_scalar(train_stats.get("disc_acc_expert"))
        gap_text = "n/a"
        if disc_acc_expert is not None:
            gap = disc_acc_gen - disc_acc_expert
            if gap < self.disc_gen_exp_gap_freeze_threshold:
                return
            gap_text = f"{gap:.3f}"

        if self.strict_disc_freeze:
            self._disc_updates_to_skip = 0
            self._disc_strict_frozen = True
            print(
                "[gail] strict freeze of discriminator updates activated "
                f"(disc_acc_gen={disc_acc_gen:.3f}, disc_acc_gap={gap_text})"
            )
            return

        if self.freeze_disc_until_gen_acc_below_threshold:
            self._disc_updates_to_skip = 0
            self._disc_until_recovered_frozen = True
            print(
                "[gail] freezing discriminator until generator accuracy recovers "
                f"(disc_acc_gen={disc_acc_gen:.3f}, "
                f"unfreeze_threshold={self.disc_gen_acc_unfreeze_threshold:.3f}, "
                f"disc_acc_gap={gap_text})"
            )
            return

        rounds_to_skip = max(1, int(self.disc_freeze_rounds))
        updates_per_round = max(1, int(self.n_disc_updates_per_round))
        skip_updates = rounds_to_skip * updates_per_round
        if skip_updates <= self._disc_updates_to_skip:
            return

        prev_skip = int(self._disc_updates_to_skip)
        self._disc_updates_to_skip = int(skip_updates)
        print(
            "[gail] freezing discriminator updates: "
            f"skip {prev_skip} -> {self._disc_updates_to_skip} "
            f"(disc_acc_gen={disc_acc_gen:.3f}, disc_acc_gap={gap_text})"
        )

    def train_disc(
        self,
        *,
        expert_samples: Optional[Dict] = None,
        gen_samples: Optional[Dict] = None,
    ) -> Dict[str, float]:
        if (
            self._disc_strict_frozen
            or self._disc_until_recovered_frozen
            or self._disc_updates_to_skip > 0
        ):
            if self._disc_updates_to_skip > 0:
                self._disc_updates_to_skip -= 1
            self._disc_total_skipped_updates += 1
            expert_loss_weight = self._compute_expert_loss_weight()
            skip_stats = self._compute_disc_batch_stats(
                expert_samples=expert_samples,
                gen_samples=gen_samples,
                expert_loss_weight=expert_loss_weight,
                backprop=False,
            )
            skip_stats["disc_expert_loss_weight"] = float(expert_loss_weight)
            skip_stats["disc_updates_skipped_remaining"] = float(self._disc_updates_to_skip)
            skip_stats["disc_frozen_step"] = 1.0
            skip_stats["disc_strict_frozen_active"] = 1.0 if self._disc_strict_frozen else 0.0
            if self._disc_until_recovered_frozen:
                probe_disc_acc_gen = self._safe_scalar(skip_stats.get("disc_acc_gen"))
                if (
                    probe_disc_acc_gen is not None
                    and self.disc_gen_acc_unfreeze_threshold is not None
                    and probe_disc_acc_gen < self.disc_gen_acc_unfreeze_threshold
                ):
                    self._disc_until_recovered_frozen = False
                    print(
                        "[gail] resuming discriminator updates after generator accuracy "
                        f"recovered (disc_acc_gen={probe_disc_acc_gen:.3f}, "
                        f"unfreeze_threshold={self.disc_gen_acc_unfreeze_threshold:.3f})"
                    )
            skip_stats["disc_until_recovered_frozen_active"] = (
                1.0 if self._disc_until_recovered_frozen else 0.0
            )
            skip_stats["disc_probe_evaluated"] = 1.0
            self._update_cached_disc_stats(skip_stats)
            with self.logger.accumulate_means("disc"):
                self.logger.record("global_step", self._global_step)
                for k, v in skip_stats.items():
                    self.logger.record(k, v)
                self._record_disc_control_metrics(
                    expert_loss_weight=expert_loss_weight,
                    frozen_step=1.0,
                )
                self._disc_step += 1
                self.logger.dump(self._disc_step)
            return skip_stats

        with self.logger.accumulate_means("disc"):
            self._disc_opt.zero_grad()
            expert_loss_weight = self._compute_expert_loss_weight()
            train_stats = self._compute_disc_batch_stats(
                expert_samples=expert_samples,
                gen_samples=gen_samples,
                expert_loss_weight=expert_loss_weight,
                backprop=True,
            )
            self._disc_opt.step()
            self._disc_step += 1

            train_stats["disc_expert_loss_weight"] = float(expert_loss_weight)
            train_stats["disc_updates_skipped_remaining"] = float(self._disc_updates_to_skip)
            train_stats["disc_frozen_step"] = 0.0
            train_stats["disc_strict_frozen_active"] = 1.0 if self._disc_strict_frozen else 0.0
            train_stats["disc_until_recovered_frozen_active"] = (
                1.0 if self._disc_until_recovered_frozen else 0.0
            )
            train_stats["disc_probe_evaluated"] = 0.0
            train_stats["disc_event_weighting_enabled"] = float(self.event_aware_disc_weighting)
            self._maybe_adjust_disc_updates(train_stats)
            self._maybe_schedule_disc_freeze(train_stats)
            train_stats["disc_updates_skipped_remaining"] = float(self._disc_updates_to_skip)
            train_stats["disc_strict_frozen_active"] = 1.0 if self._disc_strict_frozen else 0.0
            train_stats["disc_until_recovered_frozen_active"] = (
                1.0 if self._disc_until_recovered_frozen else 0.0
            )
            self._update_cached_disc_stats(train_stats)
            self.logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self._record_disc_control_metrics(
                expert_loss_weight=expert_loss_weight,
                frozen_step=0.0,
            )
            self.logger.dump(self._disc_step)

        return train_stats


class GAILWandbTrainMetricsCallback(BaseCallback):
    """Log high-frequency progress metrics to W&B during generator updates."""

    def __init__(
        self,
        *,
        wandb_run: Any,
        total_timesteps: int,
        log_freq_timesteps: int = 10_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.wandb_run = wandb_run
        self.total_timesteps = int(total_timesteps)
        self.log_freq_timesteps = max(1, int(log_freq_timesteps))
        self.last_log = 0
        self.start_time: Optional[float] = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_log) < self.log_freq_timesteps:
            return True
        elapsed = time.time() - self.start_time if self.start_time is not None else 0.0
        fps = self.num_timesteps / elapsed if elapsed > 0 else 0.0
        progress = (
            100.0 * float(self.num_timesteps) / float(self.total_timesteps)
            if self.total_timesteps > 0
            else 0.0
        )
        payload: Dict[str, float] = {
            "train/timesteps": float(self.num_timesteps),
            "train/fps": float(fps),
            "train/progress_pct": float(progress),
            "train/elapsed_sec": float(elapsed),
        }
        try:
            self.wandb_run.log(payload, step=int(self.num_timesteps))
        except Exception as e:
            print(f"[warn] W&B train metric logging failed: {e}")
        self.last_log = self.num_timesteps
        return True


class GAILWandbRoundMetricsCallback:
    """Log train metrics once per GAIL round when aggregate keys are available."""

    def __init__(self, *, wandb_run: Any, gen_algo: PPO, total_timesteps: int):
        self.wandb_run = wandb_run
        self.gen_algo = gen_algo
        self.total_timesteps = int(total_timesteps)
        self.start_time = time.time()

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(out):
            return None
        return out

    def _first_present(
        self,
        values: Mapping[str, Any],
        keys: List[str],
    ) -> Optional[float]:
        for key in keys:
            if key in values:
                parsed = self._safe_float(values[key])
                if parsed is not None:
                    return parsed
        return None

    def _collect_metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        logger_obj = getattr(self.gen_algo, "logger", None)
        name_to_value = getattr(logger_obj, "name_to_value", None)
        if not isinstance(name_to_value, dict):
            return out

        disc_loss = self._first_present(
            name_to_value,
            ["mean/disc/disc_loss", "raw/disc/disc_loss", "disc/disc_loss", "disc_loss"],
        )
        disc_acc_total = self._first_present(
            name_to_value,
            ["mean/disc/disc_acc", "raw/disc/disc_acc", "disc/disc_acc", "disc_acc"],
        )
        disc_acc_expert = self._first_present(
            name_to_value,
            [
                "mean/disc/disc_acc_expert",
                "raw/disc/disc_acc_expert",
                "disc/disc_acc_expert",
                "disc_acc_expert",
            ],
        )
        disc_acc_gen = self._first_present(
            name_to_value,
            [
                "mean/disc/disc_acc_gen",
                "raw/disc/disc_acc_gen",
                "disc/disc_acc_gen",
                "disc_acc_gen",
            ],
        )
        disc_n_updates = self._first_present(
            name_to_value,
            [
                "mean/disc/disc/adaptive_n_disc_updates_per_round",
                "raw/disc/disc/adaptive_n_disc_updates_per_round",
                "mean/disc/adaptive_n_disc_updates_per_round",
                "raw/disc/adaptive_n_disc_updates_per_round",
                "disc/adaptive_n_disc_updates_per_round",
                "adaptive_n_disc_updates_per_round",
            ],
        )
        disc_expert_loss_weight = self._first_present(
            name_to_value,
            [
                "mean/disc/disc/expert_loss_weight",
                "raw/disc/disc/expert_loss_weight",
                "mean/disc/expert_loss_weight",
                "raw/disc/expert_loss_weight",
                "disc/expert_loss_weight",
                "expert_loss_weight",
                "mean/disc/disc_expert_loss_weight",
                "raw/disc/disc_expert_loss_weight",
                "disc_expert_loss_weight",
            ],
        )
        disc_frozen_step = self._first_present(
            name_to_value,
            [
                "mean/disc/disc/frozen_step",
                "raw/disc/disc/frozen_step",
                "mean/disc/frozen_step",
                "raw/disc/frozen_step",
                "disc/frozen_step",
                "disc_frozen_step",
            ],
        )
        disc_freeze_updates_to_skip = self._first_present(
            name_to_value,
            [
                "mean/disc/disc/freeze_updates_to_skip",
                "raw/disc/disc/freeze_updates_to_skip",
                "mean/disc/freeze_updates_to_skip",
                "raw/disc/freeze_updates_to_skip",
                "disc/freeze_updates_to_skip",
                "freeze_updates_to_skip",
                "disc_updates_skipped_remaining",
            ],
        )
        disc_freeze_total_skipped_updates = self._first_present(
            name_to_value,
            [
                "mean/disc/disc/freeze_total_skipped_updates",
                "raw/disc/disc/freeze_total_skipped_updates",
                "mean/disc/freeze_total_skipped_updates",
                "raw/disc/freeze_total_skipped_updates",
                "disc/freeze_total_skipped_updates",
                "freeze_total_skipped_updates",
            ],
        )
        disc_strict_frozen_active = self._first_present(
            name_to_value,
            [
                "mean/disc/disc/strict_frozen_active",
                "raw/disc/disc/strict_frozen_active",
                "mean/disc/strict_frozen_active",
                "raw/disc/strict_frozen_active",
                "disc/strict_frozen_active",
                "disc_strict_frozen_active",
            ],
        )
        disc_until_recovered_frozen_active = self._first_present(
            name_to_value,
            [
                "mean/disc/disc/until_recovered_frozen_active",
                "raw/disc/disc/until_recovered_frozen_active",
                "mean/disc/until_recovered_frozen_active",
                "raw/disc/until_recovered_frozen_active",
                "disc/until_recovered_frozen_active",
                "disc_until_recovered_frozen_active",
            ],
        )
        disc_probe_evaluated = self._first_present(
            name_to_value,
            [
                "mean/disc/disc/probe_evaluated",
                "raw/disc/disc/probe_evaluated",
                "mean/disc/probe_evaluated",
                "raw/disc/probe_evaluated",
                "disc/probe_evaluated",
                "disc_probe_evaluated",
            ],
        )
        gen_entropy_loss = self._first_present(
            name_to_value,
            [
                "mean/gen/train/entropy_loss",
                "raw/gen/train/entropy_loss",
                "gen/train/entropy_loss",
                "train/entropy_loss",
            ],
        )
        gen_approx_kl = self._first_present(
            name_to_value,
            [
                "mean/gen/train/approx_kl",
                "raw/gen/train/approx_kl",
                "gen/train/approx_kl",
                "train/approx_kl",
            ],
        )
        gen_clip_fraction = self._first_present(
            name_to_value,
            [
                "mean/gen/train/clip_fraction",
                "raw/gen/train/clip_fraction",
                "gen/train/clip_fraction",
                "train/clip_fraction",
            ],
        )
        bc_anchor_loss = self._first_present(
            name_to_value,
            ["gen/bc_anchor_loss", "mean/gen/bc_anchor_loss", "raw/gen/bc_anchor_loss"],
        )
        bc_anchor_prob_true_act = self._first_present(
            name_to_value,
            [
                "gen/bc_anchor_prob_true_act",
                "mean/gen/bc_anchor_prob_true_act",
                "raw/gen/bc_anchor_prob_true_act",
            ],
        )
        bc_anchor_scale = self._first_present(
            name_to_value,
            ["gen/bc_anchor_scale", "mean/gen/bc_anchor_scale", "raw/gen/bc_anchor_scale"],
        )

        if disc_loss is not None:
            out["train/discriminator_loss"] = disc_loss
        if disc_acc_total is not None:
            out["train/discriminator_accuracy_total"] = disc_acc_total
            out["train/discriminator_accuracy"] = disc_acc_total
        if disc_acc_gen is not None:
            out["train/discriminator_accuracy_gen"] = disc_acc_gen
        if disc_acc_expert is not None:
            out["train/discriminator_accuracy_expert"] = disc_acc_expert
            out["train/discriminator_accuracy_discriminator"] = disc_acc_expert
        if disc_n_updates is not None:
            out["train/discriminator_n_disc_updates_per_round"] = disc_n_updates
        if disc_expert_loss_weight is not None:
            out["train/discriminator_expert_loss_weight"] = disc_expert_loss_weight
        if disc_frozen_step is not None:
            out["train/discriminator_frozen_step"] = disc_frozen_step
        if disc_freeze_updates_to_skip is not None:
            out["train/discriminator_freeze_updates_to_skip"] = disc_freeze_updates_to_skip
        if disc_freeze_total_skipped_updates is not None:
            out["train/discriminator_freeze_total_skipped_updates"] = (
                disc_freeze_total_skipped_updates
            )
        if disc_strict_frozen_active is not None:
            out["train/discriminator_strict_frozen_active"] = disc_strict_frozen_active
        if disc_until_recovered_frozen_active is not None:
            out["train/discriminator_until_recovered_frozen_active"] = (
                disc_until_recovered_frozen_active
            )
        if disc_probe_evaluated is not None:
            out["train/discriminator_probe_evaluated"] = disc_probe_evaluated
        if gen_entropy_loss is not None:
            out["train/generator_entropy_loss"] = gen_entropy_loss
        if gen_approx_kl is not None:
            out["train/generator_approx_kl"] = gen_approx_kl
        if gen_clip_fraction is not None:
            out["train/generator_clip_fraction"] = gen_clip_fraction
        if bc_anchor_loss is not None:
            out["train/bc_anchor_loss"] = bc_anchor_loss
        if bc_anchor_prob_true_act is not None:
            out["train/bc_anchor_prob_true_act"] = bc_anchor_prob_true_act
        if bc_anchor_scale is not None:
            out["train/bc_anchor_scale"] = bc_anchor_scale

        return out

    def __call__(self, round_num: int) -> None:
        timesteps = int(getattr(self.gen_algo, "num_timesteps", 0))
        elapsed = max(0.0, time.time() - self.start_time)
        fps = float(timesteps) / elapsed if elapsed > 0 else 0.0
        progress = (
            100.0 * float(timesteps) / float(self.total_timesteps)
            if self.total_timesteps > 0
            else 0.0
        )
        payload: Dict[str, float] = {
            "train/round": float(round_num),
            "train/timesteps": float(timesteps),
            "train/progress_pct": float(progress),
            "train/fps": float(fps),
            "train/elapsed_sec": float(elapsed),
        }
        payload.update(self._collect_metrics())
        try:
            self.wandb_run.log(payload, step=timesteps)
        except Exception as e:
            print(f"[warn] W&B round metric logging failed: {e}")


class GAILCheckpointCallback:
    """Save policy and reward-net checkpoints every N timesteps."""

    def __init__(
        self,
        *,
        gen_algo: PPO,
        gail_algo: GAIL,
        checkpoints_dir: str,
        interval_timesteps: int,
    ):
        self.gen_algo = gen_algo
        self.gail_algo = gail_algo
        self.checkpoints_dir = str(checkpoints_dir)
        self.interval_timesteps = max(1, int(interval_timesteps))
        self.next_save_timestep = int(self.interval_timesteps)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def _save_checkpoint(self, *, timesteps: int, round_num: int) -> None:
        step_tag = f"{int(timesteps):08d}"
        policy_path = os.path.join(
            self.checkpoints_dir,
            f"ppo_policy_gail_step_{step_tag}.zip",
        )
        reward_net_path = os.path.join(
            self.checkpoints_dir,
            f"gail_reward_net_step_{step_tag}.pt",
        )
        meta_path = os.path.join(
            self.checkpoints_dir,
            f"checkpoint_step_{step_tag}.json",
        )
        self.gen_algo.save(policy_path)
        torch.save(self.gail_algo._reward_net.state_dict(), reward_net_path)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "timesteps": int(timesteps),
                    "round": int(round_num),
                    "policy_path": policy_path,
                    "reward_net_path": reward_net_path,
                    "created_at": datetime.datetime.now().isoformat(),
                },
                fh,
                indent=2,
            )
        print(
            "[checkpoint] saved intermediate checkpoint "
            f"(timesteps={timesteps:,}, round={round_num})"
        )

    def __call__(self, round_num: int) -> None:
        timesteps = int(getattr(self.gen_algo, "num_timesteps", 0))
        while timesteps >= self.next_save_timestep:
            self._save_checkpoint(
                timesteps=self.next_save_timestep,
                round_num=int(round_num),
            )
            self.next_save_timestep += int(self.interval_timesteps)


class CompositeGAILRoundCallback:
    """Fan out a GAIL round callback to multiple callbacks."""

    def __init__(self, callbacks: Sequence[Any]):
        self.callbacks = [cb for cb in callbacks if cb is not None]

    def __call__(self, round_num: int) -> None:
        for callback in self.callbacks:
            callback(round_num)


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


def _to_bool_vector(x: Any, num_envs: int) -> np.ndarray:
    """Convert tensor/array-like values to bool array with shape (num_envs,)."""
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.size == 0:
        return np.zeros((num_envs,), dtype=bool)
    flat = arr.reshape(-1).astype(bool)
    if flat.size != int(num_envs):
        raise ValueError(f"Expected {num_envs} flags, got {flat.size}.")
    return flat


def _termination_mode_from_flags(*, done: bool, flags: Mapping[str, bool]) -> str:
    if not bool(done):
        return "none"
    for name in TERMINATION_MODE_PRIORITY:
        if bool(flags.get(name, False)):
            return name
    active = [name for name, value in flags.items() if bool(value)]
    if active:
        return str(active[0])
    return "unknown"


def _obs_to_array(obs: Any) -> np.ndarray:
    """Convert obs output to a plain float32 array with shape (N, D)."""
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        if "policy" in obs:
            arr = np.asarray(obs["policy"], dtype=np.float32)
        else:
            arr = np.asarray(next(iter(obs.values())), dtype=np.float32)
    else:
        arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _build_opponent_obs(obs: Any) -> np.ndarray:
    """
    Remap observation for opponent policy.
    For standard 12D layout:
      [ball(0:4), peg1(4:8), peg2(8:12)] -> swap first 8 fields as [peg1, peg2, ball].
    """
    obs_arr = _obs_to_array(obs)
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

    env_info: Dict[str, Any] = {
        "observation_space": player_obs_space,
        "action_space": player_action_space,
        "agents": 1,
        "value_size": 1,
    }
    return env_info


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
    """Reconstruct expert-style engineered features from local 12D Klask observations."""
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

    extras = np.concatenate(
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
    )
    return extras.astype(np.float32)


def _augment_local_obs_to_expected_dim(obs_arr: np.ndarray, expected_dim: int) -> np.ndarray:
    """Expand 12D local obs to known richer layouts expected by legacy RL-Games checkpoints."""
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
    """Expose RL-Games player through an SB3-like `predict()` interface."""

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
        obs_arr = _augment_local_obs_to_expected_dim(_obs_to_array(obs), self.obs_dim)
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
    # This adapter is used inside the SB3/GAIL stack where the policy interface
    # is normalized to [-1, 1]. Let the wrapper handle clipping/inversion rather
    # than rl-games rescaling against its own action bounds.
    if hasattr(player, "clip_actions"):
        player.clip_actions = False
    # rl-games players treat observations as unbatched until get_batch_size() is
    # called once. Without this, a (N, obs_dim) tensor is unsqueezed to
    # (1, N, obs_dim) and flattened incorrectly inside the actor MLP.
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


class ZeroOpponentPolicy:
    """Fallback fixed-opponent policy used until a real policy is loaded."""

    def predict(self, obs: Any, deterministic: bool = True):
        obs_arr = _obs_to_array(obs)
        acts = np.zeros((obs_arr.shape[0], 2), dtype=np.float32)
        return acts, None


class FixedOpponentVecEnvWrapper(VecEnvWrapper):
    """
    Peg1-only learner wrapper:
    - Learner action space is 2D (peg1).
    - Peg2 action comes from a frozen opponent policy.
    """

    def __init__(
        self,
        venv: Sb3VecEnvWrapper,
        *,
        opponent_policy: Any,
        deterministic_opponent: bool = True,
        invert_opponent_actions: bool = True,
        action_limit: float = 1.0,
    ):
        # The base env is unbounded, so this Box *is* the generator's action bound.
        self.action_limit = float(action_limit)
        action_space = gym.spaces.Box(low=-self.action_limit, high=self.action_limit,
                                      shape=(2,), dtype=np.float32)
        super().__init__(venv, action_space=action_space)
        self.opponent_policy = opponent_policy
        self.deterministic_opponent = bool(deterministic_opponent)
        self.invert_opponent_actions = bool(invert_opponent_actions)
        self._last_obs: Optional[np.ndarray] = None

    def reset(self):
        obs = self.venv.reset()
        self._last_obs = _obs_to_array(obs)
        return obs

    def step_async(self, actions):
        if self._last_obs is None:
            raise RuntimeError("FixedOpponentVecEnvWrapper.step_async called before reset().")

        peg1_actions = np.asarray(actions, dtype=np.float32)
        if peg1_actions.ndim == 1:
            peg1_actions = peg1_actions.reshape(1, -1)
        if peg1_actions.shape[1] < 2:
            raise ValueError(
                f"Peg1 actions must have at least 2 dims, got shape={peg1_actions.shape}."
            )
        peg1_actions = peg1_actions[:, :2]

        opp_obs = _build_opponent_obs(self._last_obs)
        opp_actions, _ = self.opponent_policy.predict(
            opp_obs,
            deterministic=self.deterministic_opponent,
        )
        opp_actions = np.asarray(opp_actions, dtype=np.float32)
        if opp_actions.ndim == 1:
            opp_actions = opp_actions.reshape(1, -1)
        if opp_actions.shape[1] < 2:
            raise ValueError(
                f"Opponent policy must output at least 2 dims, got shape={opp_actions.shape}."
            )
        opp_actions = opp_actions[:, :2]
        if self.invert_opponent_actions:
            opp_actions = -opp_actions
        opp_actions = np.clip(opp_actions, -1.0, 1.0)

        full_actions = np.concatenate([peg1_actions, opp_actions], axis=1).astype(np.float32)
        self.venv.step_async(full_actions)

    def step_wait(self):
        obs, rew, dones, infos = self.venv.step_wait()
        self._last_obs = _obs_to_array(obs)
        return obs, rew, dones, infos


class TerminationInfoVecEnvWrapper(VecEnvWrapper):
    """Inject per-step termination fields into VecEnv infos for generated trajectories."""

    def __init__(self, venv: Sb3VecEnvWrapper, *, preterminal_window_steps: int = 0):
        super().__init__(venv)
        term_mgr = getattr(venv.unwrapped, "termination_manager", None)
        self._has_term_mgr = term_mgr is not None
        self._preterminal_window_steps = max(0, int(preterminal_window_steps))
        active_terms: List[str] = []
        if term_mgr is not None:
            active_terms = list(getattr(term_mgr, "active_terms", []))
        self._tracked_terms = list(dict.fromkeys([*active_terms, *CORE_TERMINATION_MODES]))
        self._recent_infos: List[collections.deque] = [
            collections.deque(maxlen=max(1, self._preterminal_window_steps))
            for _ in range(self.num_envs)
        ]
        if not self._has_term_mgr:
            print("[warn] termination_manager unavailable: generated infos won't include live term flags.")
        elif not active_terms:
            print("[warn] termination_manager.active_terms is empty: tracking only core term names.")

    def _read_term_flags(self) -> Dict[str, np.ndarray]:
        term_mgr = getattr(self.venv.unwrapped, "termination_manager", None)
        if term_mgr is None:
            return {}
        out: Dict[str, np.ndarray] = {}
        for name in self._tracked_terms:
            try:
                out[name] = _to_bool_vector(term_mgr.get_term(name), self.num_envs)
            except Exception:
                out[name] = np.zeros((self.num_envs,), dtype=bool)
        return out

    def reset(self):
        self._recent_infos = [
            collections.deque(maxlen=max(1, self._preterminal_window_steps))
            for _ in range(self.num_envs)
        ]
        return self.venv.reset()

    def step_wait(self):
        obs, rew, dones, infos = self.venv.step_wait()
        dones_np = np.asarray(dones).reshape(-1).astype(bool)

        if infos is None:
            info_list: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        else:
            info_list = [dict(info) if isinstance(info, dict) else {} for info in infos]
            if len(info_list) < self.num_envs:
                info_list.extend({} for _ in range(self.num_envs - len(info_list)))

        term_flags = self._read_term_flags()
        if not term_flags:
            term_flags = {name: np.zeros((self.num_envs,), dtype=bool) for name in self._tracked_terms}
            for env_idx in range(self.num_envs):
                info_i = info_list[env_idx]
                for name in self._tracked_terms:
                    val = info_i.get(name, info_i.get(f"term_{name}", False))
                    if name == "time_out":
                        val = (
                            bool(val)
                            or bool(info_i.get("timeout", False))
                            or bool(info_i.get("TimeLimit.truncated", False))
                        )
                    term_flags[name][env_idx] = bool(val)

        for env_idx in range(self.num_envs):
            info_i = info_list[env_idx]
            done_i = bool(dones_np[env_idx]) if env_idx < dones_np.size else False
            info_i["done"] = done_i
            active_terms: List[str] = []
            per_env_flags: Dict[str, bool] = {}
            for name in self._tracked_terms:
                flag = bool(term_flags.get(name, np.zeros((self.num_envs,), dtype=bool))[env_idx])
                per_env_flags[name] = flag
                info_i[f"term_{name}"] = flag
                if flag:
                    active_terms.append(name)
            # Keep same semantics as synthetic dataset: non-empty only on done transitions.
            info_i["termination_terms"] = active_terms if done_i else []
            info_i["termination_mode"] = _termination_mode_from_flags(
                done=done_i,
                flags=per_env_flags,
            )
            if self._preterminal_window_steps > 0:
                history = self._recent_infos[env_idx]
                history.append(info_i)
                if done_i and active_terms:
                    history_list = list(history)
                    for hist_idx, hist_info in enumerate(history_list):
                        distance = len(history_list) - hist_idx - 1
                        for term in active_terms:
                            hist_info[f"preterm_{term}"] = True
                            key = f"preterm_steps_to_{term}"
                            existing = hist_info.get(key)
                            if existing is None:
                                hist_info[key] = distance
                            else:
                                try:
                                    hist_info[key] = min(int(existing), distance)
                                except Exception:
                                    hist_info[key] = distance
                    history.clear()

        return obs, rew, dones, info_list


def make_env(
    num_envs: int,
    seed: int,
    sim_device: Optional[str] = None,
    *,
    peg1_only_fixed_opponent: bool = False,
    opponent_policy: Optional[Any] = None,
    deterministic_opponent: bool = True,
    invert_opponent_actions: bool = True,
    preterminal_window_steps: int = 0,
    action_limit: float = 1.0,
    actuator_model: bool = False,
) -> Sb3VecEnvWrapper:
    cfg = EnvCfg()
    cfg.scene.num_envs = int(num_envs)
    cfg.seed = int(seed)
    if sim_device is not None:
        # Keep simulation device aligned with --device used for policy training.
        try:
            cfg.sim.device = str(sim_device)
        except Exception:
            pass
    base_env = gym.make("Isaac-Klask-v0", cfg=cfg)
    if bool(actuator_model):
        # Stand the learned actuator model between commanded and realised velocity,
        # as the physical robot does. Demonstrations and every evaluation already
        # apply it; enabling it here makes generator rollouts consistent with both.
        base_env = ActuatorModelWrapper(base_env, device=str(sim_device or "cuda:0"))
    sb3_env = Sb3VecEnvWrapper(base_env)
    wrapped_env: VecEnvWrapper = sb3_env
    if bool(peg1_only_fixed_opponent):
        wrapped_env = FixedOpponentVecEnvWrapper(
            wrapped_env,
            opponent_policy=(opponent_policy if opponent_policy is not None else ZeroOpponentPolicy()),
            deterministic_opponent=bool(deterministic_opponent),
            invert_opponent_actions=bool(invert_opponent_actions),
            action_limit=float(action_limit),
        )
    return TerminationInfoVecEnvWrapper(
        wrapped_env,
        preterminal_window_steps=int(preterminal_window_steps),
    )


def parse_hid_sizes(arg_value: str) -> List[int]:
    parts = [p.strip() for p in str(arg_value).split(",") if p.strip()]
    if not parts:
        return []
    return [int(p) for p in parts]


def normalize_path(path_value: Optional[str]) -> Optional[str]:
    if path_value is None:
        return None
    return os.path.abspath(os.path.expanduser(str(path_value)))


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    return bool(value)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_bc_bootstrap_metadata_from_policy_path(
    policy_path: str,
    *,
    init_source: str,
    mode: str,
) -> BCBootstrapInfo:
    policy_path = str(normalize_path(policy_path))
    run_dir = os.path.dirname(policy_path)
    config_path = os.path.join(run_dir, "config.json")
    payload: Dict[str, Any] = {}
    inner: Dict[str, Any] = {}
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        inner = dict(payload.get("config", {}))

    return BCBootstrapInfo(
        enabled=True,
        init_source=str(init_source),
        mode=str(mode),
        policy_path=policy_path,
        run_dir=(run_dir if os.path.isdir(run_dir) else None),
        run_tag=inner.get("run_tag"),
        demo_base=normalize_path(inner.get("demo_base")),
        demo_source_format=inner.get("demo_source_format"),
        masked_velocities=(
            _safe_bool(inner.get("masked_velocities"), default=False)
            if "masked_velocities" in inner
            else (
                _safe_bool(inner.get("zero_mask_velocity_obs_human"), default=False)
                if str(inner.get("demo_source_format")) == DEMO_SOURCE_HUMAN_RAW
                else _safe_bool(inner.get("zero_mask_velocity_obs_sim"), default=False)
            )
        ),
        strip_sim_action_history=(
            None
            if "strip_sim_action_history" not in inner
            else bool(inner.get("strip_sim_action_history"))
        ),
        zero_mask_velocity_obs_human=(
            None
            if "zero_mask_velocity_obs_human" not in inner
            else bool(inner.get("zero_mask_velocity_obs_human"))
        ),
        zero_mask_velocity_obs_sim=(
            None
            if "zero_mask_velocity_obs_sim" not in inner
            else bool(inner.get("zero_mask_velocity_obs_sim"))
        ),
        val_split_fraction=_safe_float(inner.get("val_split_fraction")),
        seed=_safe_int(inner.get("seed")),
        created_at=payload.get("created_at"),
    )


def resolve_bc_bootstrap_info(
    *,
    bc_runs_root: str,
    demo_base: str,
    demo_source_format: str,
    masked_velocities: bool,
    strip_sim_action_history: bool,
    zero_mask_velocity_obs_human: bool,
    zero_mask_velocity_obs_sim: bool,
    init_mode: str,
) -> BCBootstrapInfo:
    bc_runs_root = str(normalize_path(bc_runs_root))
    demo_base = str(normalize_path(demo_base))
    if not os.path.isdir(bc_runs_root):
        raise FileNotFoundError(f"BC bootstrap runs root not found: {bc_runs_root}")

    candidates: List[Tuple[Tuple[str, str], BCBootstrapInfo]] = []
    for entry in os.scandir(bc_runs_root):
        if not entry.is_dir():
            continue
        config_path = os.path.join(entry.path, "config.json")
        if not os.path.isfile(config_path):
            continue
        with open(config_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        inner = dict(payload.get("config", {}))
        candidate_demo_base = normalize_path(inner.get("demo_base"))
        if candidate_demo_base != demo_base:
            continue
        if str(inner.get("demo_source_format", demo_source_format)) != str(demo_source_format):
            continue
        candidate_masked_velocities = (
            _safe_bool(inner.get("masked_velocities"), default=False)
            if "masked_velocities" in inner
            else (
                _safe_bool(inner.get("zero_mask_velocity_obs_human"), default=False)
                if str(demo_source_format) == DEMO_SOURCE_HUMAN_RAW
                else _safe_bool(inner.get("zero_mask_velocity_obs_sim"), default=False)
            )
        )
        if bool(candidate_masked_velocities) != bool(masked_velocities):
            continue
        if _safe_bool(inner.get("strip_sim_action_history"), default=False) != bool(
            strip_sim_action_history
        ):
            continue
        if _safe_bool(inner.get("zero_mask_velocity_obs_human"), default=False) != bool(
            zero_mask_velocity_obs_human
        ):
            continue
        if _safe_bool(inner.get("zero_mask_velocity_obs_sim"), default=False) != bool(
            zero_mask_velocity_obs_sim
        ):
            continue
        policy_path = payload.get("final_policy_path") or os.path.join(
            entry.path, "ppo_policy_bc_final.zip"
        )
        policy_path = normalize_path(policy_path)
        if policy_path is None or not os.path.isfile(policy_path):
            continue
        info = BCBootstrapInfo(
            enabled=True,
            init_source="auto",
            mode=str(init_mode),
            policy_path=str(policy_path),
            run_dir=str(entry.path),
            run_tag=inner.get("run_tag"),
            demo_base=candidate_demo_base,
            demo_source_format=inner.get("demo_source_format"),
            masked_velocities=bool(candidate_masked_velocities),
            strip_sim_action_history=(
                None
                if "strip_sim_action_history" not in inner
                else bool(inner.get("strip_sim_action_history"))
            ),
            zero_mask_velocity_obs_human=(
                None
                if "zero_mask_velocity_obs_human" not in inner
                else bool(inner.get("zero_mask_velocity_obs_human"))
            ),
            zero_mask_velocity_obs_sim=(
                None
                if "zero_mask_velocity_obs_sim" not in inner
                else bool(inner.get("zero_mask_velocity_obs_sim"))
            ),
            val_split_fraction=_safe_float(inner.get("val_split_fraction")),
            seed=_safe_int(inner.get("seed")),
            created_at=payload.get("created_at"),
        )
        sort_key = (str(payload.get("created_at", "")), os.path.basename(entry.path))
        candidates.append((sort_key, info))

    if not candidates:
        raise FileNotFoundError(
            "No compatible BC checkpoint found for bootstrap under "
            f"{bc_runs_root} with demo_base={demo_base}, "
            f"demo_source_format={demo_source_format}, "
            f"strip_sim_action_history={strip_sim_action_history}, "
            f"zero_mask_velocity_obs_human={zero_mask_velocity_obs_human}, "
            f"zero_mask_velocity_obs_sim={zero_mask_velocity_obs_sim}."
        )

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def bootstrap_generator_from_bc(
    learner: PPO,
    *,
    bc_policy_path: str,
    init_mode: str,
    device: str,
) -> Dict[str, int]:
    bc_model = PPO.load(str(normalize_path(bc_policy_path)), device=device)
    source_state = bc_model.policy.state_dict()
    target_state = learner.policy.state_dict()

    loaded_state: Dict[str, torch.Tensor] = {}
    skipped_value_keys = 0
    skipped_shape_keys = 0
    skipped_missing_target_keys = 0

    for key, tensor in source_state.items():
        if init_mode == BC_BOOTSTRAP_MODE_ACTOR_ONLY and key.startswith(
            BC_BOOTSTRAP_VALUE_KEY_PREFIXES
        ):
            skipped_value_keys += 1
            continue
        target_tensor = target_state.get(key)
        if target_tensor is None:
            skipped_missing_target_keys += 1
            continue
        if tuple(target_tensor.shape) != tuple(tensor.shape):
            skipped_shape_keys += 1
            continue
        loaded_state[key] = tensor.detach().clone()

    if not loaded_state:
        raise ValueError(
            "BC bootstrap resolved a checkpoint but no compatible policy weights could be loaded."
        )

    merged_state = dict(target_state)
    merged_state.update(loaded_state)
    learner.policy.load_state_dict(merged_state, strict=False)
    del bc_model

    return {
        "loaded_key_count": int(len(loaded_state)),
        "source_key_count": int(len(source_state)),
        "skipped_value_key_count": int(skipped_value_keys),
        "skipped_shape_mismatch_key_count": int(skipped_shape_keys),
        "skipped_missing_target_key_count": int(skipped_missing_target_keys),
    }


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


def slice_peg1_actions(
    trajectories: List[Trajectory],
    *,
    action_start: int = 0,
    action_dim: int = 2,
) -> List[Trajectory]:
    """Convert demonstration actions to peg1-only actions by slicing leading action dims."""
    out: List[Trajectory] = []
    start = int(action_start)
    width = int(action_dim)
    stop = start + width
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


def split_train_val_trajectories(
    trajectories: List[Trajectory],
    group_ids: Sequence[Hashable],
    *,
    val_fraction: float = 0.1,
    seed: int = 0,
    split_level: str = "file",
) -> Tuple[List[Trajectory], List[Trajectory], List[int], List[int]]:
    """Split trajectories into train/validation sets by file or by trajectory."""
    n = len(trajectories)
    all_indices = list(range(n))
    if n <= 1 or val_fraction <= 0.0:
        return list(trajectories), [], all_indices, []
    if len(group_ids) != n:
        raise ValueError(
            f"group_ids length {len(group_ids)} does not match trajectories length {n}"
        )

    rng = np.random.default_rng(seed)
    val_indices: List[int] = []
    split_key = str(split_level).strip().lower()
    if split_key == "file":
        group_to_indices: Dict[Hashable, List[int]] = {}
        for idx, gid in enumerate(group_ids):
            group_to_indices.setdefault(gid, []).append(idx)
        groups = list(group_to_indices.keys())
        if len(groups) <= 1:
            return list(trajectories), [], all_indices, []
        n_val_groups = max(1, int(round(len(groups) * float(val_fraction))))
        n_val_groups = min(n_val_groups, len(groups) - 1)
        perm = rng.permutation(len(groups))
        val_groups = {groups[i] for i in perm[:n_val_groups]}
        val_indices = [idx for idx, gid in enumerate(group_ids) if gid in val_groups]
    elif split_key == "trajectory":
        n_val = max(1, int(round(n * float(val_fraction))))
        n_val = min(n_val, n - 1)
        perm = rng.permutation(n)
        val_indices = sorted(int(i) for i in perm[:n_val])
    else:
        raise ValueError(f"Unsupported split_level={split_level!r}. Expected 'file' or 'trajectory'.")

    val_set = set(val_indices)
    train_indices = [idx for idx in all_indices if idx not in val_set]
    if not train_indices:
        return list(trajectories), [], all_indices, []

    train = [trajectories[idx] for idx in train_indices]
    val = [trajectories[idx] for idx in val_indices]
    return train, val, train_indices, val_indices


def _load_config_defaults(argv: list[str]) -> Dict[str, Any]:
    """If --config <path.json> is on the command line, load it and merge with DEFAULTS."""
    merged = dict(DEFAULTS)
    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]
            with open(config_path, "r") as fh:
                merged.update(json.load(fh))
            break
        elif argv[i].startswith("--config="):
            config_path = argv[i].split("=", 1)[1]
            with open(config_path, "r") as fh:
                merged.update(json.load(fh))
            break
        i += 1
    return merged


def parse_args() -> argparse.Namespace:
    defaults = _load_config_defaults(sys.argv[1:])
    parser = argparse.ArgumentParser(description="Train GAIL policy on configurable dataset.")
    parser.add_argument("--config", type=str, default=None, help="JSON config file. Values are used as defaults; CLI flags override.")
    parser.add_argument("--demo-base", type=str, default=defaults["demo_base"])
    parser.add_argument(
        "--demo-source-format",
        choices=[DEMO_SOURCE_HUMAN_RAW, DEMO_SOURCE_SIM_NATIVE],
        default=defaults["demo_source_format"],
    )
    parser.add_argument("--save-root", type=str, default=defaults["save_root"])
    parser.add_argument("--run-tag", type=str, default=defaults["run_tag"])
    parser.add_argument(
        "--bc-bootstrap-runs-root",
        type=str,
        default=defaults["bc_bootstrap_runs_root"],
        help=(
            "Directory containing previously trained BC runs. Used only when "
            "--init-generator-from-bc-auto is enabled."
        ),
    )
    parser.add_argument(
        "--init-generator-from-bc-path",
        type=str,
        default=defaults["init_generator_from_bc_path"],
        help=(
            "Optional path to a saved BC PPO checkpoint (`ppo_policy_bc_final.zip`) "
            "used to initialize the GAIL generator before adversarial training."
        ),
    )
    parser.add_argument(
        "--init-generator-from-bc-auto",
        action=argparse.BooleanOptionalAction,
        default=defaults["init_generator_from_bc_auto"],
        help=(
            "Automatically resolve the latest compatible BC checkpoint from "
            "--bc-bootstrap-runs-root based on demo path and preprocessing flags."
        ),
    )
    parser.add_argument(
        "--init-generator-from-bc-mode",
        choices=[BC_BOOTSTRAP_MODE_ACTOR_ONLY, BC_BOOTSTRAP_MODE_FULL_POLICY],
        default=defaults["init_generator_from_bc_mode"],
        help=(
            "How much of the BC checkpoint to copy into the GAIL generator. "
            "`actor_only` copies policy trunk/action head but leaves critic weights untouched."
        ),
    )
    parser.add_argument(
        "--bc-anchor-during-gail",
        action=argparse.BooleanOptionalAction,
        default=defaults["bc_anchor_during_gail"],
        help=(
            "Run small supervised BC actor updates on expert minibatches after each rollout "
            "to keep the generator near the expert manifold during GAIL."
        ),
    )
    parser.add_argument(
        "--bc-anchor-batch-size",
        type=int,
        default=defaults["bc_anchor_batch_size"],
        help="Expert minibatch size for each BC anchor update during GAIL.",
    )
    parser.add_argument(
        "--bc-anchor-updates-per-round",
        type=int,
        default=defaults["bc_anchor_updates_per_round"],
        help="Number of BC anchor updates to run after each rollout collection.",
    )
    parser.add_argument(
        "--bc-anchor-learning-rate",
        type=float,
        default=defaults["bc_anchor_learning_rate"],
        help="Learning rate for the BC anchor optimizer.",
    )
    parser.add_argument(
        "--bc-anchor-ent-weight",
        type=float,
        default=defaults["bc_anchor_ent_weight"],
        help="Entropy regularization weight used inside the BC anchor loss.",
    )
    parser.add_argument(
        "--bc-anchor-l2-weight",
        type=float,
        default=defaults["bc_anchor_l2_weight"],
        help="L2 regularization weight used inside the BC anchor loss.",
    )
    parser.add_argument(
        "--bc-anchor-decay-fraction",
        type=float,
        default=defaults["bc_anchor_decay_fraction"],
        help=(
            "Linearly decay BC anchor strength to zero over this fraction of total timesteps "
            "(e.g. 0.6 means anchor fades out by 60% progress)."
        ),
    )
    parser.add_argument(
        "--save-intermediate-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=defaults["save_intermediate_checkpoints"],
        help="Save intermediate policy and reward-net checkpoints during training.",
    )
    parser.add_argument(
        "--checkpoint-interval-timesteps",
        type=int,
        default=defaults["checkpoint_interval_timesteps"],
        help="Save an intermediate checkpoint every N generator timesteps.",
    )
    parser.add_argument(
        "--reward-net-type",
        choices=[REWARD_NET_TYPE_BASIC_SHAPED, REWARD_NET_TYPE_KLASK_RELATIVE_SHAPED],
        default=defaults["reward_net_type"],
        help="Reward-net architecture to use for discriminator/reward learning.",
    )
    parser.add_argument(
        "--action-limit",
        type=float,
        default=defaults["action_limit"],
        help="Per-axis velocity command bound (m/s) for the GAIL generator. "
             "0.2 = hardware limit (default); 1.0 = paper reproduction.",
    )
    parser.add_argument(
        "--actuator-model",
        action=argparse.BooleanOptionalAction,
        default=defaults["actuator_model"],
        help="Apply the learned actuator model during generator rollouts. On by "
             "default; --no-actuator-model reproduces the paper's training setup.",
    )
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--device", type=str, default=defaults["device"])
    parser.add_argument(
        "--peg1-only-fixed-opponent",
        action=argparse.BooleanOptionalAction,
        default=defaults["peg1_only_fixed_opponent"],
        help=(
            "Train Peg1-only policy (2D actions) while Peg2 is controlled by a frozen opponent policy. "
            "When enabled, demo actions are sliced to first 2 dims."
        ),
    )
    parser.add_argument(
        "--fixed-opponent-policy-path",
        type=str,
        default=defaults["fixed_opponent_policy_path"],
        help=(
            "Path to frozen opponent policy checkpoint (.zip for SB3 PPO, .pth for RL-Games). "
            "Required when --peg1-only-fixed-opponent is enabled."
        ),
    )
    parser.add_argument(
        "--fixed-opponent-policy-format",
        type=str,
        default=defaults["fixed_opponent_policy_format"],
        help=(
            "Explicit fixed-opponent policy format. "
            f"Valid values: {POLICY_FORMAT_SB3_PPO}, {POLICY_FORMAT_RL_GAMES}. "
            "If omitted, inferred from checkpoint extension."
        ),
    )
    parser.add_argument(
        "--fixed-opponent-rl-games-config-path",
        type=str,
        default=defaults["fixed_opponent_rl_games_config_path"],
        help=(
            "RL-Games agent.yaml for the fixed opponent. "
            f"Required when --fixed-opponent-policy-format={POLICY_FORMAT_RL_GAMES}."
        ),
    )
    parser.add_argument(
        "--fixed-opponent-deterministic",
        action=argparse.BooleanOptionalAction,
        default=defaults["fixed_opponent_deterministic"],
        help="Use deterministic actions for fixed opponent policy.",
    )
    parser.add_argument(
        "--invert-opponent-actions",
        action=argparse.BooleanOptionalAction,
        default=defaults["invert_opponent_actions"],
        help=(
            "Negate fixed-opponent actions before composing full env action. "
            "Keep enabled to match opponent-frame action convention used elsewhere."
        ),
    )
    parser.add_argument("--num-envs", type=int, default=defaults["num_envs"])
    parser.add_argument("--total-timesteps", type=int, default=defaults["total_timesteps"])
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--n-steps", type=int, default=defaults["n_steps"])
    parser.add_argument("--gamma", type=float, default=defaults["gamma"])
    parser.add_argument("--gae-lambda", type=float, default=defaults["gae_lambda"])
    parser.add_argument("--ent-coef", type=float, default=defaults["ent_coef"])
    parser.add_argument("--clip-range", type=float, default=defaults["clip_range"])
    parser.add_argument("--vf-coef", type=float, default=defaults["vf_coef"])
    parser.add_argument("--n-epochs", type=int, default=defaults["n_epochs"])
    parser.add_argument("--n-disc-updates", type=int, default=defaults["n_disc_updates"])
    parser.add_argument(
        "--adaptive-disc-updates",
        action=argparse.BooleanOptionalAction,
        default=defaults["adaptive_disc_updates"],
        help=(
            "Adapt n_disc_updates_per_round online: decrease when disc_acc is too high "
            "and increase when disc_acc is too low."
        ),
    )
    parser.add_argument(
        "--disc-acc-low",
        type=float,
        default=defaults["disc_acc_low"],
        help="Increase discriminator updates when disc_acc is below this threshold.",
    )
    parser.add_argument(
        "--disc-acc-high",
        type=float,
        default=defaults["disc_acc_high"],
        help="Decrease discriminator updates when disc_acc is above this threshold.",
    )
    parser.add_argument(
        "--min-disc-updates",
        type=int,
        default=defaults["min_disc_updates"],
        help="Lower bound for adaptive n_disc_updates_per_round.",
    )
    parser.add_argument(
        "--max-disc-updates",
        type=int,
        default=defaults["max_disc_updates"],
        help="Upper bound for adaptive n_disc_updates_per_round.",
    )
    parser.add_argument(
        "--balance-disc-expert-loss",
        action=argparse.BooleanOptionalAction,
        default=defaults["balance_disc_expert_loss"],
        help=(
            "Upweight expert-class BCE term when disc_acc_gen exceeds disc_acc_expert "
            "by more than --disc-balance-gap-target."
        ),
    )
    parser.add_argument(
        "--disc-balance-gap-target",
        type=float,
        default=defaults["disc_balance_gap_target"],
        help=(
            "Allowed disc_acc_gen - disc_acc_expert gap before extra expert-class "
            "loss weighting starts."
        ),
    )
    parser.add_argument(
        "--disc-expert-loss-gain",
        type=float,
        default=defaults["disc_expert_loss_gain"],
        help="Gain for expert loss upweighting when the class-accuracy gap is positive.",
    )
    parser.add_argument(
        "--disc-expert-loss-max-weight",
        type=float,
        default=defaults["disc_expert_loss_max_weight"],
        help="Upper bound on expert-class BCE weight multiplier.",
    )
    parser.add_argument(
        "--freeze-disc-on-high-gen-acc",
        action=argparse.BooleanOptionalAction,
        default=defaults["freeze_disc_on_high_gen_acc"],
        help="Freeze discriminator updates when generated-sample accuracy gets too high.",
    )
    parser.add_argument(
        "--disc-gen-acc-freeze-threshold",
        type=float,
        default=defaults["disc_gen_acc_freeze_threshold"],
        help="Freeze trigger on discriminator accuracy over generated samples.",
    )
    parser.add_argument(
        "--disc-gen-exp-gap-freeze-threshold",
        type=float,
        default=defaults["disc_gen_exp_gap_freeze_threshold"],
        help=(
            "Freeze trigger requires (disc_acc_gen - disc_acc_expert) to exceed this "
            "gap (set 0 to ignore the gap condition)."
        ),
    )
    parser.add_argument(
        "--disc-freeze-rounds",
        type=int,
        default=defaults["disc_freeze_rounds"],
        help="Rounds worth of discriminator steps to skip once a freeze is triggered.",
    )
    parser.add_argument(
        "--strict-disc-freeze",
        action=argparse.BooleanOptionalAction,
        default=defaults["strict_disc_freeze"],
        help=(
            "Permanently freeze discriminator updates after first freeze trigger "
            "(overrides round-based skipping once active)."
        ),
    )
    parser.add_argument(
        "--freeze-disc-until-gen-acc-below-threshold",
        action=argparse.BooleanOptionalAction,
        default=defaults["freeze_disc_until_gen_acc_below_threshold"],
        help=(
            "Freeze discriminator updates after trigger until probed "
            "disc_acc_gen drops below --disc-gen-acc-unfreeze-threshold."
        ),
    )
    parser.add_argument(
        "--disc-gen-acc-unfreeze-threshold",
        type=float,
        default=defaults["disc_gen_acc_unfreeze_threshold"],
        help=(
            "Unfreeze threshold for --freeze-disc-until-gen-acc-below-threshold. "
            "Defaults to --disc-gen-acc-freeze-threshold if omitted."
        ),
    )
    parser.add_argument("--disc-weight-decay", type=float, default=defaults["disc_weight_decay"])
    parser.add_argument("--disc-learning-rate", type=float, default=defaults["disc_learning_rate"])
    parser.add_argument(
        "--disc-label-smoothing",
        type=float,
        default=defaults["disc_label_smoothing"],
    )
    parser.add_argument(
        "--disc-reward-hid-sizes",
        type=str,
        default=",".join(str(x) for x in defaults["disc_reward_hid_sizes"]),
        help="Comma-separated hidden sizes for reward network MLP head (e.g. 32 or 64,64).",
    )
    parser.add_argument(
        "--disc-potential-hid-sizes",
        type=str,
        default=",".join(str(x) for x in defaults["disc_potential_hid_sizes"]),
        help="Comma-separated hidden sizes for potential network head (e.g. 32,32).",
    )
    parser.add_argument(
        "--event-aware-disc-weighting",
        action=argparse.BooleanOptionalAction,
        default=defaults["event_aware_disc_weighting"],
        help=(
            "Use transition infos termination flags to reweight discriminator BCE: "
            "upweight expert good events and generated bad events."
        ),
    )
    parser.add_argument(
        "--event-expert-goal-weight",
        type=float,
        default=defaults["event_expert_goal_weight"],
        help="Multiplier for expert transitions with goal_scored/opponent_in_goal.",
    )
    parser.add_argument(
        "--event-expert-bad-weight",
        type=float,
        default=defaults["event_expert_bad_weight"],
        help="Multiplier for expert transitions with goal_conceded/player_in_goal.",
    )
    parser.add_argument(
        "--event-expert-timeout-weight",
        type=float,
        default=defaults["event_expert_timeout_weight"],
        help="Multiplier for expert transitions with time_out.",
    )
    parser.add_argument(
        "--event-generated-goal-weight",
        type=float,
        default=defaults["event_generated_goal_weight"],
        help="Multiplier for generated transitions with goal_scored/opponent_in_goal.",
    )
    parser.add_argument(
        "--event-generated-bad-weight",
        type=float,
        default=defaults["event_generated_bad_weight"],
        help="Multiplier for generated transitions with goal_conceded/player_in_goal.",
    )
    parser.add_argument(
        "--event-generated-timeout-weight",
        type=float,
        default=defaults["event_generated_timeout_weight"],
        help="Multiplier for generated transitions with time_out.",
    )
    parser.add_argument(
        "--event-preterminal-window-steps",
        type=int,
        default=defaults["event_preterminal_window_steps"],
        help=(
            "If > 0, apply event-aware weighting to the final k steps before a terminal event "
            "for both expert and generated trajectories."
        ),
    )
    parser.add_argument(
        "--event-preterminal-decay",
        type=float,
        default=defaults["event_preterminal_decay"],
        help=(
            "Geometric decay applied across the pre-terminal window; 1.0 means constant weight, "
            "<1 decays with distance from the terminal event."
        ),
    )
    parser.add_argument("--min-len", type=int, default=defaults["min_len"])
    parser.add_argument(
        "--strip-sim-action-history",
        action=argparse.BooleanOptionalAction,
        default=defaults["strip_sim_action_history"],
        help=(
            "For sim_native demos, keep only first 12 observation dims "
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
        "--masked-velocities",
        action="store_true",
        default=defaults["masked_velocities"],
        help=(
            "Zero demonstration observation channels [8:12] (velocity/action-like channels). "
            "Default: disabled."
        ),
    )
    # Deprecated: use --masked-velocities.
    parser.add_argument(
        "--zero-mask-velocity-obs-human",
        action=argparse.BooleanOptionalAction,
        default=defaults["zero_mask_velocity_obs_human"],
        help=argparse.SUPPRESS,
    )
    # Deprecated: use --masked-velocities.
    parser.add_argument(
        "--zero-mask-velocity-obs-sim",
        action=argparse.BooleanOptionalAction,
        default=defaults["zero_mask_velocity_obs_sim"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--val-split-fraction",
        type=float,
        default=defaults["val_split_fraction"],
        help=(
            "Validation holdout fraction for demonstrations. "
            "Applies before augmentation. Set 0 with --no-val-split to disable."
        ),
    )
    parser.add_argument(
        "--val-split-level",
        type=str,
        choices=["file", "trajectory"],
        default=defaults["val_split_level"],
        help="Validation split granularity (default: file-level split).",
    )
    parser.add_argument(
        "--val-split-seed",
        type=int,
        default=None,
        help="Seed for train/validation split. Defaults to --seed.",
    )
    parser.add_argument(
        "--no-val-split",
        action="store_true",
        help="Disable validation split and train on all demonstrations.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default=defaults["wandb_project"])
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default=defaults["wandb_mode"],
    )
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-train-log-freq-timesteps",
        type=int,
        default=defaults["wandb_train_log_freq_timesteps"],
        help="W&B train-metric log frequency in timesteps.",
    )
    # Deprecated autoscaling args are kept as hidden no-ops for backward compatibility.
    parser.add_argument(
        "--auto-scale-total-timesteps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--timesteps-scale-rule",
        type=str,
        choices=["sqrt", "linear", "none"],
        default="none",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--timesteps-scale-base-steps",
        type=int,
        default=80_000,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--timesteps-scale-min-factor",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--timesteps-scale-max-factor",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    # Backward-compatible deprecated flag: accepted but ignored.
    parser.add_argument(
        "--auto-scale-batch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,
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
    peg1_only_fixed_opponent = bool(args.peg1_only_fixed_opponent)
    fixed_opponent_policy_path = (
        None
        if args.fixed_opponent_policy_path is None
        else os.path.abspath(os.path.expanduser(str(args.fixed_opponent_policy_path)))
    )
    fixed_opponent_policy_format = None
    fixed_opponent_rl_games_config_path = (
        None
        if args.fixed_opponent_rl_games_config_path is None
        else os.path.abspath(os.path.expanduser(str(args.fixed_opponent_rl_games_config_path)))
    )
    bc_bootstrap_runs_root = str(normalize_path(args.bc_bootstrap_runs_root))
    init_generator_from_bc_path = normalize_path(args.init_generator_from_bc_path)
    init_generator_from_bc_auto = bool(args.init_generator_from_bc_auto)
    init_generator_from_bc_mode = str(args.init_generator_from_bc_mode)
    bc_anchor_during_gail = bool(args.bc_anchor_during_gail)
    bc_anchor_batch_size = max(1, int(args.bc_anchor_batch_size))
    bc_anchor_updates_per_round = max(1, int(args.bc_anchor_updates_per_round))
    bc_anchor_learning_rate = float(args.bc_anchor_learning_rate)
    bc_anchor_ent_weight = float(args.bc_anchor_ent_weight)
    bc_anchor_l2_weight = float(args.bc_anchor_l2_weight)
    bc_anchor_decay_fraction = float(args.bc_anchor_decay_fraction)
    save_intermediate_checkpoints = bool(args.save_intermediate_checkpoints)
    checkpoint_interval_timesteps = max(1, int(args.checkpoint_interval_timesteps))
    reward_net_type = str(args.reward_net_type)
    fixed_opponent_deterministic = bool(args.fixed_opponent_deterministic)
    invert_opponent_actions = bool(args.invert_opponent_actions)
    if init_generator_from_bc_path is not None and init_generator_from_bc_auto:
        raise ValueError(
            "--init-generator-from-bc-path and --init-generator-from-bc-auto "
            "cannot both be enabled."
        )
    if init_generator_from_bc_path is not None and not os.path.isfile(init_generator_from_bc_path):
        raise FileNotFoundError(
            f"BC bootstrap checkpoint not found: {init_generator_from_bc_path}"
        )
    if init_generator_from_bc_auto and not os.path.isdir(bc_bootstrap_runs_root):
        raise FileNotFoundError(f"BC bootstrap runs root not found: {bc_bootstrap_runs_root}")
    if peg1_only_fixed_opponent:
        if not fixed_opponent_policy_path:
            raise ValueError(
                "--peg1-only-fixed-opponent requires --fixed-opponent-policy-path."
            )
        if not os.path.exists(fixed_opponent_policy_path):
            raise FileNotFoundError(
                f"Fixed opponent policy not found: {fixed_opponent_policy_path}"
            )
        fixed_opponent_policy_format = resolve_policy_format_value(
            args.fixed_opponent_policy_format,
            checkpoint=fixed_opponent_policy_path,
            flag_name="--fixed-opponent-policy-format",
        )
        if fixed_opponent_policy_format == POLICY_FORMAT_RL_GAMES:
            if not fixed_opponent_rl_games_config_path:
                raise ValueError(
                    "--fixed-opponent-rl-games-config-path is required when the fixed opponent "
                    f"uses {POLICY_FORMAT_RL_GAMES} format."
                )
            if not os.path.isfile(fixed_opponent_rl_games_config_path):
                raise FileNotFoundError(
                    "Fixed opponent RL-Games config not found: "
                    f"{fixed_opponent_rl_games_config_path}"
                )
    if str(_APP_LAUNCH_DEVICE).strip().lower() != str(device).strip().lower():
        print(
            "[warn] AppLauncher was initialized with "
            f"device={_APP_LAUNCH_DEVICE}, but training resolved --device to {device}."
        )
    effective_num_envs = int(args.num_envs)
    requested_total_timesteps = int(args.total_timesteps)
    effective_total_timesteps = int(max(1, requested_total_timesteps))
    adaptive_disc_updates = bool(args.adaptive_disc_updates)
    disc_acc_low = float(args.disc_acc_low)
    disc_acc_high = float(args.disc_acc_high)
    min_disc_updates = max(1, int(args.min_disc_updates))
    max_disc_updates = max(min_disc_updates, int(args.max_disc_updates))
    balance_disc_expert_loss = bool(args.balance_disc_expert_loss)
    disc_balance_gap_target = float(args.disc_balance_gap_target)
    disc_expert_loss_gain = max(0.0, float(args.disc_expert_loss_gain))
    disc_expert_loss_max_weight = max(1.0, float(args.disc_expert_loss_max_weight))
    freeze_disc_on_high_gen_acc = bool(args.freeze_disc_on_high_gen_acc)
    disc_gen_acc_freeze_threshold = float(args.disc_gen_acc_freeze_threshold)
    disc_gen_exp_gap_freeze_threshold = max(0.0, float(args.disc_gen_exp_gap_freeze_threshold))
    disc_freeze_rounds = max(1, int(args.disc_freeze_rounds))
    strict_disc_freeze = bool(args.strict_disc_freeze)
    freeze_disc_until_gen_acc_below_threshold = bool(
        args.freeze_disc_until_gen_acc_below_threshold
    )
    disc_gen_acc_unfreeze_threshold = (
        None
        if args.disc_gen_acc_unfreeze_threshold is None
        else float(args.disc_gen_acc_unfreeze_threshold)
    )
    event_aware_disc_weighting = bool(args.event_aware_disc_weighting)
    event_expert_goal_weight = float(args.event_expert_goal_weight)
    event_expert_bad_weight = float(args.event_expert_bad_weight)
    event_expert_timeout_weight = float(args.event_expert_timeout_weight)
    event_generated_goal_weight = float(args.event_generated_goal_weight)
    event_generated_bad_weight = float(args.event_generated_bad_weight)
    event_generated_timeout_weight = float(args.event_generated_timeout_weight)
    event_preterminal_window_steps = max(0, int(args.event_preterminal_window_steps))
    event_preterminal_decay = float(args.event_preterminal_decay)
    if disc_acc_low >= disc_acc_high:
        raise ValueError(
            "--disc-acc-low must be strictly smaller than --disc-acc-high "
            f"(got {disc_acc_low} >= {disc_acc_high})."
        )
    if disc_gen_acc_freeze_threshold <= 0.0 or disc_gen_acc_freeze_threshold >= 1.0:
        raise ValueError(
            "--disc-gen-acc-freeze-threshold must be in (0, 1); "
            f"got {disc_gen_acc_freeze_threshold}."
        )
    if strict_disc_freeze and freeze_disc_until_gen_acc_below_threshold:
        raise ValueError(
            "--strict-disc-freeze and --freeze-disc-until-gen-acc-below-threshold "
            "cannot both be enabled."
        )
    if freeze_disc_until_gen_acc_below_threshold:
        if disc_gen_acc_unfreeze_threshold is None:
            disc_gen_acc_unfreeze_threshold = float(disc_gen_acc_freeze_threshold)
        if disc_gen_acc_unfreeze_threshold <= 0.0 or disc_gen_acc_unfreeze_threshold >= 1.0:
            raise ValueError(
                "--disc-gen-acc-unfreeze-threshold must be in (0, 1) when "
                "--freeze-disc-until-gen-acc-below-threshold is enabled."
            )
        if disc_gen_acc_unfreeze_threshold > disc_gen_acc_freeze_threshold:
            raise ValueError(
                "--disc-gen-acc-unfreeze-threshold must be <= "
                "--disc-gen-acc-freeze-threshold."
            )
    for weight_name, weight_value in (
        ("--event-expert-goal-weight", event_expert_goal_weight),
        ("--event-expert-bad-weight", event_expert_bad_weight),
        ("--event-expert-timeout-weight", event_expert_timeout_weight),
        ("--event-generated-goal-weight", event_generated_goal_weight),
        ("--event-generated-bad-weight", event_generated_bad_weight),
        ("--event-generated-timeout-weight", event_generated_timeout_weight),
    ):
        if weight_value <= 0.0:
            raise ValueError(f"{weight_name} must be > 0, got {weight_value}.")
    if bc_anchor_learning_rate <= 0.0:
        raise ValueError(
            f"--bc-anchor-learning-rate must be > 0, got {bc_anchor_learning_rate}."
        )
    if bc_anchor_decay_fraction <= 0.0:
        raise ValueError(
            f"--bc-anchor-decay-fraction must be > 0, got {bc_anchor_decay_fraction}."
        )
    if checkpoint_interval_timesteps <= 0:
        raise ValueError(
            f"--checkpoint-interval-timesteps must be > 0, got {checkpoint_interval_timesteps}."
        )
    if event_preterminal_decay <= 0.0:
        raise ValueError(
            f"--event-preterminal-decay must be > 0, got {event_preterminal_decay}."
        )
    initial_disc_updates = int(args.n_disc_updates)
    if initial_disc_updates < min_disc_updates or initial_disc_updates > max_disc_updates:
        clamped_updates = min(max(initial_disc_updates, min_disc_updates), max_disc_updates)
        print(
            "[warn] --n-disc-updates outside adaptive bounds; "
            f"clamping {initial_disc_updates} -> {clamped_updates}."
        )
        initial_disc_updates = clamped_updates

    if bool(getattr(args, "auto_scale_batch", False)):
        print("[warn] --auto-scale-batch is deprecated and ignored. num_envs remains fixed.")
    if (
        bool(getattr(args, "auto_scale_total_timesteps", False))
        or str(getattr(args, "timesteps_scale_rule", "none")).strip().lower() != "none"
        or float(getattr(args, "timesteps_scale_min_factor", 1.0)) != 1.0
        or float(getattr(args, "timesteps_scale_max_factor", 1.0)) != 1.0
    ):
        print(
            "[warn] timestep autoscaling flags are deprecated and ignored. "
            "Using fixed --total-timesteps."
        )
    if float(args.val_split_fraction) < 0.0 or float(args.val_split_fraction) >= 1.0:
        raise ValueError(
            "--val-split-fraction must be in [0, 1), "
            f"got {float(args.val_split_fraction)}."
        )

    val_split_seed = int(args.seed if args.val_split_seed is None else args.val_split_seed)
    val_split_fraction = 0.0 if bool(args.no_val_split) else float(args.val_split_fraction)
    val_split_level = str(args.val_split_level).strip().lower()

    fixed_opponent_policy = None
    venv = make_env(
        num_envs=effective_num_envs,
        seed=args.seed,
        sim_device=device,
        peg1_only_fixed_opponent=peg1_only_fixed_opponent,
        opponent_policy=fixed_opponent_policy,
        deterministic_opponent=fixed_opponent_deterministic,
        invert_opponent_actions=invert_opponent_actions,
        preterminal_window_steps=event_preterminal_window_steps,
        action_limit=float(args.action_limit),
        actuator_model=bool(args.actuator_model),
    )
    print(
        f"[setup] generator action bound +/-{float(args.action_limit):.3g} m/s; "
        f"actuator model {'ON' if args.actuator_model else 'OFF'}.",
        flush=True,
    )
    if peg1_only_fixed_opponent:
        print(
            "[setup] loading fixed opponent policy: "
            f"path={fixed_opponent_policy_path} format={fixed_opponent_policy_format}"
        )
        if fixed_opponent_policy_format == POLICY_FORMAT_SB3_PPO:
            fixed_opponent_policy = PPO.load(fixed_opponent_policy_path, device=device)
        elif fixed_opponent_policy_format == POLICY_FORMAT_RL_GAMES:
            full_action_space = getattr(venv.unwrapped, "action_space", venv.action_space)
            fixed_opponent_policy = build_rl_games_predict_policy(
                checkpoint_path=fixed_opponent_policy_path,
                config_path=str(fixed_opponent_rl_games_config_path),
                device=device,
                num_envs=int(effective_num_envs),
                observation_space=venv.observation_space,
                action_space=full_action_space,
            )
        else:
            raise ValueError(
                f"Unsupported fixed opponent format: {fixed_opponent_policy_format!r}"
            )
        if hasattr(venv, "venv") and isinstance(venv.venv, FixedOpponentVecEnvWrapper):
            venv.venv.opponent_policy = fixed_opponent_policy
        else:
            raise RuntimeError("Expected FixedOpponentVecEnvWrapper when loading fixed opponent.")
    wandb_run = None
    gail_round_callback = None
    bc_bootstrap_info = BCBootstrapInfo(
        enabled=False,
        init_source=None,
        mode=None,
        policy_path=None,
        run_dir=None,
        run_tag=None,
        demo_base=None,
        demo_source_format=None,
        masked_velocities=None,
        strip_sim_action_history=None,
        zero_mask_velocity_obs_human=None,
        zero_mask_velocity_obs_sim=None,
        val_split_fraction=None,
        seed=None,
        created_at=None,
    )

    try:
        print("[dataset] loading demonstrations...")
        if args.augment is None:
            print(
                f"[dataset] --augment not set; defaulting to augment={augment_enabled} "
                f"for source={args.demo_source_format}."
            )
        else:
            print(f"[dataset] using explicit augment={augment_enabled}.")
        pkl_paths = collect_pkl_paths(args.demo_base)
        demo_action_space_for_load = (
            venv.unwrapped.action_space if peg1_only_fixed_opponent else venv.action_space
        )
        raw_trajectories, group_ids, raw_steps = load_trajectories(
            pkl_paths,
            action_space=demo_action_space_for_load,
            source_format=args.demo_source_format,
            min_len=args.min_len,
        )
        print(
            "[dataset] source={src} files={n_files} raw_traj={n_traj} raw_steps={n_steps}".format(
                src=args.demo_source_format,
                n_files=len(pkl_paths),
                n_traj=len(raw_trajectories),
                n_steps=int(raw_steps),
            )
        )
        print(
            "[timesteps] fixed total_timesteps={ts:,}".format(
                ts=int(effective_total_timesteps),
            )
        )

        if (
            args.demo_source_format == DEMO_SOURCE_SIM_NATIVE
            and bool(args.strip_sim_action_history)
        ):
            raw_trajectories, stripped_count, seen_dims = strip_sim_native_action_history(
                raw_trajectories,
                keep_obs_dim=BASE_OBS_DIM,
            )
            print(
                "[dataset] strip_sim_action_history=True "
                f"seen_obs_dims={seen_dims} stripped_traj={stripped_count}"
            )

        masked_velocities = bool(args.masked_velocities)
        if args.demo_source_format == DEMO_SOURCE_HUMAN_RAW:
            masked_velocities = masked_velocities or bool(args.zero_mask_velocity_obs_human)
        elif args.demo_source_format == DEMO_SOURCE_SIM_NATIVE:
            masked_velocities = masked_velocities or bool(args.zero_mask_velocity_obs_sim)

        if masked_velocities:
            raw_trajectories = zero_mask_observation_channels(
                raw_trajectories,
                start=MASK_ACTION_LIKE_OBS_START,
                end=MASK_ACTION_LIKE_OBS_END,
            )
            print(
                "[dataset] zero-masked observation channels "
                f"[{MASK_ACTION_LIKE_OBS_START}:{MASK_ACTION_LIKE_OBS_END}] "
                f"for source={args.demo_source_format}."
            )

        train_raw_trajs, val_trajs, train_indices, val_indices = split_train_val_trajectories(
            raw_trajectories,
            group_ids=group_ids,
            val_fraction=float(val_split_fraction),
            seed=int(val_split_seed),
            split_level=val_split_level,
        )
        if event_preterminal_window_steps > 0:
            train_raw_trajs = annotate_preterminal_event_windows(
                train_raw_trajs,
                window_steps=event_preterminal_window_steps,
            )
            print(
                "[dataset] annotated expert pre-terminal windows: "
                f"k={event_preterminal_window_steps} steps"
            )
        if augment_enabled:
            trajectories, train_aug_n = augment_with_axis_mirrors(train_raw_trajs)
        else:
            trajectories = list(train_raw_trajs)
            train_aug_n = 0
        if peg1_only_fixed_opponent:
            trajectories = slice_peg1_actions(
                trajectories,
                action_start=0,
                action_dim=2,
            )
            print(
                "[dataset] peg1_only_fixed_opponent=True: "
                "sliced demo actions to first 2 dims (peg1 controls only)."
            )

        train_raw_steps = int(sum(len(tr.acts) for tr in train_raw_trajs))
        val_steps = int(sum(len(tr.acts) for tr in val_trajs))
        train_steps = int(sum(len(tr.acts) for tr in trajectories))

        train_files = sorted({str(group_ids[idx]) for idx in train_indices})
        val_files = sorted({str(group_ids[idx]) for idx in val_indices})
        split_enabled = bool(val_split_fraction > 0.0 and len(val_indices) > 0)
        dataset_split = {
            "enabled": bool(split_enabled),
            "split_level": str(val_split_level),
            "val_split_fraction": float(val_split_fraction),
            "val_split_seed": int(val_split_seed),
            "peg1_only_fixed_opponent": bool(peg1_only_fixed_opponent),
            "demo_actions_sliced_to_peg1": bool(peg1_only_fixed_opponent),
            "train_trajectories_raw": int(len(train_raw_trajs)),
            "val_trajectories_raw": int(len(val_trajs)),
            "train_steps_raw": int(train_raw_steps),
            "val_steps_raw": int(val_steps),
            "train_steps_used": int(train_steps),
            "train_trajectories_used": int(len(trajectories)),
            "train_augmented_trajectories_added": int(train_aug_n),
            "train_files": train_files,
            "val_files": val_files,
        }
        print(
            "[dataset split] level={level} enabled={enabled} fraction={frac:.3f} seed={seed} | "
            "train(raw): {n_train_raw} traj, {train_raw_steps} steps | "
            "val(raw): {n_val} traj, {val_steps} steps | "
            "train(used): {n_train_used} traj, {train_steps} steps".format(
                level=val_split_level,
                enabled=split_enabled,
                frac=float(val_split_fraction),
                seed=int(val_split_seed),
                n_train_raw=len(train_raw_trajs),
                train_raw_steps=train_raw_steps,
                n_val=len(val_trajs),
                val_steps=val_steps,
                n_train_used=len(trajectories),
                train_steps=train_steps,
            )
        )

        dataset_stats = DatasetLoadStats(
            source_format=str(args.demo_source_format),
            n_files=int(len(pkl_paths)),
            n_traj_raw=int(len(raw_trajectories)),
            n_steps_raw=int(raw_steps),
            n_traj_after_aug=int(len(trajectories)),
            n_steps_after_aug=int(train_steps),
        )

        if trajectories and not isinstance(trajectories[0].obs, dict):
            demo_obs_dim = int(np.asarray(trajectories[0].obs).shape[-1])
            env_obs_dim = int(np.prod(venv.observation_space.shape))
            if demo_obs_dim != env_obs_dim:
                raise ValueError(
                    f"Observation dimension mismatch after preprocessing: "
                    f"demo_obs_dim={demo_obs_dim}, env_obs_dim={env_obs_dim}."
                )

        if init_generator_from_bc_path is not None:
            bc_bootstrap_info = _load_bc_bootstrap_metadata_from_policy_path(
                init_generator_from_bc_path,
                init_source="explicit",
                mode=init_generator_from_bc_mode,
            )
        elif init_generator_from_bc_auto:
            bc_bootstrap_info = resolve_bc_bootstrap_info(
                bc_runs_root=bc_bootstrap_runs_root,
                demo_base=args.demo_base,
                demo_source_format=str(args.demo_source_format),
                masked_velocities=bool(masked_velocities),
                strip_sim_action_history=bool(args.strip_sim_action_history),
                zero_mask_velocity_obs_human=bool(args.zero_mask_velocity_obs_human),
                zero_mask_velocity_obs_sim=bool(args.zero_mask_velocity_obs_sim),
                init_mode=init_generator_from_bc_mode,
            )

        if bc_bootstrap_info.enabled:
            if bc_bootstrap_info.val_split_fraction is not None:
                bc_bootstrap_info.split_fraction_matches_current = bool(
                    abs(float(bc_bootstrap_info.val_split_fraction) - float(val_split_fraction))
                    < 1e-12
                )
            if bc_bootstrap_info.seed is not None:
                bc_bootstrap_info.seed_matches_current = bool(
                    int(bc_bootstrap_info.seed) == int(val_split_seed)
                )
            print(
                "[bc bootstrap] source={src} mode={mode} policy={policy} run_tag={run_tag}".format(
                    src=bc_bootstrap_info.init_source,
                    mode=bc_bootstrap_info.mode,
                    policy=bc_bootstrap_info.policy_path,
                    run_tag=bc_bootstrap_info.run_tag,
                )
            )
            if (
                bc_bootstrap_info.val_split_fraction is not None
                and bc_bootstrap_info.split_fraction_matches_current is False
            ):
                print(
                    "[warn] BC bootstrap checkpoint was trained with "
                    f"val_split_fraction={bc_bootstrap_info.val_split_fraction}, "
                    f"but current GAIL run uses {float(val_split_fraction)}."
                )
            if bc_bootstrap_info.seed is not None and bc_bootstrap_info.seed_matches_current is False:
                print(
                    "[warn] BC bootstrap checkpoint was trained with "
                    f"seed={bc_bootstrap_info.seed}, but current GAIL split seed is "
                    f"{int(val_split_seed)}."
                )

        policy_name = (
            "MultiInputPolicy"
            if isinstance(venv.observation_space, gym.spaces.Dict)
            else "MlpPolicy"
        )
        learner = PPO(
            policy_name,
            venv,
            device=device,
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            n_steps=int(args.n_steps),
            ent_coef=float(args.ent_coef),
            clip_range=float(args.clip_range),
            vf_coef=float(args.vf_coef),
            n_epochs=int(args.n_epochs),
            seed=int(args.seed),
            verbose=1,
        )
        if bc_bootstrap_info.enabled:
            bootstrap_stats = bootstrap_generator_from_bc(
                learner,
                bc_policy_path=str(bc_bootstrap_info.policy_path),
                init_mode=str(init_generator_from_bc_mode),
                device=device,
            )
            for key, value in bootstrap_stats.items():
                setattr(bc_bootstrap_info, key, int(value))
            print(
                "[bc bootstrap] loaded_keys={loaded}/{total} skipped_value={skip_value} "
                "skipped_shape={skip_shape} skipped_missing={skip_missing}".format(
                    loaded=bc_bootstrap_info.loaded_key_count,
                    total=bc_bootstrap_info.source_key_count,
                    skip_value=bc_bootstrap_info.skipped_value_key_count,
                    skip_shape=bc_bootstrap_info.skipped_shape_mismatch_key_count,
                    skip_missing=bc_bootstrap_info.skipped_missing_target_key_count,
                )
            )

        disc_reward_hid_sizes = parse_hid_sizes(args.disc_reward_hid_sizes)
        disc_potential_hid_sizes = parse_hid_sizes(args.disc_potential_hid_sizes)
        reward_net_kwargs: Dict[str, Any] = {
            "observation_space": venv.observation_space,
            "action_space": venv.action_space,
            "normalize_input_layer": RunningNorm,
        }
        if disc_reward_hid_sizes:
            reward_net_kwargs["reward_hid_sizes"] = tuple(disc_reward_hid_sizes)
        if disc_potential_hid_sizes:
            reward_net_kwargs["potential_hid_sizes"] = tuple(disc_potential_hid_sizes)
        if reward_net_type == REWARD_NET_TYPE_BASIC_SHAPED:
            reward_net = BasicShapedRewardNet(**reward_net_kwargs).to(device)
        elif reward_net_type == REWARD_NET_TYPE_KLASK_RELATIVE_SHAPED:
            reward_net = KlaskRelativeShapedRewardNet(**reward_net_kwargs).to(device)
        else:
            raise ValueError(f"Unsupported reward_net_type={reward_net_type!r}")

        n_demo_steps = int(sum(len(tr.acts) for tr in trajectories))
        rollout_batch = int(effective_num_envs) * int(args.n_steps)
        demo_batch_size = min(n_demo_steps, max(2048, rollout_batch // 2))
        if demo_batch_size < 1:
            raise ValueError("No valid demonstration steps loaded for GAIL training.")

        disc_opt_kwargs: Optional[Dict[str, Any]] = {}
        if float(args.disc_learning_rate) > 0:
            disc_opt_kwargs["lr"] = float(args.disc_learning_rate)
        if float(args.disc_weight_decay) > 0:
            disc_opt_kwargs["weight_decay"] = float(args.disc_weight_decay)
        if not disc_opt_kwargs:
            disc_opt_kwargs = None

        gail_cls = GAILWithLabelSmoothing
        gail_kwargs: Dict[str, Any] = {
            "label_smoothing": float(args.disc_label_smoothing),
            "adaptive_disc_updates": adaptive_disc_updates,
            "disc_acc_low": disc_acc_low,
            "disc_acc_high": disc_acc_high,
            "min_disc_updates": min_disc_updates,
            "max_disc_updates": max_disc_updates,
            "balance_disc_expert_loss": balance_disc_expert_loss,
            "disc_balance_gap_target": disc_balance_gap_target,
            "disc_expert_loss_gain": disc_expert_loss_gain,
            "disc_expert_loss_max_weight": disc_expert_loss_max_weight,
            "freeze_disc_on_high_gen_acc": freeze_disc_on_high_gen_acc,
            "disc_gen_acc_freeze_threshold": disc_gen_acc_freeze_threshold,
            "disc_gen_exp_gap_freeze_threshold": disc_gen_exp_gap_freeze_threshold,
            "disc_freeze_rounds": disc_freeze_rounds,
            "strict_disc_freeze": strict_disc_freeze,
            "freeze_disc_until_gen_acc_below_threshold": (
                freeze_disc_until_gen_acc_below_threshold
            ),
            "disc_gen_acc_unfreeze_threshold": disc_gen_acc_unfreeze_threshold,
            "event_aware_disc_weighting": event_aware_disc_weighting,
            "event_expert_goal_weight": event_expert_goal_weight,
            "event_expert_bad_weight": event_expert_bad_weight,
            "event_expert_timeout_weight": event_expert_timeout_weight,
            "event_generated_goal_weight": event_generated_goal_weight,
            "event_generated_bad_weight": event_generated_bad_weight,
            "event_generated_timeout_weight": event_generated_timeout_weight,
            "event_preterminal_window_steps": event_preterminal_window_steps,
            "event_preterminal_decay": event_preterminal_decay,
        }

        print(
            "[gail] demos={n_demo} traj, {n_demo_steps} steps | "
            "rollout_batch={rollout_batch} | demo_batch_size={demo_batch_size}".format(
                n_demo=len(trajectories),
                n_demo_steps=n_demo_steps,
                rollout_batch=rollout_batch,
                demo_batch_size=demo_batch_size,
            )
        )
        if peg1_only_fixed_opponent:
            print(
                "[gail] peg1_only_fixed_opponent=True "
                f"(opponent={fixed_opponent_policy_path}, "
                f"format={fixed_opponent_policy_format}, "
                f"deterministic={fixed_opponent_deterministic}, "
                f"invert_actions={invert_opponent_actions})"
            )
        print(
            f"[gail] training timesteps={int(effective_total_timesteps):,} "
            f"(requested={int(requested_total_timesteps):,}), "
            f"n_disc_updates={int(initial_disc_updates)}, device={device}"
        )
        print(
            "[gail] adaptive_disc_updates="
            f"{adaptive_disc_updates} (disc_acc_low={disc_acc_low:.3f}, "
            f"disc_acc_high={disc_acc_high:.3f}, "
            f"min_disc_updates={min_disc_updates}, max_disc_updates={max_disc_updates})"
        )
        print(
            "[gail] class_balance="
            f"{balance_disc_expert_loss} (gap_target={disc_balance_gap_target:.3f}, "
            f"expert_loss_gain={disc_expert_loss_gain:.3f}, "
            f"expert_loss_max_weight={disc_expert_loss_max_weight:.3f})"
        )
        print(
            "[gail] freeze_on_high_gen_acc="
            f"{freeze_disc_on_high_gen_acc} (gen_acc_threshold={disc_gen_acc_freeze_threshold:.3f}, "
            f"gen_exp_gap_threshold={disc_gen_exp_gap_freeze_threshold:.3f}, "
            f"freeze_rounds={disc_freeze_rounds}, strict={strict_disc_freeze}, "
            "until_below_threshold="
            f"{freeze_disc_until_gen_acc_below_threshold}, "
            f"unfreeze_threshold={disc_gen_acc_unfreeze_threshold})"
        )
        print(
            "[gail] event_aware_disc_weighting="
            f"{event_aware_disc_weighting} "
            f"(expert_goal={event_expert_goal_weight:.3f}, "
            f"expert_bad={event_expert_bad_weight:.3f}, "
            f"expert_timeout={event_expert_timeout_weight:.3f}, "
            f"generated_goal={event_generated_goal_weight:.3f}, "
            f"generated_bad={event_generated_bad_weight:.3f}, "
            f"generated_timeout={event_generated_timeout_weight:.3f}, "
            f"preterminal_window={event_preterminal_window_steps}, "
            f"preterminal_decay={event_preterminal_decay:.3f})"
        )
        print(f"[gail] reward_net_type={reward_net_type}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"klask_gail_{timestamp}_{args.run_tag}"
        run_dir = os.path.join(args.save_root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        if save_intermediate_checkpoints:
            os.makedirs(checkpoints_dir, exist_ok=True)
            print(
                "[checkpoint] intermediate checkpointing enabled "
                f"(interval={checkpoint_interval_timesteps:,} timesteps)"
            )

        gail = gail_cls(
            demonstrations=trajectories,
            venv=venv,
            gen_algo=learner,
            reward_net=reward_net,
            allow_variable_horizon=True,
            demo_batch_size=demo_batch_size,
            gen_replay_buffer_capacity=rollout_batch,
            n_disc_updates_per_round=int(initial_disc_updates),
            disc_opt_kwargs=disc_opt_kwargs,
            **gail_kwargs,
        )

        if bc_anchor_during_gail:
            bc_anchor_cb = BCAnchorDuringGAILCallback(
                demonstrations=trajectories,
                total_timesteps=int(effective_total_timesteps),
                batch_size=int(bc_anchor_batch_size),
                updates_per_round=int(bc_anchor_updates_per_round),
                learning_rate=float(bc_anchor_learning_rate),
                ent_weight=float(bc_anchor_ent_weight),
                l2_weight=float(bc_anchor_l2_weight),
                decay_fraction=float(bc_anchor_decay_fraction),
                verbose=0,
            )
            if gail.gen_callback is None:
                gail.gen_callback = bc_anchor_cb
            else:
                gail.gen_callback = CallbackList([gail.gen_callback, bc_anchor_cb])
            print(
                "[gail] bc_anchor_during_gail=True "
                f"(batch_size={bc_anchor_batch_size}, "
                f"updates_per_round={bc_anchor_updates_per_round}, "
                f"lr={bc_anchor_learning_rate:.2e}, "
                f"decay_fraction={bc_anchor_decay_fraction:.3f})"
            )

        round_callbacks: List[Any] = []
        if save_intermediate_checkpoints:
            round_callbacks.append(
                GAILCheckpointCallback(
                    gen_algo=learner,
                    gail_algo=gail,
                    checkpoints_dir=checkpoints_dir,
                    interval_timesteps=int(checkpoint_interval_timesteps),
                )
            )

        if args.wandb:
            if wandb is None:
                print("[warn] wandb is not installed; proceeding without wandb logging.")
            elif args.wandb_mode == "disabled":
                print("[info] --wandb requested but --wandb-mode=disabled; skipping wandb.")
            else:
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    mode=args.wandb_mode,
                    name=args.wandb_run_name,
                    config={
                        "demo_base": args.demo_base,
                        "demo_source_format": args.demo_source_format,
                        "augment": bool(augment_enabled),
                        "strip_sim_action_history": bool(args.strip_sim_action_history),
                        "bc_bootstrap_runs_root": bc_bootstrap_runs_root,
                        "bc_bootstrap_enabled": bool(bc_bootstrap_info.enabled),
                        "bc_bootstrap": bc_bootstrap_info.to_dict(),
                        "bc_anchor_during_gail": bool(bc_anchor_during_gail),
                        "bc_anchor_batch_size": int(bc_anchor_batch_size),
                        "bc_anchor_updates_per_round": int(bc_anchor_updates_per_round),
                        "bc_anchor_learning_rate": float(bc_anchor_learning_rate),
                        "bc_anchor_ent_weight": float(bc_anchor_ent_weight),
                        "bc_anchor_l2_weight": float(bc_anchor_l2_weight),
                        "bc_anchor_decay_fraction": float(bc_anchor_decay_fraction),
                        "save_intermediate_checkpoints": bool(save_intermediate_checkpoints),
                        "checkpoint_interval_timesteps": int(checkpoint_interval_timesteps),
                        "reward_net_type": reward_net_type,
                "action_limit": float(args.action_limit),
                "actuator_model": bool(args.actuator_model),
                        "masked_velocities": bool(masked_velocities),
                        "zero_mask_velocity_obs_human": bool(args.zero_mask_velocity_obs_human),
                        "zero_mask_velocity_obs_sim": bool(args.zero_mask_velocity_obs_sim),
                        "no_val_split": bool(args.no_val_split),
                        "val_split_fraction": float(val_split_fraction),
                        "val_split_level": str(val_split_level),
                        "val_split_seed": int(val_split_seed),
                        "peg1_only_fixed_opponent": bool(peg1_only_fixed_opponent),
                        "fixed_opponent_policy_path": fixed_opponent_policy_path,
                        "fixed_opponent_policy_format": fixed_opponent_policy_format,
                        "fixed_opponent_rl_games_config_path": fixed_opponent_rl_games_config_path,
                        "fixed_opponent_deterministic": bool(fixed_opponent_deterministic),
                        "invert_opponent_actions": bool(invert_opponent_actions),
                        "requested_num_envs": int(args.num_envs),
                        "num_envs": int(effective_num_envs),
                        "requested_total_timesteps": int(requested_total_timesteps),
                        "total_timesteps": int(effective_total_timesteps),
                        "learning_rate": float(args.learning_rate),
                        "batch_size": int(args.batch_size),
                        "n_steps": int(args.n_steps),
                        "gamma": float(args.gamma),
                        "gae_lambda": float(args.gae_lambda),
                        "ent_coef": float(args.ent_coef),
                        "clip_range": float(args.clip_range),
                        "vf_coef": float(args.vf_coef),
                        "n_epochs": int(args.n_epochs),
                        "n_disc_updates": int(initial_disc_updates),
                        "adaptive_disc_updates": bool(adaptive_disc_updates),
                        "disc_acc_low": float(disc_acc_low),
                        "disc_acc_high": float(disc_acc_high),
                        "min_disc_updates": int(min_disc_updates),
                        "max_disc_updates": int(max_disc_updates),
                        "balance_disc_expert_loss": bool(balance_disc_expert_loss),
                        "disc_balance_gap_target": float(disc_balance_gap_target),
                        "disc_expert_loss_gain": float(disc_expert_loss_gain),
                        "disc_expert_loss_max_weight": float(disc_expert_loss_max_weight),
                        "freeze_disc_on_high_gen_acc": bool(freeze_disc_on_high_gen_acc),
                        "disc_gen_acc_freeze_threshold": float(disc_gen_acc_freeze_threshold),
                        "disc_gen_exp_gap_freeze_threshold": float(
                            disc_gen_exp_gap_freeze_threshold
                        ),
                        "disc_freeze_rounds": int(disc_freeze_rounds),
                        "strict_disc_freeze": bool(strict_disc_freeze),
                        "freeze_disc_until_gen_acc_below_threshold": bool(
                            freeze_disc_until_gen_acc_below_threshold
                        ),
                        "disc_gen_acc_unfreeze_threshold": disc_gen_acc_unfreeze_threshold,
                        "disc_weight_decay": float(args.disc_weight_decay),
                        "disc_learning_rate": float(args.disc_learning_rate),
                        "disc_label_smoothing": float(args.disc_label_smoothing),
                        "disc_reward_hid_sizes": disc_reward_hid_sizes,
                        "disc_potential_hid_sizes": disc_potential_hid_sizes,
                        "event_aware_disc_weighting": bool(event_aware_disc_weighting),
                        "event_expert_goal_weight": float(event_expert_goal_weight),
                        "event_expert_bad_weight": float(event_expert_bad_weight),
                        "event_expert_timeout_weight": float(event_expert_timeout_weight),
                        "event_generated_goal_weight": float(event_generated_goal_weight),
                        "event_generated_bad_weight": float(event_generated_bad_weight),
                        "event_generated_timeout_weight": float(event_generated_timeout_weight),
                        "event_preterminal_window_steps": int(event_preterminal_window_steps),
                        "event_preterminal_decay": float(event_preterminal_decay),
                        "wandb_train_log_freq_timesteps": int(args.wandb_train_log_freq_timesteps),
                        "dataset_stats": dataset_stats.to_dict(),
                        "dataset_split": dataset_split,
                    },
                )
                train_metrics_cb = GAILWandbTrainMetricsCallback(
                    wandb_run=wandb_run,
                    total_timesteps=int(effective_total_timesteps),
                    log_freq_timesteps=int(args.wandb_train_log_freq_timesteps),
                    verbose=0,
                )
                if gail.gen_callback is None:
                    gail.gen_callback = train_metrics_cb
                else:
                    gail.gen_callback = CallbackList([gail.gen_callback, train_metrics_cb])
                gail_round_callback = GAILWandbRoundMetricsCallback(
                    wandb_run=wandb_run,
                    gen_algo=learner,
                    total_timesteps=int(effective_total_timesteps),
                )
                round_callbacks.append(gail_round_callback)

        if round_callbacks:
            gail_round_callback = CompositeGAILRoundCallback(round_callbacks)
        else:
            gail_round_callback = None

        t0 = time.time()
        gail.train(total_timesteps=int(effective_total_timesteps), callback=gail_round_callback)
        elapsed = time.time() - t0
        print(f"[train] done in {elapsed:.1f}s")

        policy_path = os.path.join(run_dir, "ppo_policy_gail_final.zip")
        reward_net_path = os.path.join(run_dir, "gail_reward_net.pt")
        learner.save(policy_path)
        torch.save(gail._reward_net.state_dict(), reward_net_path)

        config = {
            "config": {
                "demo_base": args.demo_base,
                "demo_source_format": args.demo_source_format,
                "save_root": args.save_root,
                "run_tag": args.run_tag,
                "seed": int(args.seed),
                "device": str(device),
                "bc_bootstrap_runs_root": bc_bootstrap_runs_root,
                "bc_bootstrap_enabled": bool(bc_bootstrap_info.enabled),
                "bc_bootstrap": bc_bootstrap_info.to_dict(),
                "bc_anchor_during_gail": bool(bc_anchor_during_gail),
                "bc_anchor_batch_size": int(bc_anchor_batch_size),
                "bc_anchor_updates_per_round": int(bc_anchor_updates_per_round),
                "bc_anchor_learning_rate": float(bc_anchor_learning_rate),
                "bc_anchor_ent_weight": float(bc_anchor_ent_weight),
                "bc_anchor_l2_weight": float(bc_anchor_l2_weight),
                "bc_anchor_decay_fraction": float(bc_anchor_decay_fraction),
                "save_intermediate_checkpoints": bool(save_intermediate_checkpoints),
                "checkpoint_interval_timesteps": int(checkpoint_interval_timesteps),
                "reward_net_type": reward_net_type,
                "action_limit": float(args.action_limit),
                "actuator_model": bool(args.actuator_model),
                "peg1_only_fixed_opponent": bool(peg1_only_fixed_opponent),
                "fixed_opponent_policy_path": fixed_opponent_policy_path,
                "fixed_opponent_policy_format": fixed_opponent_policy_format,
                "fixed_opponent_rl_games_config_path": fixed_opponent_rl_games_config_path,
                "fixed_opponent_deterministic": bool(fixed_opponent_deterministic),
                "invert_opponent_actions": bool(invert_opponent_actions),
                "requested_num_envs": int(args.num_envs),
                "num_envs": int(effective_num_envs),
                "requested_total_timesteps": int(requested_total_timesteps),
                "total_timesteps": int(effective_total_timesteps),
                "learning_rate": float(args.learning_rate),
                "batch_size": int(args.batch_size),
                "n_steps": int(args.n_steps),
                "gamma": float(args.gamma),
                "gae_lambda": float(args.gae_lambda),
                "ent_coef": float(args.ent_coef),
                "clip_range": float(args.clip_range),
                "vf_coef": float(args.vf_coef),
                "n_epochs": int(args.n_epochs),
                "n_disc_updates": int(initial_disc_updates),
                "adaptive_disc_updates": bool(adaptive_disc_updates),
                "disc_acc_low": float(disc_acc_low),
                "disc_acc_high": float(disc_acc_high),
                "min_disc_updates": int(min_disc_updates),
                "max_disc_updates": int(max_disc_updates),
                "balance_disc_expert_loss": bool(balance_disc_expert_loss),
                "disc_balance_gap_target": float(disc_balance_gap_target),
                "disc_expert_loss_gain": float(disc_expert_loss_gain),
                "disc_expert_loss_max_weight": float(disc_expert_loss_max_weight),
                "freeze_disc_on_high_gen_acc": bool(freeze_disc_on_high_gen_acc),
                "disc_gen_acc_freeze_threshold": float(disc_gen_acc_freeze_threshold),
                "disc_gen_exp_gap_freeze_threshold": float(disc_gen_exp_gap_freeze_threshold),
                "disc_freeze_rounds": int(disc_freeze_rounds),
                "strict_disc_freeze": bool(strict_disc_freeze),
                "freeze_disc_until_gen_acc_below_threshold": bool(
                    freeze_disc_until_gen_acc_below_threshold
                ),
                "disc_gen_acc_unfreeze_threshold": disc_gen_acc_unfreeze_threshold,
                "disc_weight_decay": float(args.disc_weight_decay),
                "disc_learning_rate": float(args.disc_learning_rate),
                "disc_label_smoothing": float(args.disc_label_smoothing),
                "disc_reward_hid_sizes": disc_reward_hid_sizes,
                "disc_potential_hid_sizes": disc_potential_hid_sizes,
                "event_aware_disc_weighting": bool(event_aware_disc_weighting),
                "event_expert_goal_weight": float(event_expert_goal_weight),
                "event_expert_bad_weight": float(event_expert_bad_weight),
                "event_expert_timeout_weight": float(event_expert_timeout_weight),
                "event_generated_goal_weight": float(event_generated_goal_weight),
                "event_generated_bad_weight": float(event_generated_bad_weight),
                "event_generated_timeout_weight": float(event_generated_timeout_weight),
                "event_preterminal_window_steps": int(event_preterminal_window_steps),
                "event_preterminal_decay": float(event_preterminal_decay),
                "augment": bool(augment_enabled),
                "min_len": int(args.min_len),
                "strip_sim_action_history": bool(args.strip_sim_action_history),
                "masked_velocities": bool(masked_velocities),
                "zero_mask_velocity_obs_human": bool(args.zero_mask_velocity_obs_human),
                "zero_mask_velocity_obs_sim": bool(args.zero_mask_velocity_obs_sim),
                "no_val_split": bool(args.no_val_split),
                "val_split_fraction": float(val_split_fraction),
                "val_split_level": str(val_split_level),
                "val_split_seed": int(val_split_seed),
            },
            "dataset_stats": dataset_stats.to_dict(),
            "dataset_split": dataset_split,
            "run_name": run_name,
            "run_dir": run_dir,
            "checkpoints_dir": (checkpoints_dir if save_intermediate_checkpoints else None),
            "created_at": datetime.datetime.now().isoformat(),
            "training_elapsed_sec": float(elapsed),
            "final_policy_path": policy_path,
            "final_reward_net_path": reward_net_path,
        }
        with open(os.path.join(run_dir, "config.json"), "w") as fh:
            json.dump(config, fh, indent=2)
        if bool(split_enabled):
            with open(os.path.join(run_dir, "dataset_split.json"), "w") as fh:
                json.dump(dataset_split, fh, indent=2)

        with open(os.path.join(run_dir, "train_stats.json"), "w") as fh:
            json.dump(
                {
                    "training_elapsed_sec": float(elapsed),
                    "dataset_steps_used": int(sum(len(tr.acts) for tr in trajectories)),
                    "num_trajectories_used": int(len(trajectories)),
                    "train_raw_steps": int(train_raw_steps),
                    "val_steps_raw": int(val_steps),
                    "train_raw_trajectories": int(len(train_raw_trajs)),
                    "val_raw_trajectories": int(len(val_trajs)),
                    "val_split_enabled": bool(split_enabled),
                    "val_split_fraction": float(val_split_fraction),
                    "val_split_level": str(val_split_level),
                    "val_split_seed": int(val_split_seed),
                    "bc_bootstrap_enabled": bool(bc_bootstrap_info.enabled),
                    "bc_bootstrap": bc_bootstrap_info.to_dict(),
                    "bc_anchor_during_gail": bool(bc_anchor_during_gail),
                    "bc_anchor_batch_size": int(bc_anchor_batch_size),
                    "bc_anchor_updates_per_round": int(bc_anchor_updates_per_round),
                    "bc_anchor_learning_rate": float(bc_anchor_learning_rate),
                    "bc_anchor_ent_weight": float(bc_anchor_ent_weight),
                    "bc_anchor_l2_weight": float(bc_anchor_l2_weight),
                    "bc_anchor_decay_fraction": float(bc_anchor_decay_fraction),
                    "save_intermediate_checkpoints": bool(save_intermediate_checkpoints),
                    "checkpoint_interval_timesteps": int(checkpoint_interval_timesteps),
                    "reward_net_type": reward_net_type,
                "action_limit": float(args.action_limit),
                "actuator_model": bool(args.actuator_model),
                    "requested_num_envs": int(args.num_envs),
                    "effective_num_envs": int(effective_num_envs),
                    "requested_total_timesteps": int(requested_total_timesteps),
                    "effective_total_timesteps": int(effective_total_timesteps),
                    "rollout_batch_size": int(rollout_batch),
                    "demo_batch_size": int(demo_batch_size),
                    "initial_n_disc_updates_per_round": int(initial_disc_updates),
                    "final_n_disc_updates_per_round": int(gail.n_disc_updates_per_round),
                    "checkpoints_dir": (checkpoints_dir if save_intermediate_checkpoints else None),
                    "disc_total_skipped_updates": int(
                        getattr(gail, "_disc_total_skipped_updates", 0)
                    ),
                    "strict_disc_freeze": bool(strict_disc_freeze),
                    "freeze_disc_until_gen_acc_below_threshold": bool(
                        freeze_disc_until_gen_acc_below_threshold
                    ),
                    "disc_gen_acc_unfreeze_threshold": disc_gen_acc_unfreeze_threshold,
                    "disc_strict_frozen_active_final": bool(
                        getattr(gail, "_disc_strict_frozen", False)
                    ),
                    "disc_until_recovered_frozen_active_final": bool(
                        getattr(gail, "_disc_until_recovered_frozen", False)
                    ),
                    "peg1_only_fixed_opponent": bool(peg1_only_fixed_opponent),
                    "fixed_opponent_policy_path": fixed_opponent_policy_path,
                    "fixed_opponent_policy_format": fixed_opponent_policy_format,
                    "fixed_opponent_rl_games_config_path": fixed_opponent_rl_games_config_path,
                    "fixed_opponent_deterministic": bool(fixed_opponent_deterministic),
                    "invert_opponent_actions": bool(invert_opponent_actions),
                    "event_aware_disc_weighting": bool(event_aware_disc_weighting),
                    "event_expert_goal_weight": float(event_expert_goal_weight),
                    "event_expert_bad_weight": float(event_expert_bad_weight),
                    "event_expert_timeout_weight": float(event_expert_timeout_weight),
                    "event_generated_goal_weight": float(event_generated_goal_weight),
                    "event_generated_bad_weight": float(event_generated_bad_weight),
                    "event_generated_timeout_weight": float(event_generated_timeout_weight),
                    "event_preterminal_window_steps": int(event_preterminal_window_steps),
                    "event_preterminal_decay": float(event_preterminal_decay),
                    "final_num_timesteps": int(learner.num_timesteps),
                },
                fh,
                indent=2,
            )

        if wandb_run is not None:
            try:
                wandb_run.log(
                    {
                        "train/elapsed_sec": float(elapsed),
                        "train/final_timesteps": int(learner.num_timesteps),
                    }
                )
            except Exception:
                pass
            wandb_run.finish()
            wandb_run = None

        print(f"[out] {policy_path}")
        print(f"[out] {reward_net_path}")
        print(f"[out] {os.path.join(run_dir, 'config.json')}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        venv.close()


if __name__ == "__main__":
    main()
