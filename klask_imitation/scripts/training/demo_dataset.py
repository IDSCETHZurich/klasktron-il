"""Shared dataset loading utilities for BC/GAIL training.

Supports two input formats:
- human_raw: real-world demo coordinates; convert to simulation coordinates.
- sim_native: already in simulation coordinates with explicit actions.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import gymnasium as gym

from imitation.data.types import Trajectory, TrajectoryWithRew, maybe_wrap_in_dictobs


# Coordinate transformation constants.
LONG_FACTOR = 0.0008285
SHORT_FACTOR = 1.0 / 1150.0
BOARD_LONG, BOARD_SHORT = 0.44, 0.32  # (long, short)

DEMO_SOURCE_HUMAN_RAW = "human_raw"
DEMO_SOURCE_SIM_NATIVE = "sim_native"
VALID_DEMO_SOURCES = (DEMO_SOURCE_HUMAN_RAW, DEMO_SOURCE_SIM_NATIVE)


@dataclass
class DatasetLoadStats:
    source_format: str
    n_files: int
    n_traj_raw: int
    n_steps_raw: int
    n_traj_after_aug: int
    n_steps_after_aug: int

    def to_dict(self) -> Dict[str, Union[str, int]]:
        return asdict(self)


def transform_observations(
    obs: np.ndarray,
    dt: float = 1 / 50,
    smooth_window: int = 15,
) -> np.ndarray:
    """Transform real-world observations to simulation coordinates/layout."""
    o = np.asarray(obs, dtype=np.float32)
    if o.ndim == 1:
        o = o[None, :]
    if o.ndim != 2 or o.shape[1] != 12:
        return o

    # Incoming layout: [bp_x, bp_y, ?, ?, lp_x, lp_y, rp_x, rp_y, ...]
    bp = o[:, 0:2]
    lp = o[:, 4:6]
    rp = o[:, 6:8]

    def map_pos(pxy: np.ndarray) -> np.ndarray:
        # demo x->sim y, demo y->sim x, then center at board center.
        x = pxy[:, 1] * SHORT_FACTOR - BOARD_SHORT / 2
        y = pxy[:, 0] * LONG_FACTOR - BOARD_LONG / 2
        return np.stack([x, y], axis=1)

    bp_s = map_pos(bp)
    lp_s = map_pos(lp)
    rp_s = map_pos(rp)

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

    def vel_from_pos(p: np.ndarray) -> np.ndarray:
        v = np.zeros_like(p)
        if p.shape[0] > 1:
            v[:-1] = (p[1:] - p[:-1]) / dt
            v[-1] = v[-2]
        return v

    bv_s = vel_from_pos(bp_s)
    lv_s = vel_from_pos(lp_s)
    rv_s = vel_from_pos(rp_s)

    # Output layout:
    # [bp_x, bp_y, bv_x, bv_y, lp_x, lp_y, rp_x, rp_y, lv_x, lv_y, rv_x, rv_y]
    return np.concatenate([bp_s, bv_s, lp_s, rp_s, lv_s, rv_s], axis=1)


def actions_from_transformed_obs(obs: np.ndarray) -> np.ndarray:
    """Extract actions [lv_x, lv_y, rv_x, rv_y] from transformed observations."""
    o = np.asarray(obs, dtype=np.float32)
    if o.ndim != 2 or o.shape[1] != 12:
        return np.zeros((0, 4), dtype=np.float32)
    return o[:-1, [8, 9, 10, 11]]


def _ensure_tplus1(tr: Trajectory) -> None:
    t_obs = len(tr.obs)
    t_act = len(tr.acts)
    if t_obs != t_act + 1:
        raise ValueError(
            f"Trajectory obs length {t_obs} != acts length {t_act} + 1"
        )


def collect_pkl_paths(base_dir: str) -> List[str]:
    base = Path(base_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    paths = sorted(str(p) for p in base.rglob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No .pkl files found under: {base_dir}")
    return paths


def _extract_obs_acts(
    item: Union[TrajectoryWithRew, Trajectory, dict],
) -> Tuple[Union[np.ndarray, dict], Optional[np.ndarray], Optional[object], bool]:
    if isinstance(item, (TrajectoryWithRew, Trajectory)):
        obs = item.obs
        acts = item.acts
        infos = item.infos
        terminal = bool(getattr(item, "terminal", True))
        return obs, acts, infos, terminal
    if isinstance(item, dict):
        return (
            item["obs"],
            item.get("acts"),
            item.get("infos"),
            bool(item.get("terminal", True)),
        )
    raise TypeError(f"Unsupported trajectory payload type: {type(item)}")


def load_trajectories(
    paths: List[str],
    action_space: gym.Space,
    *,
    source_format: str = DEMO_SOURCE_HUMAN_RAW,
    min_len: int = 2,
) -> Tuple[List[Trajectory], List[str], int]:
    """Load trajectories from pickle files.

    Returns:
      - trajectories
      - group_ids (source file path for each trajectory)
      - total action steps before augmentation
    """
    if source_format not in VALID_DEMO_SOURCES:
        raise ValueError(
            f"Invalid source_format={source_format!r}. Valid: {VALID_DEMO_SOURCES}"
        )

    trajectories: List[Trajectory] = []
    group_ids: List[str] = []

    for path in paths:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        items: Iterable[Union[TrajectoryWithRew, Trajectory, dict]]
        if isinstance(payload, (list, tuple)):
            items = payload
        else:
            items = [payload]

        for item_idx, item in enumerate(items):
            raw_obs, raw_acts, infos, terminal = _extract_obs_acts(item)
            obs_obj = (
                maybe_wrap_in_dictobs(raw_obs)
                if isinstance(raw_obs, dict)
                else np.asarray(raw_obs, dtype=np.float32)
            )

            if isinstance(obs_obj, dict):
                if raw_acts is None:
                    raise ValueError("Dict observation payload requires explicit `acts`.")
                acts_obj = np.asarray(raw_acts, dtype=np.float32)
            else:
                if source_format == DEMO_SOURCE_HUMAN_RAW:
                    obs_obj = transform_observations(obs_obj)
                    if obs_obj.ndim != 2 or obs_obj.shape[1] != 12:
                        raise ValueError(
                            f"human_raw loader expects obs shape [T,12], got {obs_obj.shape} "
                            f"in {path} (item {item_idx}). "
                            "If this is synthetic data, use --demo-source-format sim_native."
                        )
                    acts_obj = actions_from_transformed_obs(obs_obj)
                else:
                    if raw_acts is None:
                        raise ValueError(
                            "sim_native payload requires explicit actions in each item."
                        )
                    acts_obj = np.asarray(raw_acts, dtype=np.float32)
                    if isinstance(action_space, gym.spaces.Box):
                        if acts_obj.shape[-1] != action_space.shape[-1]:
                            raise ValueError(
                                f"Action dimension mismatch for {path}: "
                                f"{acts_obj.shape[-1]} vs expected {action_space.shape[-1]}"
                            )

            if not isinstance(obs_obj, dict):
                if obs_obj.shape[0] != acts_obj.shape[0] + 1:
                    raise ValueError(
                        f"Malformed trajectory in {path} (item {item_idx}): "
                        f"obs_len={obs_obj.shape[0]} acts_len={acts_obj.shape[0]}. "
                        "Expected obs_len == acts_len + 1."
                    )

            tr = Trajectory(
                obs=obs_obj,
                acts=acts_obj,
                infos=infos,
                terminal=terminal,
            )
            if len(tr.acts) < min_len:
                continue
            _ensure_tplus1(tr)
            trajectories.append(tr)
            group_ids.append(path)

    total_steps = int(sum(len(tr.acts) for tr in trajectories))
    return trajectories, group_ids, total_steps


# layout after transform_observations (sim-native layout expected as well):
# [ball_x, ball_y, ball_vx, ball_vy, peg1_x, peg1_y, peg2_x, peg2_y, peg1_vx, peg1_vy, peg2_vx, peg2_vy]
_BP_X, _BP_Y = 0, 1
_BV_X, _BV_Y = 2, 3
_LP_X, _LP_Y = 4, 5
_RP_X, _RP_Y = 6, 7
_LV_X, _LV_Y = 8, 9
_RV_X, _RV_Y = 10, 11


def mirror_trajectory(
    tr: Trajectory,
    *,
    mirror_x_axis: bool = False,
    mirror_y_axis: bool = False,
) -> Trajectory:
    """Mirror trajectory in simulation coordinates."""
    obs = np.asarray(tr.obs).copy()
    acts = np.asarray(tr.acts).copy()

    if mirror_y_axis:
        obs[:, _BP_X] *= -1.0
        obs[:, _BV_X] *= -1.0

        lp_x, lp_y = obs[:, _LP_X].copy(), obs[:, _LP_Y].copy()
        rp_x, rp_y = obs[:, _RP_X].copy(), obs[:, _RP_Y].copy()
        obs[:, _LP_X], obs[:, _LP_Y] = -rp_x, rp_y
        obs[:, _RP_X], obs[:, _RP_Y] = -lp_x, lp_y

        lv_x, lv_y = obs[:, _LV_X].copy(), obs[:, _LV_Y].copy()
        rv_x, rv_y = obs[:, _RV_X].copy(), obs[:, _RV_Y].copy()
        obs[:, _LV_X], obs[:, _LV_Y] = -rv_x, rv_y
        obs[:, _RV_X], obs[:, _RV_Y] = -lv_x, lv_y

        lvx, lvy = acts[:, 0].copy(), acts[:, 1].copy()
        rvx, rvy = acts[:, 2].copy(), acts[:, 3].copy()
        acts[:, 0], acts[:, 1] = -rvx, rvy
        acts[:, 2], acts[:, 3] = -lvx, lvy

    if mirror_x_axis:
        obs[:, _BP_Y] *= -1.0
        obs[:, _BV_Y] *= -1.0
        obs[:, _LP_Y] *= -1.0
        obs[:, _RP_Y] *= -1.0
        obs[:, _LV_Y] *= -1.0
        obs[:, _RV_Y] *= -1.0

        acts[:, 1] *= -1.0
        acts[:, 3] *= -1.0

    return Trajectory(obs=obs, acts=acts, infos=tr.infos, terminal=tr.terminal)


def augment_with_axis_mirrors(
    trajectories: Sequence[Trajectory],
) -> Tuple[List[Trajectory], int]:
    """Return original trajectories plus x/y/xy mirrored copies."""
    aug: List[Trajectory] = []
    for tr in trajectories:
        aug.append(mirror_trajectory(tr, mirror_x_axis=True, mirror_y_axis=False))
        aug.append(mirror_trajectory(tr, mirror_x_axis=False, mirror_y_axis=True))
        aug.append(mirror_trajectory(tr, mirror_x_axis=True, mirror_y_axis=True))
    return list(trajectories) + aug, len(aug)


def load_dataset_with_optional_augmentation(
    *,
    demo_base: str,
    action_space: gym.Space,
    source_format: str,
    min_len: int = 2,
    augment_mirrors: bool = True,
) -> Tuple[List[Trajectory], List[str], DatasetLoadStats]:
    """Load trajectories and optionally augment them with axis mirrors."""
    pkl_paths = collect_pkl_paths(demo_base)
    raw_trajs, group_ids, raw_steps = load_trajectories(
        pkl_paths,
        action_space,
        source_format=source_format,
        min_len=min_len,
    )
    if augment_mirrors:
        trajectories, _ = augment_with_axis_mirrors(raw_trajs)
    else:
        trajectories = list(raw_trajs)

    stats = DatasetLoadStats(
        source_format=source_format,
        n_files=len(pkl_paths),
        n_traj_raw=len(raw_trajs),
        n_steps_raw=int(raw_steps),
        n_traj_after_aug=len(trajectories),
        n_steps_after_aug=int(sum(len(tr.acts) for tr in trajectories)),
    )
    return trajectories, group_ids, stats
