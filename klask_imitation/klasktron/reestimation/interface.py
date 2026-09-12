"""Contracts and adapter for integrating the external human reestimation pipeline.

The reestimation pipeline is maintained in a separate repository.  This module
defines the artifact contract and provides a validate-and-import function that
converts raw exports into the benchmark dataset tier format.

Expected input: a directory of .pkl files, each containing a list of
``imitation.data.types.TrajectoryWithRew`` (or plain ``Trajectory``) objects
with the following layout:

    obs   : float32 ndarray, shape (T+1, 12)
    acts  : float32 ndarray, shape (T, 4)
    infos : optional per-step metadata (may be None)
    terminal : bool

Observation columns (human_raw layout, before benchmark transform):

    [bp_x, bp_y, bv_x, bv_y, lp_x, lp_y, rp_x, rp_y, lv_x, lv_y, rv_x, rv_y]
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Mapping, Optional, Tuple, Union

import numpy as np

EXPECTED_OBS_DIM = 12
EXPECTED_ACT_DIM = 4


@dataclass(frozen=True)
class ReestimationArtifact:
    """Description of a cleaned trajectory export produced by the external pipeline."""

    path: Path
    source_id: str
    source_format: str
    step_count: int
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Summary returned by :func:`validate_and_import`."""

    valid: bool
    n_files: int
    n_trajectories: int
    n_steps: int
    errors: List[str]
    warnings: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_obs_acts(item):
    """Return (obs, acts, terminal) from a Trajectory, TrajectoryWithRew, or dict."""
    if hasattr(item, "obs"):
        obs = np.asarray(item.obs, dtype=np.float32)
        acts = np.asarray(item.acts, dtype=np.float32)
        terminal = bool(getattr(item, "terminal", True))
    elif isinstance(item, dict):
        obs = np.asarray(item["obs"], dtype=np.float32)
        acts = np.asarray(item["acts"], dtype=np.float32)
        terminal = bool(item.get("terminal", True))
    else:
        raise TypeError(f"Unsupported trajectory type: {type(item)}")
    return obs, acts, terminal


def validate_export_dir(export_dir: Union[str, Path]) -> ValidationResult:
    """Validate that a directory of pkl files conforms to the reestimation contract.

    Does NOT modify any files — read-only validation.
    """
    export_dir = Path(export_dir)
    errors: List[str] = []
    warnings: List[str] = []

    if not export_dir.is_dir():
        return ValidationResult(
            valid=False, n_files=0, n_trajectories=0, n_steps=0,
            errors=[f"Directory not found: {export_dir}"], warnings=[],
        )

    pkl_paths = sorted(export_dir.rglob("*.pkl"))
    if not pkl_paths:
        return ValidationResult(
            valid=False, n_files=0, n_trajectories=0, n_steps=0,
            errors=[f"No .pkl files found under {export_dir}"], warnings=[],
        )

    total_trajs = 0
    total_steps = 0

    for path in pkl_paths:
        try:
            with open(path, "rb") as fh:
                payload = pickle.load(fh)
        except Exception as exc:
            errors.append(f"{path.name}: failed to unpickle: {exc}")
            continue

        items = payload if isinstance(payload, (list, tuple)) else [payload]
        for idx, item in enumerate(items):
            try:
                obs, acts, terminal = _extract_obs_acts(item)
            except (TypeError, KeyError) as exc:
                errors.append(f"{path.name}[{idx}]: {exc}")
                continue

            if obs.ndim != 2:
                errors.append(f"{path.name}[{idx}]: obs.ndim={obs.ndim}, expected 2")
                continue
            if obs.shape[1] != EXPECTED_OBS_DIM:
                errors.append(
                    f"{path.name}[{idx}]: obs has {obs.shape[1]} columns, "
                    f"expected {EXPECTED_OBS_DIM}"
                )
            if acts.ndim != 2 or acts.shape[1] != EXPECTED_ACT_DIM:
                errors.append(
                    f"{path.name}[{idx}]: acts shape {acts.shape}, "
                    f"expected (T, {EXPECTED_ACT_DIM})"
                )
            if obs.shape[0] != acts.shape[0] + 1:
                errors.append(
                    f"{path.name}[{idx}]: obs length {obs.shape[0]} != "
                    f"acts length {acts.shape[0]} + 1"
                )
            if acts.shape[0] < 2:
                warnings.append(f"{path.name}[{idx}]: very short trajectory ({acts.shape[0]} steps)")

            total_trajs += 1
            total_steps += acts.shape[0]

    return ValidationResult(
        valid=len(errors) == 0,
        n_files=len(pkl_paths),
        n_trajectories=total_trajs,
        n_steps=total_steps,
        errors=errors,
        warnings=warnings,
    )


def validate_and_import(
    export_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    tier_name: str = "human",
    source_format: str = "human_raw",
    chunk_size: int = 64,
    min_len: int = 2,
) -> Tuple[ValidationResult, Optional[Path]]:
    """Validate reestimation exports and write them into the benchmark tier layout.

    Parameters
    ----------
    export_dir:
        Directory containing .pkl files from the reestimation pipeline.
    output_dir:
        Destination directory.  A ``{tier_name}/trajectories/`` subtree and
        ``metadata.json`` will be created here.
    tier_name:
        Dataset tier label (default ``"human"``).
    source_format:
        One of ``"human_raw"`` or ``"sim_native"``.
    chunk_size:
        Number of trajectories per output chunk file.
    min_len:
        Minimum trajectory length (action steps) to keep.

    Returns
    -------
    (result, tier_dir) where ``tier_dir`` is the written tier path on success,
    or None on validation failure.
    """
    from imitation.data.types import Trajectory

    result = validate_export_dir(export_dir)
    if not result.valid:
        return result, None

    # Re-load and filter trajectories.
    export_dir = Path(export_dir)
    output_dir = Path(output_dir)
    pkl_paths = sorted(export_dir.rglob("*.pkl"))

    trajectories: List[Trajectory] = []
    for path in pkl_paths:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        items = payload if isinstance(payload, (list, tuple)) else [payload]
        for item in items:
            obs, acts, terminal = _extract_obs_acts(item)
            if acts.shape[0] < min_len:
                continue
            infos = getattr(item, "infos", None)
            if isinstance(item, dict):
                infos = item.get("infos")
            trajectories.append(
                Trajectory(obs=obs, acts=acts, infos=infos, terminal=terminal)
            )

    # Write chunked pkl files.
    tier_dir = output_dir / tier_name
    traj_dir = tier_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    written_files: List[str] = []
    for start in range(0, len(trajectories), chunk_size):
        chunk = trajectories[start : start + chunk_size]
        fname = f"chunk_{start // chunk_size:05d}.pkl"
        with open(traj_dir / fname, "wb") as fh:
            pickle.dump(chunk, fh, protocol=pickle.HIGHEST_PROTOCOL)
        written_files.append(fname)
        for tr in chunk:
            total_steps += len(tr.acts)

    metadata = {
        "tier": tier_name,
        "source_format": source_format,
        "episode_count": len(trajectories),
        "step_count": total_steps,
        "frame_rate_hz": 50,
        "chunk_size": chunk_size,
        "files": written_files,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(tier_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    result.n_trajectories = len(trajectories)
    result.n_steps = total_steps

    return result, tier_dir
