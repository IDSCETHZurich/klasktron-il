"""Prepare human and augmented dataset tiers for Hugging Face upload.

Reads the raw human demonstrations from data/expert_trajectories_cleaned,
transforms them to simulation coordinates, applies mirror augmentation,
and writes both tiers into the layout expected by the HF downloader:

    data/human/trajectories/*.pkl        + metadata.json
    data/augmented/trajectories/*.pkl    + metadata.json

Usage:
    python klask_imitation/scripts/prepare_dataset_tiers.py
    python klask_imitation/scripts/prepare_dataset_tiers.py --input-dir data/expert_trajectories_cleaned
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Ensure the package is importable when running from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "klask_imitation"))

from imitation.data.types import Trajectory

from scripts.training.demo_dataset import (
    augment_with_axis_mirrors,
    collect_pkl_paths,
    load_trajectories,
)

import gymnasium as gym


def _dummy_action_space() -> gym.spaces.Box:
    """4-dim continuous action space matching Klask peg velocities."""
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)


def _strip_rewards(tr) -> Trajectory:
    """Convert TrajectoryWithRew to plain Trajectory (rewards are placeholder zeros)."""
    return Trajectory(
        obs=np.asarray(tr.obs, dtype=np.float32),
        acts=np.asarray(tr.acts, dtype=np.float32),
        infos=tr.infos,
        terminal=tr.terminal,
    )


def _write_tier(
    trajectories: list[Trajectory],
    output_dir: Path,
    tier_name: str,
    source_format: str,
    chunk_size: int = 64,
) -> dict:
    """Write trajectories as chunked pkl files + metadata.json."""
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    total_trajs = 0
    written_files: list[str] = []

    for start in range(0, len(trajectories), chunk_size):
        chunk = trajectories[start : start + chunk_size]
        fname = f"chunk_{start // chunk_size:05d}.pkl"
        path = traj_dir / fname
        with open(path, "wb") as fh:
            pickle.dump(chunk, fh, protocol=pickle.HIGHEST_PROTOCOL)
        written_files.append(fname)
        for tr in chunk:
            total_trajs += 1
            total_steps += len(tr.acts)

    metadata = {
        "tier": tier_name,
        "source_format": source_format,
        "episode_count": total_trajs,
        "step_count": total_steps,
        "frame_rate_hz": 50,
        "chunk_size": chunk_size,
        "files": written_files,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare human + augmented dataset tiers.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(_REPO_ROOT / "data" / "expert_trajectories_cleaned"),
        help="Directory with raw human .pkl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_REPO_ROOT / "data"),
        help="Root output directory (tiers created as subdirs).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Trajectories per .pkl chunk file.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    action_space = _dummy_action_space()

    # -- Load & transform human data ------------------------------------------
    print(f"Loading human demos from: {input_dir}")
    pkl_paths = collect_pkl_paths(str(input_dir))
    human_trajs, _, human_steps = load_trajectories(
        pkl_paths,
        action_space,
        source_format="human_raw",
        min_len=2,
    )
    human_trajs = [_strip_rewards(tr) for tr in human_trajs]
    print(f"  {len(human_trajs)} trajectories, {human_steps} steps")

    # -- Write human tier ------------------------------------------------------
    human_dir = output_dir / "human"
    human_meta = _write_tier(
        human_trajs, human_dir, "human", "sim_native", args.chunk_size,
    )
    print(f"Human tier written to {human_dir}")
    print(f"  {human_meta['episode_count']} trajectories, {human_meta['step_count']} steps")

    # -- Augment & write augmented tier ----------------------------------------
    augmented_trajs, n_added = augment_with_axis_mirrors(human_trajs)
    aug_dir = output_dir / "augmented"
    aug_meta = _write_tier(
        augmented_trajs, aug_dir, "augmented", "sim_native", args.chunk_size,
    )
    print(f"Augmented tier written to {aug_dir}")
    print(f"  {aug_meta['episode_count']} trajectories, {aug_meta['step_count']} steps")
    print(f"  ({n_added} mirror copies added to {len(human_trajs)} originals)")

    print("\nDone. Tiers ready for Hugging Face upload.")


if __name__ == "__main__":
    main()
