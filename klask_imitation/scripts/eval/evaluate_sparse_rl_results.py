"""
Evaluate sparse-RL Peg1-only (2D action) policy results using the same pipeline as
evaluate_gail_results.py, with sparse-RL-specific defaults.

Defaults evaluate Peg1-only policy against a fixed BC/bootstrap opponent,
matching the training-time opponent action construction.
"""

import sys
import random
import json
import os
from pathlib import Path

seed = random.randint(0, 1000000)
print(f"Using random seed {seed} for evaluation")

# Importing this will also initialize Isaac Lab (AppLauncher) as in the GAIL evaluator.
from evaluate_gail_results import main as evaluate_main

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAINED_MODELS_DIR = Path(
    os.environ.get("KLASKTRON_TRAINED_MODELS_DIR", _REPO_ROOT / "trained_models")
)

SPARSE_RL_RUN_DIR = (
    os.environ.get(
        "KLASKTRON_SPARSE_RL_RUN_DIR",
        str(_TRAINED_MODELS_DIR / "klask_sparse_rl_20260223_111724_GAIL_BC_sparse_RL"),
    )
)
SPARSE_RL_POLICY_PATH = (
    os.environ.get(
        "KLASKTRON_SPARSE_RL_POLICY_PATH",
        str(Path(SPARSE_RL_RUN_DIR) / "ppo_policy_sparse_rl_best_peg1_ratio.zip"),
    )
)


def _default_opponent_policy_path(run_dir: str) -> str | None:
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r") as f:
            payload = json.load(f)
    except Exception:
        return None
    path = payload.get("config", {}).get("pretrained_policy_path", None)
    if isinstance(path, str) and path:
        return path
    return None


def _ensure_arg(flag: str, value: str | None = None) -> None:
    if flag in sys.argv:
        return
    if value is None:
        sys.argv.append(flag)
    else:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    opponent_policy_path = _default_opponent_policy_path(SPARSE_RL_RUN_DIR)
    _ensure_arg("--run-dir", SPARSE_RL_RUN_DIR)
    _ensure_arg("--policy-path", SPARSE_RL_POLICY_PATH)
    if opponent_policy_path is not None:
        _ensure_arg("--opponent-policy-path", opponent_policy_path)
    _ensure_arg("--no-reward-net")
    _ensure_arg("--peg1-only-fixed-opponent")
    _ensure_arg("--seed", str(seed))
    evaluate_main()
