"""
Evaluate BC policy results using the same pipeline as evaluate_gail_results.py,
with BC-specific defaults.
"""

import os
import random
import sys
from pathlib import Path

# Importing this will also initialize Isaac Lab (AppLauncher) as in the GAIL evaluator.
from evaluate_gail_results import main as evaluate_main

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAINED_MODELS_DIR = Path(
    os.environ.get("KLASKTRON_TRAINED_MODELS_DIR", _REPO_ROOT / "trained_models")
)
_DEFAULT_BC_RUN_DIR = (
    _TRAINED_MODELS_DIR
    / "data_wall_experiments"
    / "bc_runs"
    / "klask_bc_20260303_161251_HUMAN_DEMO_V2"
)

BC_RUN_DIR = os.environ.get("KLASKTRON_BC_RUN_DIR", str(_DEFAULT_BC_RUN_DIR))
BC_POLICY_PATH = os.environ.get(
    "KLASKTRON_BC_POLICY_PATH",
    str(Path(BC_RUN_DIR) / "ppo_policy_bc_final.zip"),
)
seed = random.SystemRandom().randint(0, 2**31 - 1)
print(f"Using random seed {seed} for evaluation")


def _ensure_arg(flag: str, value: str | None = None) -> None:
    if flag in sys.argv:
        return
    if value is None:
        sys.argv.append(flag)
    else:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _ensure_arg("--run-dir", BC_RUN_DIR)
    _ensure_arg("--policy-path", BC_POLICY_PATH)
    _ensure_arg("--opponent-policy-path", BC_POLICY_PATH)
    _ensure_arg("--no-reward-net")
    _ensure_arg("--peg1-only-fixed-opponent")
    _ensure_arg("--deterministic-opponent")
    _ensure_arg("--seed", str(seed))
    import os as _os
    import sys as _sys
    import traceback as _tb

    try:
        evaluate_main()
    except BaseException:
        _tb.print_exc()
        _sys.stdout.flush()
        _sys.stderr.flush()
        _os._exit(1)
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(0)
