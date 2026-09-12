#!/usr/bin/env bash
set -euo pipefail

# --- Environment -------------------------------------------------------------
# All machine-specific locations are environment variables with defaults.
# Override any of them before invoking this script.
KLASKTRON_ROOT="${KLASKTRON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$HOME/IsaacLab}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
ROS_SETUP="${ROS_SETUP:-$HOME/ros2_ws/install/setup.bash}"
PYTHON_BIN="${PYTHON_BIN:-python}"
# -----------------------------------------------------------------------------

# Train BC-bootstrapped, event-aware GAIL policies across all official synthetic scales.
# The learned policy controls Peg1 only; Peg2 is frozen to PPO expert agent.
# This launcher resolves a matching BC checkpoint per dataset scale and passes it explicitly
# into train_gail_sweep.py for reproducible generator initialization.
#
# Usage:
#   bash run_gail_bc_bootstrap_all_synthetic.sh [WANDB_PROJECT] [DEVICE]
#
# Example:
#   bash run_gail_bc_bootstrap_all_synthetic.sh klask-gail-bc-bootstrap-v1 cuda:0

WANDB_PROJECT="${1:-klask-gail-bc-bootstrap-v1}"
DEVICE="${2:-cuda:0}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export COLCON_TRACE="${COLCON_TRACE:-0}"
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
set +u
source "${ROS_SETUP}"
set -u
cd "${ISAACLAB_ROOT}"

TRAIN_SCRIPT="${KLASKTRON_ROOT}/klask_imitation/scripts/training/train_gail_sweep.py"
SAVE_ROOT="${KLASKTRON_ROOT}/trained_models/data_wall_experiments/gail_runs"
DATASET_ROOT="${KLASKTRON_ROOT}/artifacts/data_wall/datasets"
BC_RUNS_ROOT="${KLASKTRON_ROOT}/trained_models/data_wall_experiments/bc_runs"
LAUNCH_MANIFEST_ROOT="${SAVE_ROOT}/launch_manifests"
PPO_EXPERT_DIR="${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one"
PPO_EXPERT_CHECKPOINT="${PPO_EXPERT_DIR}/klask.pth"
PPO_EXPERT_CONFIG="${PPO_EXPERT_DIR}/agent.yaml"

SCALES=("80000" "250000" "1000000" "5000000")
TAGS=("80k" "250k" "1Mio" "5Mio")
BC_RUN_TAGS=("DW_EXPERT_80000" "DW_EXPERT_250000" "DW_EXPERT_1000000" "DW_EXPERT_5000000")

mkdir -p "${LAUNCH_MANIFEST_ROOT}"

if [[ ! -f "${PPO_EXPERT_CHECKPOINT}" ]]; then
  echo "[error] Missing PPO expert checkpoint: ${PPO_EXPERT_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${PPO_EXPERT_CONFIG}" ]]; then
  echo "[error] Missing PPO expert config: ${PPO_EXPERT_CONFIG}" >&2
  exit 1
fi

for i in "${!SCALES[@]}"; do
  SCALE="${SCALES[$i]}"
  TAG="${TAGS[$i]}"
  BC_RUN_TAG="${BC_RUN_TAGS[$i]}"
  DATASET_DIR="${DATASET_ROOT}/ppo_expert_synthetic_${SCALE}"
  DEMO_BASE="${DATASET_DIR}/trajectories"
  MANIFEST_PATH="${DATASET_DIR}/manifest.json"
  RUN_TAG="GAIL_DW_EXPERT_${TAG}_BC_BOOTSTRAP_EVENT_AWARE_RECOVERY_FREEZE_VS_PPO_EXPERT"
  WANDB_RUN_NAME="GAIL_${TAG}_bc_bootstrap_event_aware_recovery_freeze_vs_ppo_expert"
  LAUNCH_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  LAUNCH_MANIFEST_PATH="${LAUNCH_MANIFEST_ROOT}/${LAUNCH_TIMESTAMP}_${RUN_TAG}.json"

  if [[ ! -d "${DEMO_BASE}" ]]; then
    echo "[error] Missing dataset directory: ${DEMO_BASE}" >&2
    exit 1
  fi
  if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo "[error] Missing dataset manifest: ${MANIFEST_PATH}" >&2
    exit 1
  fi
  if [[ ! -d "${BC_RUNS_ROOT}" ]]; then
    echo "[error] Missing BC runs root: ${BC_RUNS_ROOT}" >&2
    exit 1
  fi

  readarray -t DATASET_META < <(
    "${PYTHON_BIN}" - "${MANIFEST_PATH}" <<'PY'
import json
import os
import sys

manifest_path = sys.argv[1]
with open(manifest_path, "r", encoding="utf-8") as fh:
    manifest = json.load(fh)

print(os.path.abspath(str(manifest.get("dataset_dir", ""))))
print(os.path.abspath(str(manifest.get("trajectory_dir", ""))))
print(int(manifest.get("actual_steps", -1)))
print(int(manifest.get("target_steps", -1)))
print(int(manifest.get("num_files", -1)))
print(int(manifest.get("num_trajectories", -1)))
PY
  )

  DATASET_DIR_MANIFEST="${DATASET_META[0]}"
  TRAJECTORY_DIR_MANIFEST="${DATASET_META[1]}"
  ACTUAL_STEPS="${DATASET_META[2]}"
  TARGET_STEPS="${DATASET_META[3]}"
  NUM_FILES="${DATASET_META[4]}"
  NUM_TRAJECTORIES="${DATASET_META[5]}"

  if [[ "${DATASET_DIR_MANIFEST}" != "${DATASET_DIR}" ]]; then
    echo "[error] Dataset manifest path mismatch for ${RUN_TAG}: expected ${DATASET_DIR}, got ${DATASET_DIR_MANIFEST}" >&2
    exit 1
  fi
  if [[ "${TRAJECTORY_DIR_MANIFEST}" != "${DEMO_BASE}" ]]; then
    echo "[error] Trajectory path mismatch for ${RUN_TAG}: expected ${DEMO_BASE}, got ${TRAJECTORY_DIR_MANIFEST}" >&2
    exit 1
  fi
  if [[ "${ACTUAL_STEPS}" != "${SCALE}" ]]; then
    echo "[error] Dataset step mismatch for ${RUN_TAG}: expected scale ${SCALE}, manifest actual_steps=${ACTUAL_STEPS}" >&2
    exit 1
  fi

  readarray -t BC_META < <(
    "${PYTHON_BIN}" - "${BC_RUNS_ROOT}" "${BC_RUN_TAG}" "${DEMO_BASE}" <<'PY'
import json
import os
import sys

bc_runs_root = os.path.abspath(sys.argv[1])
expected_run_tag = str(sys.argv[2])
expected_demo_base = os.path.abspath(sys.argv[3])

candidates = []
for entry in os.scandir(bc_runs_root):
    if not entry.is_dir():
        continue
    config_path = os.path.join(entry.path, "config.json")
    if not os.path.isfile(config_path):
        continue
    with open(config_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    cfg = dict(payload.get("config", {}))
    if str(cfg.get("run_tag")) != expected_run_tag:
        continue
    demo_base = os.path.abspath(str(cfg.get("demo_base", "")))
    if demo_base != expected_demo_base:
        continue
    if bool(cfg.get("zero_mask_velocity_obs_sim", False)):
        continue
    if not bool(cfg.get("strip_sim_action_history", False)):
        continue
    policy_path = os.path.abspath(str(payload.get("final_policy_path") or os.path.join(entry.path, "ppo_policy_bc_final.zip")))
    if not os.path.isfile(policy_path):
        continue
    sort_key = (str(payload.get("created_at", "")), os.path.basename(entry.path))
    candidates.append((sort_key, entry.path, policy_path, str(cfg.get("run_tag")), str(payload.get("created_at", ""))))

if not candidates:
    raise SystemExit(
        f"No matching BC run found for run_tag={expected_run_tag} demo_base={expected_demo_base}"
    )

candidates.sort(key=lambda item: item[0])
_, run_dir, policy_path, run_tag, created_at = candidates[-1]
print(os.path.abspath(run_dir))
print(os.path.abspath(policy_path))
print(run_tag)
print(created_at)
PY
  )

  BC_RUN_DIR="${BC_META[0]}"
  BC_POLICY_PATH="${BC_META[1]}"
  BC_RUN_TAG_RESOLVED="${BC_META[2]}"
  BC_CREATED_AT="${BC_META[3]}"

  if [[ ! -f "${BC_POLICY_PATH}" ]]; then
    echo "[error] Resolved BC policy checkpoint is missing: ${BC_POLICY_PATH}" >&2
    exit 1
  fi

  echo "[run] scale=${SCALE} tag=${TAG} run_tag=${RUN_TAG}"
  echo "[run] dataset_dir=${DATASET_DIR}"
  echo "[run] demo_base=${DEMO_BASE}"
  echo "[run] manifest=${MANIFEST_PATH}"
  echo "[run] dataset_actual_steps=${ACTUAL_STEPS} target_steps=${TARGET_STEPS} num_files=${NUM_FILES} num_trajectories=${NUM_TRAJECTORIES}"
  echo "[run] bc_run_tag=${BC_RUN_TAG_RESOLVED}"
  echo "[run] bc_run_dir=${BC_RUN_DIR}"
  echo "[run] bc_policy_path=${BC_POLICY_PATH}"
  echo "[run] fixed_opponent_checkpoint=${PPO_EXPERT_CHECKPOINT}"
  echo "[run] fixed_opponent_config=${PPO_EXPERT_CONFIG}"

  "${PYTHON_BIN}" - "${LAUNCH_MANIFEST_PATH}" <<PY
import json
import os
import sys

out_path = sys.argv[1]
payload = {
    "created_at": "${LAUNCH_TIMESTAMP}",
    "run_tag": "${RUN_TAG}",
    "wandb_project": "${WANDB_PROJECT}",
    "wandb_run_name": "${WANDB_RUN_NAME}",
    "device": "${DEVICE}",
    "scale": int("${SCALE}"),
    "tag": "${TAG}",
    "dataset_dir": "${DATASET_DIR}",
    "demo_base": "${DEMO_BASE}",
    "manifest_path": "${MANIFEST_PATH}",
    "dataset_actual_steps": int("${ACTUAL_STEPS}"),
    "dataset_target_steps": int("${TARGET_STEPS}"),
    "dataset_num_files": int("${NUM_FILES}"),
    "dataset_num_trajectories": int("${NUM_TRAJECTORIES}"),
    "bc_runs_root": "${BC_RUNS_ROOT}",
    "bc_run_dir": "${BC_RUN_DIR}",
    "bc_run_tag": "${BC_RUN_TAG_RESOLVED}",
    "bc_created_at": "${BC_CREATED_AT}",
    "bc_policy_path": "${BC_POLICY_PATH}",
    "fixed_opponent_policy_path": "${PPO_EXPERT_CHECKPOINT}",
    "fixed_opponent_policy_format": "rl_games",
    "fixed_opponent_rl_games_config_path": "${PPO_EXPERT_CONFIG}",
    "train_script": "${TRAIN_SCRIPT}",
    "save_root": "${SAVE_ROOT}",
    "command": [
        "${PYTHON_BIN}",
        "${TRAIN_SCRIPT}",
        "--demo-base", "${DEMO_BASE}",
        "--demo-source-format", "sim_native",
        "--save-root", "${SAVE_ROOT}",
        "--run-tag", "${RUN_TAG}",
        "--init-generator-from-bc-path", "${BC_POLICY_PATH}",
        "--init-generator-from-bc-mode", "actor_only",
        "--peg1-only-fixed-opponent",
        "--fixed-opponent-policy-path", "${PPO_EXPERT_CHECKPOINT}",
        "--fixed-opponent-policy-format", "rl_games",
        "--fixed-opponent-rl-games-config-path", "${PPO_EXPERT_CONFIG}",
        "--total-timesteps", "5000000",
        "--device", "${DEVICE}",
        "--num-envs", "256",
        "--n-steps", "128",
        "--val-split-fraction", "0.1",
        "--val-split-level", "file",
        "--val-split-seed", "42",
        "--no-adaptive-disc-updates",
        "--n-disc-updates", "1",
        "--no-balance-disc-expert-loss",
        "--freeze-disc-on-high-gen-acc",
        "--disc-gen-acc-freeze-threshold", "0.75",
        "--freeze-disc-until-gen-acc-below-threshold",
        "--disc-gen-acc-unfreeze-threshold", "0.60",
        "--disc-gen-exp-gap-freeze-threshold", "0.0",
        "--disc-freeze-rounds", "1",
        "--event-aware-disc-weighting",
        "--event-expert-goal-weight", "1.8",
        "--event-expert-bad-weight", "0.6",
        "--event-expert-timeout-weight", "0.8",
        "--event-generated-goal-weight", "0.7",
        "--event-generated-bad-weight", "1.8",
        "--event-generated-timeout-weight", "1.2",
        "--wandb",
        "--wandb-project", "${WANDB_PROJECT}",
        "--wandb-mode", "online",
        "--wandb-run-name", "${WANDB_RUN_NAME}",
    ],
}
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
PY
  echo "[run] launch_manifest=${LAUNCH_MANIFEST_PATH}"

  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --demo-base "${DEMO_BASE}" \
    --demo-source-format sim_native \
    --save-root "${SAVE_ROOT}" \
    --run-tag "${RUN_TAG}" \
    --init-generator-from-bc-path "${BC_POLICY_PATH}" \
    --init-generator-from-bc-mode actor_only \
    --peg1-only-fixed-opponent \
    --fixed-opponent-policy-path "${PPO_EXPERT_CHECKPOINT}" \
    --fixed-opponent-policy-format rl_games \
    --fixed-opponent-rl-games-config-path "${PPO_EXPERT_CONFIG}" \
    --total-timesteps 5000000 \
    --device "${DEVICE}" \
    --num-envs 256 \
    --n-steps 128 \
    --val-split-fraction 0.1 \
    --val-split-level file \
    --val-split-seed 42 \
    --no-adaptive-disc-updates \
    --n-disc-updates 1 \
    --no-balance-disc-expert-loss \
    --freeze-disc-on-high-gen-acc \
    --disc-gen-acc-freeze-threshold 0.75 \
    --freeze-disc-until-gen-acc-below-threshold \
    --disc-gen-acc-unfreeze-threshold 0.60 \
    --disc-gen-exp-gap-freeze-threshold 0.0 \
    --disc-freeze-rounds 1 \
    --event-aware-disc-weighting \
    --event-expert-goal-weight 1.8 \
    --event-expert-bad-weight 0.6 \
    --event-expert-timeout-weight 0.8 \
    --event-generated-goal-weight 0.7 \
    --event-generated-bad-weight 1.8 \
    --event-generated-timeout-weight 1.2 \
    --wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-mode online \
    --wandb-run-name "${WANDB_RUN_NAME}"
done

echo "[done] Finished BC-bootstrapped GAIL sweep for all synthetic scales."
