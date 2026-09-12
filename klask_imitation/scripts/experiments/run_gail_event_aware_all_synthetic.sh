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

# Train event-aware GAIL policies across all official synthetic dataset scales.
# Current defaults use recovery-freeze discriminator control.
# Usage:
#   bash run_gail_event_aware_all_synthetic.sh [WANDB_PROJECT] [DEVICE]
#
# Example:
#   bash run_gail_event_aware_all_synthetic.sh klask-gail-event-aware-v4 cuda:0

WANDB_PROJECT="${1:-klask-gail-event-aware-v4}"
DEVICE="${2:-cuda:0}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
# ROS/colcon setup scripts may reference optional trace vars under nounset mode.
export COLCON_TRACE="${COLCON_TRACE:-0}"
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
set +u
source "${ROS_SETUP}"
set -u
cd "${ISAACLAB_ROOT}"

TRAIN_SCRIPT="${KLASKTRON_ROOT}/klask_imitation/scripts/training/train_gail_sweep.py"
SAVE_ROOT="${KLASKTRON_ROOT}/trained_models/data_wall_experiments/gail_runs"
DATASET_ROOT="${KLASKTRON_ROOT}/artifacts/data_wall/datasets"
LAUNCH_MANIFEST_ROOT="${SAVE_ROOT}/launch_manifests"

SCALES=("80000" "250000" "1000000" "5000000")
TAGS=("80k" "250k" "1Mio" "5Mio")

mkdir -p "${LAUNCH_MANIFEST_ROOT}"

for i in "${!SCALES[@]}"; do
  SCALE="${SCALES[$i]}"
  TAG="${TAGS[$i]}"
  DATASET_DIR="${DATASET_ROOT}/ppo_expert_synthetic_${SCALE}"
  DEMO_BASE="${DATASET_DIR}/trajectories"
  MANIFEST_PATH="${DATASET_DIR}/manifest.json"
  RUN_TAG="GAIL_DW_EXPERT_${TAG}_EVENT_AWARE_RECOVERY_FREEZE"
  WANDB_RUN_NAME="GAIL_${TAG}_event_aware_recovery_freeze"
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

  readarray -t DATASET_META < <(
    "${PYTHON_BIN}" - "${MANIFEST_PATH}" <<'PY'
import json
import os
import sys

manifest_path = sys.argv[1]
with open(manifest_path, "r", encoding="utf-8") as fh:
    manifest = json.load(fh)

dataset_dir = os.path.abspath(str(manifest.get("dataset_dir", "")))
trajectory_dir = os.path.abspath(str(manifest.get("trajectory_dir", "")))
actual_steps = int(manifest.get("actual_steps", -1))
target_steps = int(manifest.get("target_steps", -1))
num_files = int(manifest.get("num_files", -1))
num_trajectories = int(manifest.get("num_trajectories", -1))

print(dataset_dir)
print(trajectory_dir)
print(actual_steps)
print(target_steps)
print(num_files)
print(num_trajectories)
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
  if [[ "${RUN_TAG}" != *"${TAG}"* ]]; then
    echo "[error] Run tag mismatch: ${RUN_TAG} does not contain tag ${TAG}" >&2
    exit 1
  fi

  echo "[run] scale=${SCALE} tag=${TAG} run_tag=${RUN_TAG}"
  echo "[run] dataset_dir=${DATASET_DIR}"
  echo "[run] demo_base=${DEMO_BASE}"
  echo "[run] manifest=${MANIFEST_PATH}"
  echo "[run] dataset_actual_steps=${ACTUAL_STEPS} target_steps=${TARGET_STEPS} num_files=${NUM_FILES} num_trajectories=${NUM_TRAJECTORIES}"

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
    "train_script": "${TRAIN_SCRIPT}",
    "save_root": "${SAVE_ROOT}",
    "command": [
        "${PYTHON_BIN}",
        "${TRAIN_SCRIPT}",
        "--demo-base", "${DEMO_BASE}",
        "--demo-source-format", "sim_native",
        "--save-root", "${SAVE_ROOT}",
        "--run-tag", "${RUN_TAG}",
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

echo "[done] Finished event-aware recovery-freeze GAIL sweep for all synthetic scales."
