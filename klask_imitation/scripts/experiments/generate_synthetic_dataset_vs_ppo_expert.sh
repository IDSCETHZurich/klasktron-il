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

# Generate a sim-native synthetic dataset for a peg1 policy against PPO expert agent.
#
# Default policy:
#   ${KLASKTRON_ROOT}/trained_models/data_wall_experiments/gail_runs/
#   klask_gail_20260317_150012_GAIL_DW_EXPERT_80k_BC_ANCHORED_RELATIVE_REWARD_PRETERM_VS_PPO_EXPERT/
#   ppo_policy_gail_final.zip
#
# Usage:
#   bash generate_synthetic_dataset_vs_ppo_expert.sh [options] [-- <extra generator args>]
#
# Example:
#   bash generate_synthetic_dataset_vs_ppo_expert.sh \
#     --device cuda:0 \
#     --target-steps 80000

GENERATOR_SCRIPT="${KLASKTRON_ROOT}/klask_imitation/scripts/experiments/generate_synthetic_dataset.py"
DATASET_ROOT="${KLASKTRON_ROOT}/artifacts/data_wall/datasets"
DEFAULT_POLICY_PATH="${KLASKTRON_ROOT}/trained_models/data_wall_experiments/gail_runs/klask_gail_20260317_150012_GAIL_DW_EXPERT_80k_BC_ANCHORED_RELATIVE_REWARD_PRETERM_VS_PPO_EXPERT/ppo_policy_gail_final.zip"
PPO_EXPERT_DIR="${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one"
PPO_EXPERT_CHECKPOINT="${PPO_EXPERT_DIR}/klask.pth"
PPO_EXPERT_CONFIG="${PPO_EXPERT_DIR}/agent.yaml"

POLICY_PATH="${DEFAULT_POLICY_PATH}"
DEVICE="cuda:0"
TARGET_STEPS="80000"
NUM_ENVS="256"
MIN_TRAJ_LEN="1"
FORCE="true"
OUTPUT_DIR=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash generate_synthetic_dataset_vs_ppo_expert.sh [options] [-- <extra generator args>]

Options:
  --policy-path PATH   SB3 policy checkpoint to roll out.
  --device DEVICE      Torch device for rollout. Default: cuda:0
  --target-steps N     Number of action steps to collect. Default: 80000
  --num-envs N         Number of parallel envs. Default: 256
  --min-traj-len N     Minimum retained trajectory length. Default: 1
  --output-dir PATH    Output dataset directory. Default: auto-derived from policy run dir.
  --force              Overwrite an existing manifest/output dir. Default.
  --no-force           Reuse existing dataset if present.
  -h, --help           Show this message.

Examples:
  bash generate_synthetic_dataset_vs_ppo_expert.sh
  bash generate_synthetic_dataset_vs_ppo_expert.sh --device cuda:1 --target-steps 160000
  bash generate_synthetic_dataset_vs_ppo_expert.sh --policy-path /path/to/ppo_policy_gail_final.zip -- --chunk-size 32
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path)
      POLICY_PATH="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --target-steps)
      TARGET_STEPS="$2"
      shift 2
      ;;
    --num-envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    --min-traj-len)
      MIN_TRAJ_LEN="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --force)
      FORCE="true"
      shift
      ;;
    --no-force)
      FORCE="false"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "${POLICY_PATH}" ]]; then
  echo "[error] Missing policy checkpoint: ${POLICY_PATH}" >&2
  exit 1
fi
if [[ ! -f "${PPO_EXPERT_CHECKPOINT}" ]]; then
  echo "[error] Missing PPO expert checkpoint: ${PPO_EXPERT_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${PPO_EXPERT_CONFIG}" ]]; then
  echo "[error] Missing PPO expert config: ${PPO_EXPERT_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${GENERATOR_SCRIPT}" ]]; then
  echo "[error] Missing generator script: ${GENERATOR_SCRIPT}" >&2
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  POLICY_RUN_DIR="$(basename "$(dirname "${POLICY_PATH}")")"
  OUTPUT_DIR="${DATASET_ROOT}/${POLICY_RUN_DIR}_vs_ppo_expert_${TARGET_STEPS}"
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export COLCON_TRACE="${COLCON_TRACE:-0}"
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
set +u
source "${ROS_SETUP}"
set -u
cd "${ISAACLAB_ROOT}"

CMD=(
  "${PYTHON_BIN}"
  "${GENERATOR_SCRIPT}"
  --checkpoint "${POLICY_PATH}"
  --policy-format sb3_ppo
  --opponent-policy-path "${PPO_EXPERT_CHECKPOINT}"
  --opponent-policy-format rl_games
  --opponent-rl-games-config-path "${PPO_EXPERT_CONFIG}"
  --rl-games-config-path "${PPO_EXPERT_CONFIG}"
  --target-steps "${TARGET_STEPS}"
  --num_envs "${NUM_ENVS}"
  --device "${DEVICE}"
  --min-traj-len "${MIN_TRAJ_LEN}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${FORCE}" == "true" ]]; then
  CMD+=(--force)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[run] policy_path=${POLICY_PATH}"
echo "[run] opponent_path=${PPO_EXPERT_CHECKPOINT}"
echo "[run] opponent_config=${PPO_EXPERT_CONFIG}"
echo "[run] target_steps=${TARGET_STEPS} num_envs=${NUM_ENVS} device=${DEVICE}"
echo "[run] output_dir=${OUTPUT_DIR}"

"${CMD[@]}"
