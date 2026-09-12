#!/usr/bin/env bash
set -euo pipefail

###########################
# CONFIG — edit these, or override via the environment
###########################
# Repo root, resolved from this script's location unless KLASKTRON_ROOT is set.
KLASKTRON_ROOT="${KLASKTRON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KLASKTRON_DATA_DIR="${KLASKTRON_DATA_DIR:-${KLASKTRON_ROOT}/data}"

IN_ROOT="${IN_ROOT:-${KLASKTRON_DATA_DIR}/reestimated_output_rosbags_manually_cleaned}"
OUT_ROOT="${OUT_ROOT:-${KLASKTRON_DATA_DIR}/expert_trajectories_cleaned}"
TOPIC="/ball_peg_states"
PLAYER="both"
TARGET_HZ=50
SCALE=1.0
IDLE_TIMEOUT=2.0
PY_NODE="${PY_NODE:-${KLASKTRON_DATA_DIR}/convert.py}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-57}"
###########################

export ROS_DOMAIN_ID

log(){ echo "[$(date '+%F %T')] $*"; }

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ERROR: ros2 not in PATH (source your ROS setup)." >&2
  exit 1
fi

# Find all MCAP files recursively (no metadata.yaml anymore)
mapfile -t BAG_FILES < <(find "$IN_ROOT" -type f -name '*.mcap' | sort)
[[ ${#BAG_FILES[@]} -gt 0 ]] || { echo "No .mcap files under $IN_ROOT"; exit 0; }

log "Found ${#BAG_FILES[@]} .mcap file(s). Out root: $OUT_ROOT"
echo

for BAGFILE in "${BAG_FILES[@]}"; do
  rel="${BAGFILE#"$IN_ROOT"/}"
  rel_dir="$(dirname "$rel")"
  OUTDIR="$OUT_ROOT/$rel_dir"
  mkdir -p "$OUTDIR"

  base="$(basename "${BAGFILE%.*}")"
  OUT_PREFIX="$OUTDIR/$base"

  log "=== Converting: $BAGFILE"
  log "    Output:     ${OUT_PREFIX}.pkl"

  # 1) Start converter node (background)
  python3 "$PY_NODE" \
    --out_prefix "$OUT_PREFIX" \
    --player "$PLAYER" \
    --target_hz "$TARGET_HZ" \
    --topic "$TOPIC" \
    --scale "$SCALE" \
    --idle_timeout "$IDLE_TIMEOUT" &
  CONV_PID=$!

  # Give it a moment to initialize
  sleep 0.5

  # 2) Play the bag (single MCAP file; using bag time)
  ros2 bag play -s mcap "$BAGFILE" --clock

  # 3) Wait for converter to auto-exit (it will timeout after IDLE_TIMEOUT seconds)
  # Use a loop with timeout to avoid hanging forever
  WAIT_COUNT=0
  MAX_WAIT=20  # Maximum 20 seconds to wait after bag finishes
  while kill -0 "$CONV_PID" 2>/dev/null; do
    sleep 0.5
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ $WAIT_COUNT -gt $((MAX_WAIT * 2)) ]; then
      log "WARNING: Converter didn't exit after ${MAX_WAIT}s, killing it"
      kill -TERM "$CONV_PID" 2>/dev/null || true
      sleep 1
      kill -KILL "$CONV_PID" 2>/dev/null || true
      break
    fi
  done

  # 4) Show what got written
  ls -lh "$OUT_PREFIX".pkl 2>/dev/null || true
  echo
done

log "All bags processed. Output under: $OUT_ROOT"