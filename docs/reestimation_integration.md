# Reestimation Integration Contract

The human state-reestimation pipeline lives in the separate
[klask_reestimate](https://github.com/IDSCETHZurich/klask_reestimate)
repository and is intentionally kept outside this benchmark repo.

## How The Pipeline Works

1. **`klask_reestimate`** runs a ROS2 node (`reestimate.py` / `BallPegDetection`)
   that subscribes to `/raw_image`, detects AprilTags, warps the board, segments
   ball and pegs, and filters states with a Kalman filter. It publishes
   reestimated states as `StampedPolygon` on `/ball_peg_states`.

2. **`data/convert.py`** (this repo) subscribes to `/ball_peg_states`, resamples
   at 50 Hz, and saves one `TrajectoryWithRew` pkl file per recorded bag.

3. **`data/convert_all_bags.sh`** batch-converts all reestimated output bags
   into pkl files under `data/expert_trajectories_cleaned/`.

4. The training scripts load those pkls via `demo_dataset.py`, which handles
   the `human_raw` to simulation coordinate transform.

## `/ball_peg_states` Index Mapping

```text
0: ball_position
1: ball_velocity
2: left_peg_position
3: left_peg_velocity
4: right_peg_position
5: right_peg_velocity
```

## Expected PKL Format

Each `.pkl` file produced by `convert.py` contains a Python list with one
`imitation.data.types.TrajectoryWithRew` object:

| Field | Shape | Description |
|---|---|---|
| `obs` | `(T, 12)` float32 | `[bp_x, bp_y, bv_x, bv_y, lp_x, lp_y, rp_x, rp_y, lv_x, lv_y, rv_x, rv_y]` |
| `acts` | `(T-1, 4)` float32 | `[lv_x, lv_y, rv_x, rv_y]` (streamed velocities) |
| `rews` | `(T-1,)` float32 | Zeros placeholder |
| `infos` | `None` | |
| `terminal` | `True` | |

Resampled at 50 Hz (matching simulation frequency).

## Validating New Exports

After producing new pkl files from the reestimation pipeline, validate them
before training:

```python
from klasktron.reestimation import validate_export_dir

result = validate_export_dir("data/expert_trajectories_cleaned")
print(f"valid={result.valid} files={result.n_files} traj={result.n_trajectories} steps={result.n_steps}")
if result.errors:
    for err in result.errors:
        print(f"  ERROR: {err}")
```

## Importing Into The Benchmark Tier Layout

To convert raw exports into the chunked `human/trajectories/*.pkl` +
`metadata.json` format expected by the Hugging Face downloader:

```python
from klasktron.reestimation import validate_and_import

result, tier_dir = validate_and_import(
    export_dir="data/expert_trajectories_cleaned",
    output_dir="data",
    tier_name="human",
    source_format="human_raw",
)
print(f"Written to {tier_dir}: {result.n_trajectories} traj, {result.n_steps} steps")
```

Or use the dataset preparation script which also generates the augmented tier:

```bash
python klask_imitation/scripts/prepare_dataset_tiers.py
```

## Future Merge Strategy

If the external reestimation repository is folded into this one later, place
the code behind `klask_imitation/klasktron/reestimation/`. The benchmark-facing
API (`validate_export_dir`, `validate_and_import`) stays fixed while the
implementation evolves independently.
