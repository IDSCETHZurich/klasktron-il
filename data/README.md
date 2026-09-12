# Expert Trajectories (PKL) — Overview

- How PKL files are generated
  - The ROS2 converter node (data/convert.py) subscribes to a StampedPolygon topic (default `/ball_peg_states`) and collects messages until an idle timeout or termination signal.
  - Each message appends timestamp and polygon points in the expected order:
    0: ball_pos, 1: ball_vel, 2: left_pos, 3: left_vel, 4: right_pos, 5: right_vel.
  - On finish it sorts by timestamp, removes duplicate timestamps, builds observation and action arrays, wraps them in an `imitation.data.types.TrajectoryWithRew`, and pickles a list containing that single trajectory.

- Where files are stored
  - The helper script `data/convert_all_bags.sh` scans input bag folders (default `data/reestimated_output_rosbags`) and writes outputs under `data/expert_trajectories`.
  - Output layout: `OUT_ROOT/<relative/path/to/bag>/<bag_basename>.pkl` (one .pkl per bag).

- PKL structure / formats
  - Each `.pkl` contains a Python list with one `TrajectoryWithRew` object.
  - Fields:
    - obs: float32 ndarray, shape (T, 12)
      - Columns: [bp_x, bp_y, bv_x, bv_y, lp_x, lp_y, rp_x, rp_y, lv_x, lv_y, rv_x, rv_y]
      - All values multiplied by the converter `--scale`.
    - acts: float32 ndarray, shape (T-1, 4)
      - Columns: [lv_x, lv_y, rv_x, rv_y] (streamed velocities; last obs has no action)
    - rews: float32 ndarray, shape (T-1,) — zeros placeholder
    - infos: infos = None, terminal: True
  - Resampling was performed at 50 hz (same as simulations frequency)

- Defaults and notes
  - Default topic: `/ball_peg_states`
  - Default scale: `1.0`
  - Default idle timeout: `2.0` seconds
  - Converter writes exactly one trajectory file per processed bag.

- Quick load example (Python)
```python
import pickle
from imitation.data.types import TrajectoryWithRew
with open("path/to/file.pkl","rb") as f:
    payload = pickle.load(f)  # [TrajectoryWithRew(...)]
traj = payload[0]
print(traj.obs.shape, traj.acts.shape)
```

