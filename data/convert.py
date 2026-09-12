#!/usr/bin/env python3
# Save as: convert_to_pkl_node.py
import argparse
import os
import pickle
import signal
import time
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from imitation.data.types import TrajectoryWithRew
from klask_interfaces.msg import StampedPolygon


class ConvertNode(Node):
    def __init__(self, out_prefix: str, player: str, target_hz: float,
                 topic: str, scale: float, idle_timeout: float = 2.0):
        super().__init__('convert_to_pkl_node')
        self.out_prefix = out_prefix
        self.player = player
        self.topic = topic
        self.scale = scale
        self.idle_timeout = idle_timeout
        self.target_hz = float(target_hz)
        self._dumped = False
        self._last_msg_time = time.time()
        self._should_exit = False

        self.ts: List[float] = []
        self.bp: List[np.ndarray] = []   # ball pos
        self.bv: List[np.ndarray] = []   # ball vel
        self.lp: List[np.ndarray] = []   # left pos
        self.lv: List[np.ndarray] = []   # left vel
        self.rp: List[np.ndarray] = []   # right pos
        self.rv: List[np.ndarray] = []   # right vel

        # subscribe
        self.sub = self.create_subscription(StampedPolygon, self.topic, self.cb, 50)
        self.timer = self.create_timer(0.5, self._check_timeout)

        self.get_logger().info(
            f"Listening on {self.topic}; will write ONE file to {self.out_prefix}.pkl "
            f"(idle_timeout={self.idle_timeout}s; resampling to {self.target_hz:.2f} Hz)."
        )

        signal.signal(signal.SIGINT, self._sig_handler)
        signal.signal(signal.SIGTERM, self._sig_handler)

    def _check_timeout(self):
        if self._should_exit:
            return
        elapsed = time.time() - self._last_msg_time
        if len(self.ts) > 0 and elapsed > self.idle_timeout:
            self.get_logger().info(f"No messages for {elapsed:.1f}s, finishing up…")
            self._should_exit = True
            self._build_and_dump()
            raise KeyboardInterrupt

    def _pt(self, pts, idx):
        # robust to short arrays
        if idx >= len(pts):
            return np.zeros(2, dtype=np.float32)
        p = pts[idx]
        return np.array([float(p.x), float(p.y)], dtype=np.float32)

    def cb(self, msg: StampedPolygon):
        self._last_msg_time = time.time()
        t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        pts = msg.polygon.points
        # Expected order: 0:ball_pos, 1:ball_vel, 2:left_pos, 3:left_vel, 4:right_pos, 5:right_vel
        self.ts.append(t)
        self.bp.append(self._pt(pts, 0))
        self.bv.append(self._pt(pts, 1))
        self.lp.append(self._pt(pts, 2))
        self.lv.append(self._pt(pts, 3))
        self.rp.append(self._pt(pts, 4))
        self.rv.append(self._pt(pts, 5))

    @staticmethod
    def _linresample(src_t: np.ndarray, arr: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Linear resample each column of arr(T,C) onto grid(G,), returns (G,C)."""
        out = np.empty((len(grid), arr.shape[1]), dtype=np.float32)
        src_t64 = src_t.astype(np.float64, copy=False)
        grid64 = grid.astype(np.float64, copy=False)
        a64 = arr.astype(np.float64, copy=False)
        for d in range(arr.shape[1]):
            out[:, d] = np.interp(grid64, src_t64, a64[:, d]).astype(np.float32)
        return out

    def _build_and_dump(self):
        if self._dumped:
            return
        if not self.ts:
            self.get_logger().warn("No messages received; nothing to save.")
            self._dumped = True
            return

        # 1) sort + dedup
        idx = np.argsort(self.ts)
        ts = np.asarray(self.ts, dtype=np.float64)[idx]

        def sort_stack(lst):
            return np.stack(lst, axis=0)[idx].astype(np.float32)

        bp, bv, lp, lv, rp, rv = map(sort_stack, (self.bp, self.bv, self.lp, self.lv, self.rp, self.rv))

        keep = np.ones_like(ts, dtype=bool)
        keep[1:] = ts[1:] > ts[:-1]
        ts, bp, bv, lp, lv, rp, rv = ts[keep], bp[keep], bv[keep], lp[keep], lv[keep], rp[keep], rv[keep]

        if ts.shape[0] < 2:
            self.get_logger().warn("Fewer than 2 unique timestamps; saving without resampling.")
            obs = np.concatenate([bp, bv, lp, rp, lv, rv], axis=-1) * self.scale
            acts_full = np.concatenate([lv, rv], axis=-1) * self.scale
            acts = acts_full[:-1] if obs.shape[0] >= 2 else np.zeros((0, 4), dtype=np.float32)
            rews = np.zeros((acts.shape[0],), dtype=np.float32)
            traj = TrajectoryWithRew(obs=obs, acts=acts, rews=rews, infos={"t": ts[:acts.shape[0]+1]}, terminal=True)
            self._save_traj(traj)
            return

        # 2) build uniform grid at target_hz
        Hz = max(1e-6, self.target_hz)  # guard
        t0, t1 = float(ts[0]), float(ts[-1])
        dt = 1.0 / Hz
        grid = np.arange(t0, t1, dt, dtype=np.float64)  # [t0, t1)
        if grid.size < 2:
            # if the segment is very short, force two samples
            grid = np.array([t0, t1], dtype=np.float64)
            dt = float(np.diff(grid)[0])

        # 3) resample each 2D stream onto grid (linear)
        bp_r = self._linresample(ts, bp, grid)
        bv_r = self._linresample(ts, bv, grid)
        lp_r = self._linresample(ts, lp, grid)
        lv_r = self._linresample(ts, lv, grid)
        rp_r = self._linresample(ts, rp, grid)
        rv_r = self._linresample(ts, rv, grid)

        # 4) build obs/acts on the uniform grid, then scale
        # obs length N, acts length N-1
        obs = np.concatenate([bp_r, bv_r, lp_r, rp_r, lv_r, rv_r], axis=-1).astype(np.float32)
        acts_full = np.concatenate([lv_r, rv_r], axis=-1).astype(np.float32)

        obs *= self.scale
        acts_full *= self.scale
        acts = acts_full[:-1, :]  # align: obs = acts + 1

        # 5) rewards placeholder
        rews = np.zeros((acts.shape[0],), dtype=np.float32)

        # 5b) build infos: one dict per action
        T = acts.shape[0]
        infos = [{"t0": float(grid[i]), "t1": float(grid[i+1])} for i in range(T)]

        # 6) log timing stats
        grid_dt = np.diff(grid)
        self.get_logger().info(
            f"Resampled to {Hz:.4f} Hz: N_obs={obs.shape[0]}, N_acts={acts.shape[0]}, "
            f"dt mean={grid_dt.mean():.6f}s std={grid_dt.std():.6f}s "
            f"min/max={grid_dt.min():.6f}/{grid_dt.max():.6f}s"
        )

        # 7) dump
        traj = TrajectoryWithRew(obs=obs, acts=acts, rews=rews, infos=None, terminal=True)
        self._save_traj(traj)


    def _save_traj(self, traj: TrajectoryWithRew):
        fn = f"{self.out_prefix}.pkl"
        os.makedirs(os.path.dirname(self.out_prefix) or '.', exist_ok=True)
        with open(fn, "wb") as f:
            pickle.dump([traj], f)
        self.get_logger().info(f"Saved {fn} (obs: {traj.obs.shape}, acts: {traj.acts.shape})")
        self._dumped = True
        self.get_logger().info("Done. Wrote 1 file.")

    def _sig_handler(self, *_):
        self.get_logger().info("Shutdown signal received, saving data…")
        self._should_exit = True
        try:
            self._build_and_dump()
        except Exception as e:
            self.get_logger().error(f"Error during dump: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--player", choices=["p1", "p2", "both"], default="both",
                        help="Back-compat only; output is always a single combined PKL.")
    parser.add_argument("--target_hz", type=float, default=50.0,
                        help="Uniform resampling rate (Hz).")
    parser.add_argument("--topic", default="/ball_peg_states")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Linear scale applied to positions and velocities (after resampling).")
    parser.add_argument("--idle_timeout", type=float, default=2.0,
                        help="Exit after this many seconds without messages.")
    args = parser.parse_args()

    rclpy.init()
    node = ConvertNode(args.out_prefix, args.player, args.target_hz,
                       args.topic, args.scale, args.idle_timeout)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._build_and_dump()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
