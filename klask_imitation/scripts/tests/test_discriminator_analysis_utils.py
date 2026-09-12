from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "eval"
        / "discriminator_analysis_utils.py"
    )
    spec = importlib.util.spec_from_file_location("discriminator_analysis_utils", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_context_batch() -> tuple[dict[str, np.ndarray], dict[str, slice]]:
    layout = {
        "peg1_pos": slice(0, 2),
        "ball_pos": slice(2, 4),
    }
    obs = np.asarray(
        [
            [0.10, -0.08, 0.00, 0.00],
            [0.08, -0.05, 0.02, -0.01],
            [0.06, -0.02, -0.01, 0.03],
        ],
        dtype=np.float32,
    )
    return (
        {
            "obs": obs,
            "acts": np.zeros((obs.shape[0], 2), dtype=np.float32),
            "next_obs": np.array(obs, copy=True),
            "done": np.zeros((obs.shape[0],), dtype=np.float32),
        },
        layout,
    )


def test_compute_peg_ball_proximity_profile_prefers_close_distances():
    mod = _load_module()
    context_batch, layout = _make_context_batch()

    def score_fn(obs, acts, next_obs, done):
        peg = obs[:, layout["peg1_pos"]]
        ball = obs[:, layout["ball_pos"]]
        return -np.linalg.norm(peg - ball, axis=1).astype(np.float32)

    profile = mod.compute_peg_ball_proximity_profile(
        context_batch=context_batch,
        layout=layout,
        score_fn=score_fn,
        field_x=(-1.0, 1.0),
        field_y=(-1.0, 1.0),
        distance_samples=4,
        angle_samples=8,
        max_distance=0.6,
    )

    rows = profile["rows"]
    assert len(rows) == 4
    assert rows[0]["mean_logit"] > rows[-1]["mean_logit"]
    assert profile["summary"]["near_minus_far_mean_logit"] > 0.0
    assert profile["summary"]["spearman_distance_vs_mean_logit"] < 0.0
    assert profile["summary"]["best_distance"] == rows[0]["distance"]


def test_compute_peg_ball_proximity_profile_tracks_legal_coverage():
    mod = _load_module()
    context_batch, layout = _make_context_batch()

    def score_fn(obs, acts, next_obs, done):
        peg = obs[:, layout["peg1_pos"]]
        ball = obs[:, layout["ball_pos"]]
        return -np.linalg.norm(peg - ball, axis=1).astype(np.float32)

    profile = mod.compute_peg_ball_proximity_profile(
        context_batch=context_batch,
        layout=layout,
        score_fn=score_fn,
        field_x=(-1.0, 1.0),
        field_y=(-1.0, 1.0),
        distance_samples=3,
        angle_samples=4,
        max_distance=0.25,
        legal_bounds={"x": [-0.2, 0.2], "y": [-0.2, 0.05]},
    )

    coverages = [float(row["coverage_fraction"]) for row in profile["rows"]]
    assert coverages[0] == 1.0
    assert 0.0 < coverages[1] < 1.0
    assert coverages[-1] == 0.0


def test_plot_peg_ball_proximity_profile_writes_png():
    mod = _load_module()
    context_batch, layout = _make_context_batch()

    def score_fn(obs, acts, next_obs, done):
        peg = obs[:, layout["peg1_pos"]]
        ball = obs[:, layout["ball_pos"]]
        return -np.linalg.norm(peg - ball, axis=1).astype(np.float32)

    profile = mod.compute_peg_ball_proximity_profile(
        context_batch=context_batch,
        layout=layout,
        score_fn=score_fn,
        field_x=(-1.0, 1.0),
        field_y=(-1.0, 1.0),
        distance_samples=4,
        angle_samples=8,
        max_distance=0.5,
    )

    with tempfile.TemporaryDirectory() as tmp:
        out_png = Path(tmp) / "peg1_ball_proximity_profile.png"
        mod.plot_peg_ball_proximity_profile(profile, str(out_png))
        assert out_png.exists()
        assert out_png.stat().st_size > 0
