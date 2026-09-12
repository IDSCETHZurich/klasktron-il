from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _to_float_pair(values: Any, *, label: str) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != 2:
        raise ValueError(f"{label} must contain exactly two values, got shape {arr.shape}.")
    return float(arr[0]), float(arr[1])


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denom)


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty_like(sorted_x, dtype=np.float64)
    start = 0
    while start < sorted_x.size:
        end = start + 1
        while end < sorted_x.size and sorted_x[end] == sorted_x[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[start:end] = avg_rank
        start = end
    out = np.empty_like(ranks)
    out[order] = ranks
    return out


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    return _pearson_corr(_rankdata_average_ties(x), _rankdata_average_ties(y))


def _summarize_proximity_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "num_distances_total": 0,
            "num_distances_finite": 0,
            "near_distance": float("nan"),
            "near_mean_logit": float("nan"),
            "near_coverage_fraction": float("nan"),
            "far_distance": float("nan"),
            "far_mean_logit": float("nan"),
            "far_coverage_fraction": float("nan"),
            "near_minus_far_mean_logit": float("nan"),
            "best_distance": float("nan"),
            "best_mean_logit": float("nan"),
            "pearson_distance_vs_mean_logit": float("nan"),
            "spearman_distance_vs_mean_logit": float("nan"),
            "monotone_nonincreasing_fraction": float("nan"),
        }

    distances = np.asarray([float(r["distance"]) for r in rows], dtype=np.float64)
    mean_logits = np.asarray([float(r["mean_logit"]) for r in rows], dtype=np.float64)
    coverage = np.asarray([float(r["coverage_fraction"]) for r in rows], dtype=np.float64)
    finite = np.isfinite(mean_logits)

    summary = {
        "num_distances_total": int(len(rows)),
        "num_distances_finite": int(np.sum(finite)),
        "near_distance": float("nan"),
        "near_mean_logit": float("nan"),
        "near_coverage_fraction": float("nan"),
        "far_distance": float("nan"),
        "far_mean_logit": float("nan"),
        "far_coverage_fraction": float("nan"),
        "near_minus_far_mean_logit": float("nan"),
        "best_distance": float("nan"),
        "best_mean_logit": float("nan"),
        "pearson_distance_vs_mean_logit": float("nan"),
        "spearman_distance_vs_mean_logit": float("nan"),
        "monotone_nonincreasing_fraction": float("nan"),
    }
    if not np.any(finite):
        return summary

    finite_distances = distances[finite]
    finite_mean_logits = mean_logits[finite]
    finite_coverage = coverage[finite]
    near_idx = 0
    far_idx = finite_distances.size - 1
    best_idx = int(np.argmax(finite_mean_logits))
    summary.update(
        {
            "near_distance": float(finite_distances[near_idx]),
            "near_mean_logit": float(finite_mean_logits[near_idx]),
            "near_coverage_fraction": float(finite_coverage[near_idx]),
            "far_distance": float(finite_distances[far_idx]),
            "far_mean_logit": float(finite_mean_logits[far_idx]),
            "far_coverage_fraction": float(finite_coverage[far_idx]),
            "near_minus_far_mean_logit": float(
                finite_mean_logits[near_idx] - finite_mean_logits[far_idx]
            ),
            "best_distance": float(finite_distances[best_idx]),
            "best_mean_logit": float(finite_mean_logits[best_idx]),
            "pearson_distance_vs_mean_logit": _pearson_corr(
                finite_distances, finite_mean_logits
            ),
            "spearman_distance_vs_mean_logit": _spearman_corr(
                finite_distances, finite_mean_logits
            ),
        }
    )
    if finite_mean_logits.size >= 2:
        summary["monotone_nonincreasing_fraction"] = float(
            np.mean((np.diff(finite_mean_logits) <= 0.0).astype(np.float32))
        )
    return summary


def compute_peg_ball_proximity_profile(
    *,
    context_batch: Dict[str, np.ndarray],
    layout: Dict[str, slice],
    score_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    field_x: Tuple[float, float],
    field_y: Tuple[float, float],
    peg_key: str = "peg1_pos",
    ball_key: str = "ball_pos",
    distance_samples: int = 25,
    angle_samples: int = 24,
    min_distance: float = 0.0,
    max_distance: Optional[float] = None,
    legal_bounds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if int(distance_samples) <= 0:
        raise ValueError("distance_samples must be > 0.")
    if int(angle_samples) <= 0:
        raise ValueError("angle_samples must be > 0.")
    if float(min_distance) < 0.0:
        raise ValueError("min_distance must be >= 0.")
    if peg_key not in layout:
        raise KeyError(f"Missing peg key in layout: {peg_key!r}")
    if ball_key not in layout:
        raise KeyError(f"Missing ball key in layout: {ball_key!r}")

    peg_slice = layout[peg_key]
    ball_slice = layout[ball_key]
    if int(peg_slice.stop - peg_slice.start) != 2:
        raise ValueError(f"{peg_key} must be 2D XY slice, got {peg_slice}.")
    if int(ball_slice.stop - ball_slice.start) != 2:
        raise ValueError(f"{ball_key} must be 2D XY slice, got {ball_slice}.")

    obs_ctx = np.asarray(context_batch["obs"], dtype=np.float32)
    acts_ctx = np.asarray(context_batch["acts"], dtype=np.float32)
    next_obs_ctx = np.asarray(context_batch["next_obs"], dtype=np.float32)
    done_ctx = np.asarray(context_batch["done"], dtype=np.float32)
    if obs_ctx.shape[0] == 0:
        raise ValueError("No context transitions available for peg-ball proximity analysis.")

    probe_x = (
        _to_float_pair(legal_bounds["x"], label="legal_bounds['x']")
        if legal_bounds
        else field_x
    )
    probe_y = (
        _to_float_pair(legal_bounds["y"], label="legal_bounds['y']")
        if legal_bounds
        else field_y
    )
    if max_distance is None:
        max_distance = float(math.hypot(probe_x[1] - probe_x[0], probe_y[1] - probe_y[0]))
    if float(max_distance) < float(min_distance):
        raise ValueError(
            f"max_distance ({float(max_distance)}) must be >= min_distance ({float(min_distance)})."
        )

    distances = np.linspace(
        float(min_distance),
        float(max_distance),
        int(distance_samples),
        dtype=np.float32,
    )
    angles = np.linspace(0.0, 2.0 * math.pi, int(angle_samples), endpoint=False, dtype=np.float32)
    angle_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)

    ball_pos = obs_ctx[:, ball_slice]
    base_peg = obs_ctx[:, peg_slice]
    base_next_peg = next_obs_ctx[:, peg_slice]
    total_candidates = int(obs_ctx.shape[0] * int(angle_samples))
    rows: List[Dict[str, Any]] = []

    for distance in distances:
        target_positions = ball_pos[:, None, :] + float(distance) * angle_dirs[None, :, :]
        valid = np.ones(target_positions.shape[:2], dtype=bool)
        valid &= target_positions[:, :, 0] >= float(probe_x[0])
        valid &= target_positions[:, :, 0] <= float(probe_x[1])
        valid &= target_positions[:, :, 1] >= float(probe_y[0])
        valid &= target_positions[:, :, 1] <= float(probe_y[1])

        ctx_idx, angle_idx = np.nonzero(valid)
        num_samples = int(ctx_idx.size)
        row: Dict[str, Any] = {
            "distance": float(distance),
            "num_samples": int(num_samples),
            "num_contexts_covered": int(np.unique(ctx_idx).size) if num_samples > 0 else 0,
            "num_angles_with_valid_samples": int(np.unique(angle_idx).size) if num_samples > 0 else 0,
            "coverage_fraction": float(num_samples / total_candidates) if total_candidates > 0 else 0.0,
            "mean_logit": float("nan"),
            "std_logit": float("nan"),
            "stderr_logit": float("nan"),
        }
        if num_samples == 0:
            rows.append(row)
            continue

        target_flat = target_positions[ctx_idx, angle_idx]
        obs_mod = np.array(obs_ctx[ctx_idx], copy=True)
        next_obs_mod = np.array(next_obs_ctx[ctx_idx], copy=True)
        delta = target_flat - base_peg[ctx_idx]
        obs_mod[:, peg_slice] = target_flat
        next_obs_mod[:, peg_slice] = base_next_peg[ctx_idx] + delta

        logits = np.asarray(
            score_fn(obs_mod, acts_ctx[ctx_idx], next_obs_mod, done_ctx[ctx_idx]),
            dtype=np.float32,
        ).reshape(-1)
        if logits.shape[0] != num_samples:
            raise ValueError(
                f"score_fn returned {logits.shape[0]} logits, expected {num_samples}."
            )
        row["mean_logit"] = float(np.mean(logits))
        row["std_logit"] = float(np.std(logits))
        row["stderr_logit"] = float(np.std(logits) / math.sqrt(max(1, num_samples)))
        rows.append(row)

    return {
        "peg_key": str(peg_key),
        "ball_key": str(ball_key),
        "distance_samples": int(distance_samples),
        "angle_samples": int(angle_samples),
        "min_distance": float(min_distance),
        "max_distance": float(max_distance),
        "field_x": [float(field_x[0]), float(field_x[1])],
        "field_y": [float(field_y[0]), float(field_y[1])],
        "legal_bounds": legal_bounds,
        "context_samples": int(obs_ctx.shape[0]),
        "rows": rows,
        "summary": _summarize_proximity_rows(rows),
    }


def plot_peg_ball_proximity_profile(profile: Dict[str, Any], out_path: str) -> None:
    rows = list(profile.get("rows", []))
    x = np.asarray([float(r["distance"]) for r in rows], dtype=np.float64)
    mean_logits = np.asarray([float(r["mean_logit"]) for r in rows], dtype=np.float64)
    stderr = np.asarray([float(r["stderr_logit"]) for r in rows], dtype=np.float64)
    coverage = np.asarray([float(r["coverage_fraction"]) for r in rows], dtype=np.float64)
    finite = np.isfinite(mean_logits)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.5, 7.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.4]},
    )

    if np.any(finite):
        x_finite = x[finite]
        mean_finite = mean_logits[finite]
        stderr_finite = np.nan_to_num(stderr[finite], nan=0.0)
        axes[0].plot(x_finite, mean_finite, color="tab:blue", linewidth=2.0)
        axes[0].scatter(x_finite, mean_finite, color="tab:blue", s=28)
        axes[0].fill_between(
            x_finite,
            mean_finite - stderr_finite,
            mean_finite + stderr_finite,
            color="tab:blue",
            alpha=0.18,
        )
    axes[0].set_ylabel("mean discriminator logit")
    axes[0].set_title("Peg1-ball proximity probe")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, coverage, color="tab:orange", linewidth=2.0)
    axes[1].scatter(x, coverage, color="tab:orange", s=24)
    axes[1].set_ylabel("coverage")
    axes[1].set_xlabel("peg1-ball center distance")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.25)

    summary = dict(profile.get("summary", {}))
    subtitle_bits = []
    near_minus_far = float(summary.get("near_minus_far_mean_logit", float("nan")))
    spearman = float(summary.get("spearman_distance_vs_mean_logit", float("nan")))
    if np.isfinite(near_minus_far):
        subtitle_bits.append(f"near-far={near_minus_far:.3f}")
    if np.isfinite(spearman):
        subtitle_bits.append(f"spearman={spearman:.3f}")
    if subtitle_bits:
        axes[0].text(
            0.01,
            0.98,
            " | ".join(subtitle_bits),
            transform=axes[0].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
