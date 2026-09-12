#!/usr/bin/env python3
"""Plot aggregated sparse-RL evaluation metrics across kept RL conditions.

Default conditions:
1) Sparse RL (random init)
2) Sparse RL (BC init)

Outputs:
- termination_modes_stacked.png
- key_metrics_errorbars.png
- aggregated_eval_metrics.json

Default output directory:
- artifacts/plots/eval/sparse_rl_summary
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Condition:
    label: str
    run_dir: Path


def _trained_models_dir() -> Path:
    return Path(
        os.environ.get(
            "KLASKTRON_TRAINED_MODELS_DIR",
            Path(__file__).resolve().parents[3] / "trained_models",
        )
    )


def _artifacts_dir() -> Path:
    return Path(
        os.environ.get(
            "KLASKTRON_ARTIFACTS_DIR",
            Path(__file__).resolve().parents[3] / "artifacts",
        )
    )


def _default_out_dir() -> Path:
    return _artifacts_dir() / "plots" / "eval" / "sparse_rl_summary"


DEFAULT_CONDITIONS: List[Condition] = [
    Condition(
        label="Sparse / Random Init",
        run_dir=_trained_models_dir() / "klask_sparse_rl_20260223_150905_RANDOM_sparse_RL",
    ),
    Condition(
        label="Sparse / BC Init",
        run_dir=_trained_models_dir() / "klask_sparse_rl_20260223_111724_GAIL_BC_sparse_RL",
    ),
]


TERMINATION_KEYS = [
    "term_rate_time_out",
    "term_rate_goal_scored",
    "term_rate_goal_conceded",
    "term_rate_player_in_goal",
    "term_rate_opponent_in_goal",
]

TERMINATION_LABELS = {
    "term_rate_time_out": "time_out",
    "term_rate_goal_scored": "goal_scored",
    "term_rate_goal_conceded": "goal_conceded",
    "term_rate_player_in_goal": "player_in_goal",
    "term_rate_opponent_in_goal": "opponent_in_goal",
}

KEY_METRICS = [
    "mean_env_reward",
    "mean_episode_length",
    "goals_scored_per_episode",
    "goals_conceded_per_episode",
    "goals_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and plot sparse-RL evaluation summaries across RL conditions."
    )
    parser.add_argument(
        "--num-evals",
        type=int,
        default=3,
        help="How many latest eval_* folders per condition to aggregate.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots and aggregate JSON (default: artifacts/plots/eval/sparse_rl_summary).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _as_float(d: Dict[str, Any], key: str) -> float:
    if key not in d:
        raise KeyError(f"Missing key '{key}' in summary.")
    return float(d[key])


def _aggregate(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
    }


def collect_eval_summaries(run_dir: Path, num_evals: int) -> List[Path]:
    eval_root = _artifacts_dir() / "evaluations" / run_dir.name
    search_root = eval_root if eval_root.exists() else run_dir
    eval_dirs = sorted([p for p in search_root.iterdir() if p.is_dir() and p.name.startswith("eval_")])
    if len(eval_dirs) < num_evals:
        raise RuntimeError(
            f"{search_root}: expected at least {num_evals} eval_* dirs, found {len(eval_dirs)}"
        )
    selected = eval_dirs[-num_evals:]
    summary_paths = [p / "summary.json" for p in selected]
    missing = [p for p in summary_paths if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing summary files: {missing}")
    return summary_paths


def aggregate_condition(cond: Condition, num_evals: int) -> Dict[str, Any]:
    summary_paths = collect_eval_summaries(cond.run_dir, num_evals=num_evals)
    summaries = [_load_json(p) for p in summary_paths]

    # Derived metric: goals_ratio = scored / conceded (per run), then aggregate across runs.
    goals_ratio_values: List[float] = []
    for s in summaries:
        gs = _as_float(s, "goals_scored_per_episode")
        gc = _as_float(s, "goals_conceded_per_episode")
        goals_ratio_values.append(float("nan") if gc == 0.0 else gs / gc)

    metrics: Dict[str, Dict[str, float]] = {}
    for key in KEY_METRICS:
        if key == "goals_ratio":
            metrics[key] = _aggregate(goals_ratio_values)
        else:
            metrics[key] = _aggregate([_as_float(s, key) for s in summaries])

    for key in TERMINATION_KEYS:
        metrics[key] = _aggregate([_as_float(s, key) for s in summaries])

    # Keep some additional rates from summary for convenience.
    extra_keys = ["goal_episode_rate", "goal_steps_rate", "conceded_steps_rate"]
    for key in extra_keys:
        metrics[key] = _aggregate([_as_float(s, key) for s in summaries])

    return {
        "label": cond.label,
        "run_dir": str(cond.run_dir),
        "eval_dirs_used": [p.parent.name for p in summary_paths],
        "metrics": metrics,
    }


def plot_termination_stacked(aggregates: List[Dict[str, Any]], out_path: Path, dpi: int) -> None:
    labels = [a["label"] for a in aggregates]
    x = np.arange(len(labels))
    width = 0.72
    bottom = np.zeros(len(labels), dtype=np.float64)

    colors = {
        "term_rate_time_out": "#4C78A8",
        "term_rate_goal_scored": "#59A14F",
        "term_rate_goal_conceded": "#E15759",
        "term_rate_player_in_goal": "#F28E2B",
        "term_rate_opponent_in_goal": "#B07AA1",
    }

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)

    for key in TERMINATION_KEYS:
        vals = np.array([a["metrics"][key]["mean"] for a in aggregates], dtype=np.float64)
        vals = np.nan_to_num(vals, nan=0.0)
        ax.bar(
            x,
            vals,
            width=width,
            bottom=bottom,
            label=TERMINATION_LABELS[key],
            color=colors[key],
            edgecolor="white",
            linewidth=0.6,
        )
        bottom += vals

    for xi, total in zip(x, bottom):
        ax.text(xi, total + 0.01, f"{total:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Mean termination-flag rate per episode")
    ax.set_title(
        "Termination mode composition across conditions\n"
    )
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        title="Termination",
    )

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_key_metrics(aggregates: List[Dict[str, Any]], out_path: Path, dpi: int) -> None:
    labels = [a["label"] for a in aggregates]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax_reward, ax_len = axes[0]
    ax_goals, ax_ratio = axes[1]

    # mean_env_reward
    mean_reward = np.array([a["metrics"]["mean_env_reward"]["mean"] for a in aggregates], dtype=np.float64)
    std_reward = np.array([a["metrics"]["mean_env_reward"]["std"] for a in aggregates], dtype=np.float64)
    ax_reward.bar(x, mean_reward, yerr=std_reward, capsize=4, color="#4C78A8")
    ax_reward.set_title("Mean Env Reward")
    ax_reward.set_xticks(x)
    ax_reward.set_xticklabels(labels, rotation=12, ha="right")
    ax_reward.grid(axis="y", alpha=0.25, linewidth=0.6)

    # mean_episode_length
    mean_len = np.array([a["metrics"]["mean_episode_length"]["mean"] for a in aggregates], dtype=np.float64)
    std_len = np.array([a["metrics"]["mean_episode_length"]["std"] for a in aggregates], dtype=np.float64)
    ax_len.bar(x, mean_len, yerr=std_len, capsize=4, color="#9C755F")
    ax_len.set_title("Mean Episode Length")
    ax_len.set_xticks(x)
    ax_len.set_xticklabels(labels, rotation=12, ha="right")
    ax_len.grid(axis="y", alpha=0.25, linewidth=0.6)

    # goals scored/conceded
    w = 0.36
    mean_scored = np.array([a["metrics"]["goals_scored_per_episode"]["mean"] for a in aggregates], dtype=np.float64)
    std_scored = np.array([a["metrics"]["goals_scored_per_episode"]["std"] for a in aggregates], dtype=np.float64)
    mean_conc = np.array([a["metrics"]["goals_conceded_per_episode"]["mean"] for a in aggregates], dtype=np.float64)
    std_conc = np.array([a["metrics"]["goals_conceded_per_episode"]["std"] for a in aggregates], dtype=np.float64)
    ax_goals.bar(x - w / 2, mean_scored, width=w, yerr=std_scored, capsize=3, label="scored", color="#59A14F")
    ax_goals.bar(x + w / 2, mean_conc, width=w, yerr=std_conc, capsize=3, label="conceded", color="#E15759")
    ax_goals.set_title("Goals per Episode")
    ax_goals.set_xticks(x)
    ax_goals.set_xticklabels(labels, rotation=12, ha="right")
    ax_goals.legend(frameon=True)
    ax_goals.grid(axis="y", alpha=0.25, linewidth=0.6)

    # goals ratio
    mean_ratio = np.array([a["metrics"]["goals_ratio"]["mean"] for a in aggregates], dtype=np.float64)
    std_ratio = np.array([a["metrics"]["goals_ratio"]["std"] for a in aggregates], dtype=np.float64)
    ax_ratio.bar(x, mean_ratio, yerr=std_ratio, capsize=4, color="#B07AA1")
    ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_ratio.set_title("Goals Ratio (scored / conceded)")
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels(labels, rotation=12, ha="right")
    ax_ratio.grid(axis="y", alpha=0.25, linewidth=0.6)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else _default_out_dir().expanduser().resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregates = [aggregate_condition(cond, num_evals=args.num_evals) for cond in DEFAULT_CONDITIONS]

    term_png = out_dir / "termination_modes_stacked.png"
    key_png = out_dir / "key_metrics_errorbars.png"
    json_out = out_dir / "aggregated_eval_metrics.json"

    plot_termination_stacked(aggregates, term_png, dpi=args.dpi)
    plot_key_metrics(aggregates, key_png, dpi=args.dpi)

    with json_out.open("w") as f:
        json.dump({"conditions": aggregates}, f, indent=2)

    print(f"[out] Wrote {term_png}")
    print(f"[out] Wrote {key_png}")
    print(f"[out] Wrote {json_out}")


if __name__ == "__main__":
    main()
