#!/usr/bin/env python3
"""Aggregate data-wall experiment outputs and produce scaling-law artifacts.

Expected input:
  - results_raw.csv produced by run_data_wall_experiment.py

Default output directory:
  - <results_dir>/aggregate
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PRIMARY_METRIC = "mean_env_reward"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate BC/GAIL scaling-law outputs.")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("results_raw.csv"),
        help="Path to run_data_wall_experiment results_raw.csv (default: ./results_raw.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for report/table/plot (default: <results_dir>/aggregate).",
    )
    parser.add_argument("--primary-metric", type=str, default=DEFAULT_PRIMARY_METRIC)
    parser.add_argument(
        "--plateau-improvement-threshold",
        type=float,
        default=0.05,
        help="Plateau condition: two consecutive synthetic scale jumps improve by less than this ratio.",
    )
    parser.add_argument("--control-hz", type=float, default=50.0, help="Control frequency for step->time conversion.")
    parser.add_argument("--human-overhead-factor", type=float, default=2.0, help="Multiplier for collection+cleaning overhead.")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        if math.isnan(out):
            return float("nan")
        return out
    except Exception:
        return float("nan")


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for row in reader:
            row["dataset_steps"] = int(row["dataset_steps"])
            row["seed"] = int(row.get("seed", 0))
            row["mean_env_reward"] = _safe_float(row.get("mean_env_reward"))
            row["goals_scored_per_episode"] = _safe_float(row.get("goals_scored_per_episode"))
            row["goals_conceded_per_episode"] = _safe_float(row.get("goals_conceded_per_episode"))
            row["mean_episode_length"] = _safe_float(row.get("mean_episode_length"))
            rows.append(row)
    return rows


def reduce_by_mean(rows: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    """Group by dataset/algorithm and average across seeds."""
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["algorithm"], row["dataset_id"], int(row["dataset_steps"]))
        grouped[key].append(row)

    out: List[Dict[str, Any]] = []
    for (algorithm, dataset_id, dataset_steps), items in grouped.items():
        metric_values = np.asarray([_safe_float(i.get(metric)) for i in items], dtype=np.float64)
        keep = metric_values[np.isfinite(metric_values)]
        mean_val = float(np.mean(keep)) if keep.size else float("nan")
        std_val = float(np.std(keep)) if keep.size else float("nan")
        base = dict(items[0])
        base["algorithm"] = algorithm
        base["dataset_id"] = dataset_id
        base["dataset_steps"] = int(dataset_steps)
        base[f"{metric}_mean"] = mean_val
        base[f"{metric}_std"] = std_val
        base["num_runs"] = int(len(items))
        out.append(base)
    out.sort(key=lambda r: (r["algorithm"], int(r["dataset_steps"])))
    return out


def find_plateau_steps(
    rows: List[Dict[str, Any]],
    *,
    metric_key: str,
    threshold: float,
) -> Tuple[Optional[int], List[Dict[str, Any]]]:
    """Return first step count satisfying two consecutive <threshold improvements."""
    if len(rows) < 3:
        return None, []

    diagnostics: List[Dict[str, Any]] = []
    for i in range(1, len(rows)):
        prev = rows[i - 1]
        curr = rows[i]
        prev_m = _safe_float(prev.get(metric_key))
        curr_m = _safe_float(curr.get(metric_key))
        denom = max(abs(prev_m), 1e-6)
        imp = float((curr_m - prev_m) / denom) if np.isfinite(prev_m) and np.isfinite(curr_m) else float("nan")
        diagnostics.append(
            {
                "from_steps": int(prev["dataset_steps"]),
                "to_steps": int(curr["dataset_steps"]),
                "improvement_ratio": imp,
            }
        )

    for i in range(1, len(diagnostics)):
        imp_prev = diagnostics[i - 1]["improvement_ratio"]
        imp_curr = diagnostics[i]["improvement_ratio"]
        if np.isfinite(imp_prev) and np.isfinite(imp_curr):
            if imp_prev < threshold and imp_curr < threshold:
                return int(diagnostics[i]["to_steps"]), diagnostics
    return None, diagnostics


def human_hours_for_steps(steps: int, *, control_hz: float, overhead_factor: float) -> float:
    return float(steps / control_hz / 3600.0 * overhead_factor)


def plot_scaling(
    *,
    rows: List[Dict[str, Any]],
    metric: str,
    out_path: Path,
    dpi: int,
) -> None:
    alg_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        alg_to_rows[row["algorithm"]].append(row)

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    colors = {"BC": "#1f77b4", "GAIL": "#2ca02c"}
    markers = {"human": "D", "synthetic": "o"}

    for algorithm, entries in sorted(alg_to_rows.items()):
        entries = sorted(entries, key=lambda r: int(r["dataset_steps"]))
        x = np.asarray([int(r["dataset_steps"]) for r in entries], dtype=np.float64)
        y = np.asarray([_safe_float(r.get(f"{metric}_mean")) for r in entries], dtype=np.float64)
        yerr = np.asarray([_safe_float(r.get(f"{metric}_std")) for r in entries], dtype=np.float64)
        color = colors.get(algorithm, None)
        ax.plot(x, y, label=algorithm, linewidth=2.0, color=color)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, alpha=0.4, capsize=3)
        for r in entries:
            ds_kind = str(r.get("dataset_kind", "synthetic"))
            marker = markers.get(ds_kind, "o")
            ax.scatter(
                [int(r["dataset_steps"])],
                [_safe_float(r.get(f"{metric}_mean"))],
                marker=marker,
                s=55,
                color=color,
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Dataset size (action timesteps, log scale)")
    ax.set_ylabel(metric)
    ax.set_title("Synthetic-data scaling law with human baseline reference")
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(frameon=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_csv = args.results_csv.expanduser().resolve()
    if not results_csv.exists():
        raise FileNotFoundError(
            f"results csv not found: {results_csv}. Pass --results-csv or run the script from a data-wall output directory."
        )
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else results_csv.parent / "aggregate"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_raw = load_rows(results_csv)
    rows = reduce_by_mean(rows_raw, metric=args.primary_metric)

    report: Dict[str, Any] = {
        "generated_at": str(np.datetime64("now")),
        "primary_metric": args.primary_metric,
        "plateau_improvement_threshold": float(args.plateau_improvement_threshold),
        "control_hz": float(args.control_hz),
        "human_overhead_factor": float(args.human_overhead_factor),
        "algorithms": {},
    }

    table_rows: List[Dict[str, Any]] = []
    for r in rows:
        table_rows.append(
            {
                "algorithm": r["algorithm"],
                "dataset_id": r["dataset_id"],
                "dataset_kind": r.get("dataset_kind", ""),
                "dataset_steps": int(r["dataset_steps"]),
                f"{args.primary_metric}_mean": _safe_float(r.get(f"{args.primary_metric}_mean")),
                f"{args.primary_metric}_std": _safe_float(r.get(f"{args.primary_metric}_std")),
                "num_runs": int(r.get("num_runs", 1)),
                "mean_env_reward": _safe_float(r.get("mean_env_reward")),
                "goals_scored_per_episode": _safe_float(r.get("goals_scored_per_episode")),
                "goals_conceded_per_episode": _safe_float(r.get("goals_conceded_per_episode")),
                "mean_episode_length": _safe_float(r.get("mean_episode_length")),
            }
        )

    for algorithm in sorted(set(r["algorithm"] for r in rows)):
        alg_rows = [r for r in rows if r["algorithm"] == algorithm]
        synth_rows = sorted(
            [r for r in alg_rows if str(r.get("dataset_kind", "")).lower() == "synthetic"],
            key=lambda r: int(r["dataset_steps"]),
        )
        plateau_steps, diagnostics = find_plateau_steps(
            synth_rows,
            metric_key=f"{args.primary_metric}_mean",
            threshold=float(args.plateau_improvement_threshold),
        )
        lower_bound = None
        if plateau_steps is None and synth_rows:
            lower_bound = int(max(int(r["dataset_steps"]) for r in synth_rows))

        report["algorithms"][algorithm] = {
            "plateau_steps": int(plateau_steps) if plateau_steps is not None else None,
            "plateau_human_hours": (
                human_hours_for_steps(
                    int(plateau_steps),
                    control_hz=args.control_hz,
                    overhead_factor=args.human_overhead_factor,
                )
                if plateau_steps is not None
                else None
            ),
            "lower_bound_steps_if_no_plateau": lower_bound,
            "lower_bound_human_hours_if_no_plateau": (
                human_hours_for_steps(
                    int(lower_bound),
                    control_hz=args.control_hz,
                    overhead_factor=args.human_overhead_factor,
                )
                if lower_bound is not None
                else None
            ),
            "improvement_diagnostics": diagnostics,
        }

    png_path = out_dir / f"scaling_law_{args.primary_metric}.png"
    report_path = out_dir / "data_wall_report.json"
    table_path = out_dir / "data_wall_table.csv"

    plot_scaling(rows=rows, metric=args.primary_metric, out_path=png_path, dpi=args.dpi)
    with report_path.open("w") as fh:
        json.dump(report, fh, indent=2)

    with table_path.open("w", newline="") as fh:
        fieldnames = list(table_rows[0].keys()) if table_rows else [
            "algorithm",
            "dataset_id",
            "dataset_kind",
            "dataset_steps",
            f"{args.primary_metric}_mean",
            f"{args.primary_metric}_std",
            "num_runs",
            "mean_env_reward",
            "goals_scored_per_episode",
            "goals_conceded_per_episode",
            "mean_episode_length",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in table_rows:
            writer.writerow(row)

    print(f"[out] {png_path}")
    print(f"[out] {report_path}")
    print(f"[out] {table_path}")


if __name__ == "__main__":
    main()
