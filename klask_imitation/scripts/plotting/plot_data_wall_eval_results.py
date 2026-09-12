#!/usr/bin/env python3
"""Plot BC data-wall evaluation results from results_raw.csv.

Expected input:
  - results_raw.csv produced by run_data_wall_experiment.py

Outputs:
  - one PNG per metric, with one subplot per matchup
  - aggregated_matchup_metrics.csv
  - plot_manifest.json

Default output directory:
  - <results_dir>/plots/eval
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_METRICS = [
    "mean_env_reward",
    "goal_diff_per_episode",
    "goals_scored_per_episode",
    "goals_conceded_per_episode",
]
DEFAULT_GRID_METRICS = [
    "goal_diff_per_episode",
    "mean_env_reward",
]
DEFAULT_BASELINE_MATCHUP = "expert_vs_expert"
DEFAULT_MATCHUP_ORDER = [
    "bc_vs_expert",
    "bc_vs_bc",
    "expert_vs_bc",
    "expert_vs_expert",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot policy-vs-policy BC data-wall evaluation results.")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("results_raw.csv"),
        help="Path to results_raw.csv (default: ./results_raw.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <results_dir>/plots/eval).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help=f"Comma-separated metric names. Default: {','.join(DEFAULT_METRICS)}",
    )
    parser.add_argument(
        "--x-field",
        type=str,
        default="dataset_steps",
        choices=["dataset_steps", "dataset_steps_used_train"],
        help="X-axis field for scaling curves.",
    )
    parser.add_argument(
        "--baseline-matchup",
        type=str,
        default=DEFAULT_BASELINE_MATCHUP,
        help="If present, draw this matchup as horizontal baseline in other subplots.",
    )
    parser.add_argument("--title-prefix", type=str, default="Data-Wall Evaluation")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--matchups",
        type=str,
        default="",
        help="Optional comma-separated subset (default: all in file, ordered).",
    )
    parser.add_argument(
        "--plot-agent-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to generate 6x6 agent score-grid heatmaps.",
    )
    parser.add_argument(
        "--grid-metrics",
        type=str,
        default=",".join(DEFAULT_GRID_METRICS),
        help=f"Comma-separated metrics for grid plots. Default: {','.join(DEFAULT_GRID_METRICS)}",
    )
    parser.add_argument(
        "--plot-debiased-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to generate pairwise de-biased skill/bias grids from both directions.",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        if math.isnan(out):
            return float("nan")
        return out
    except Exception:
        return float("nan")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _format_score_text(val: float, metric_key: str) -> str:
    """Format matrix-cell scores for display."""
    if not np.isfinite(val):
        return "NA"
    if metric_key in {"goal_diff_per_episode", "mean_env_reward"}:
        txt = f"{val:.2g}"  # two significant figures
        if txt in {"-0", "-0.0", "-0.00"}:
            return "0"
        return txt
    return f"{val:.2f}"


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows: List[Dict[str, Any]] = []
        for row in reader:
            row["matchup"] = str(row.get("matchup", ""))
            row["dataset_id"] = str(row.get("dataset_id", ""))
            row["dataset_kind"] = str(row.get("dataset_kind", "")).lower()
            row["dataset_steps"] = _safe_int(row.get("dataset_steps", 0), default=0)
            row["dataset_steps_used_train"] = _safe_int(row.get("dataset_steps_used_train", 0), default=0)
            rows.append(row)
    return rows


def _is_expert_agent(agent_id: str) -> bool:
    s = str(agent_id).lower()
    return s in {"ppo_expert", "ppo_expert"}


def _dataset_id_from_policy_id(policy_id: str) -> str:
    m = re.match(r"^klask_bc_\d{8}_\d{6}_(.+)$", str(policy_id))
    if m:
        return str(m.group(1))
    return str(policy_id)


def _parse_scaled_number(token: str) -> int:
    m = re.fullmatch(r"(?i)(\d+(?:\.\d+)?)([km]?)", str(token).strip())
    if not m:
        return 0
    base = float(m.group(1))
    suf = m.group(2).lower()
    if suf == "k":
        base *= 1_000.0
    elif suf == "m":
        base *= 1_000_000.0
    return int(round(base))


def _parse_steps_from_dataset_id(dataset_id: str) -> int:
    for pat in [
        r"(?i)DW_[A-Z0-9]+_(\d+(?:\.\d+)?[kKmM]?)(?:_|$)",
        r"(?i)_(\d+(?:\.\d+)?[kKmM]?)(?:_|$)",
    ]:
        for m in re.finditer(pat, str(dataset_id)):
            steps = _parse_scaled_number(m.group(1))
            if steps > 0:
                return steps
    return 0


def _default_meta_from_agent_id(agent_id: str) -> Dict[str, Any]:
    if _is_expert_agent(agent_id):
        return {"dataset_id": "ppo_expert", "dataset_kind": "synthetic", "dataset_steps": 0}
    ds_id = _dataset_id_from_policy_id(agent_id)
    kind = "human" if "HUMAN" in ds_id.upper() else "synthetic"
    steps = _parse_steps_from_dataset_id(ds_id)
    return {"dataset_id": ds_id, "dataset_kind": kind, "dataset_steps": int(steps)}


def _collect_agent_meta(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Collect dataset metadata for each agent id present in the results."""
    meta: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        policy_id = str(row.get("policy_id", ""))
        opp_id = str(row.get("opponent_id", ""))
        ds_id = str(row.get("dataset_id", ""))
        ds_kind = str(row.get("dataset_kind", "")).lower()
        ds_steps = _safe_int(row.get("dataset_steps", 0), default=0)

        if policy_id:
            meta.setdefault(policy_id, _default_meta_from_agent_id(policy_id))
            # For BC policy rows, policy-side dataset fields are reliable.
            if not _is_expert_agent(policy_id):
                if ds_id:
                    meta[policy_id]["dataset_id"] = ds_id
                if ds_kind:
                    meta[policy_id]["dataset_kind"] = ds_kind
                if ds_steps > 0:
                    meta[policy_id]["dataset_steps"] = int(ds_steps)

        if opp_id:
            meta.setdefault(opp_id, _default_meta_from_agent_id(opp_id))
    return meta


def _agent_label(agent_id: str, meta: Dict[str, Dict[str, Any]]) -> str:
    if _is_expert_agent(agent_id):
        return "PPO Expert"

    info = meta.get(agent_id, {})
    ds_id = str(info.get("dataset_id", agent_id))
    ds_steps = _safe_int(info.get("dataset_steps", 0), default=0)
    ds_id_up = ds_id.upper()

    if "HUMAN" in ds_id_up:
        return "BC Human"
    if ds_steps >= 1_000_000 and ds_steps % 1_000_000 == 0:
        return f"BC {ds_steps // 1_000_000}Mio"
    if ds_steps >= 1_000:
        return f"BC {ds_steps // 1_000}k"
    if ds_steps > 0:
        return f"BC {ds_steps}"
    return f"BC {ds_id}"


def _agent_sort_key(agent_id: str, meta: Dict[str, Dict[str, Any]]) -> Tuple[int, int, str]:
    if _is_expert_agent(agent_id):
        return (2, 0, agent_id)
    info = meta.get(agent_id, {})
    kind = str(info.get("dataset_kind", "")).lower()
    steps = _safe_int(info.get("dataset_steps", 0), default=0)
    if "human" in kind:
        return (1, 0, agent_id)
    return (0, steps if steps > 0 else 10**12, agent_id)


def _row_to_pair(row: Dict[str, Any]) -> Tuple[str, str]:
    matchup = str(row.get("matchup", ""))
    policy_id = str(row.get("policy_id", ""))
    opponent_id = str(row.get("opponent_id", ""))
    if matchup in ("bc_vs_expert", "bc_vs_bc"):
        return policy_id, opponent_id if opponent_id else policy_id
    if matchup in ("expert_vs_bc", "expert_vs_expert"):
        return policy_id, opponent_id if opponent_id else policy_id
    return policy_id, opponent_id


def _aggregate_pair_metric(
    rows: List[Dict[str, Any]],
    metric: str,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        player, opponent = _row_to_pair(row)
        if not player or not opponent:
            continue
        v = _safe_float(row.get(metric))
        if np.isfinite(v):
            grouped[(player, opponent)].append(v)

    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, vals in grouped.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "count": float(arr.size),
        }
    return out


def _build_agent_metric_matrix(
    *,
    rows: List[Dict[str, Any]],
    metric: str,
) -> Tuple[List[str], Dict[str, Dict[str, Any]], np.ndarray]:
    meta = _collect_agent_meta(rows)
    agents = sorted(meta.keys(), key=lambda a: _agent_sort_key(a, meta))
    if not agents:
        raise RuntimeError("No agents found for agent matrix.")

    pair_stats = _aggregate_pair_metric(rows, metric)
    n = len(agents)
    mat = np.full((n, n), np.nan, dtype=np.float64)

    agent_to_idx = {a: i for i, a in enumerate(agents)}
    for (player, opponent), stats in pair_stats.items():
        if player in agent_to_idx and opponent in agent_to_idx:
            mat[agent_to_idx[player], agent_to_idx[opponent]] = float(stats["mean"])
    return agents, meta, mat


def _compute_debiased_and_bias(raw_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = int(raw_mat.shape[0])
    skill = np.full((n, n), np.nan, dtype=np.float64)
    bias = np.full((n, n), np.nan, dtype=np.float64)

    for i in range(n):
        diag = raw_mat[i, i]
        # By definition, de-biased self-matchup score is zero.
        skill[i, i] = 0.0
        bias[i, i] = diag if np.isfinite(diag) else np.nan

    for i in range(n):
        for j in range(i + 1, n):
            a = raw_mat[i, j]
            b = raw_mat[j, i]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            s = 0.5 * (a - b)
            d = 0.5 * (a + b)
            skill[i, j] = s
            skill[j, i] = -s
            bias[i, j] = d
            bias[j, i] = d
    return skill, bias


def _bias_summary_text(bias_mat: np.ndarray) -> str:
    n = int(bias_mat.shape[0])
    off_diag: List[float] = []
    diag_vals: List[float] = []
    for i in range(n):
        v = bias_mat[i, i]
        if np.isfinite(v):
            diag_vals.append(float(v))
        for j in range(n):
            if i == j:
                continue
            x = bias_mat[i, j]
            if np.isfinite(x):
                off_diag.append(float(x))
    if off_diag:
        mean_abs_off = float(np.mean(np.abs(off_diag)))
        max_abs_off = float(np.max(np.abs(off_diag)))
    else:
        mean_abs_off = float("nan")
        max_abs_off = float("nan")
    diag_mean = float(np.mean(diag_vals)) if diag_vals else float("nan")
    return (
        f"bias: mean|offdiag|={mean_abs_off:.2g}, "
        f"max|offdiag|={max_abs_off:.2g}, "
        f"diag_mean={diag_mean:.2g}"
    )


def _plot_agent_matrix(
    *,
    matrix: np.ndarray,
    agents: List[str],
    meta: Dict[str, Dict[str, Any]],
    metric_key: str,
    metric_label: str,
    title: str,
    note_text: Optional[str],
    out_path: Path,
    dpi: int,
) -> None:
    n = int(matrix.shape[0])
    finite = matrix[np.isfinite(matrix)]
    if finite.size:
        lim = float(np.max(np.abs(finite)))
        lim = max(lim, 1e-6)
    else:
        lim = 1.0

    fig, ax = plt.subplots(figsize=(1.6 * n + 2.5, 1.2 * n + 2.3), constrained_layout=True)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#d9d9d9")
    im = ax.imshow(matrix, cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest", aspect="auto")

    x_labels = [_agent_label(a, meta) for a in agents]
    y_labels = [_agent_label(a, meta) for a in agents]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Player")
    ax.set_title(title)
    if note_text:
        ax.text(
            0.01,
            0.99,
            note_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 2.2},
        )

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if np.isfinite(val):
                txt = _format_score_text(float(val), metric_key)
                color = "white" if abs(val) > 0.55 * lim else "black"
            else:
                txt = "NA"
                color = "#666666"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_label)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _write_agent_matrix_csv(
    *,
    matrix: np.ndarray,
    agents: List[str],
    meta: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    labels = [_agent_label(a, meta) for a in agents]
    fieldnames = ["player"] + labels
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i, row_label in enumerate(labels):
            row: Dict[str, Any] = {"player": row_label}
            for j, col_label in enumerate(labels):
                val = matrix[i, j]
                row[col_label] = float(val) if np.isfinite(val) else ""
            writer.writerow(row)


def _aggregate_rows(
    *,
    rows: List[Dict[str, Any]],
    metrics: List[str],
    x_field: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("matchup", "")),
            str(row.get("dataset_id", "")),
            str(row.get("dataset_kind", "")),
            int(row.get(x_field, 0)),
        )
        grouped[key].append(row)

    out: List[Dict[str, Any]] = []
    for (matchup, dataset_id, dataset_kind, x_value), items in grouped.items():
        agg: Dict[str, Any] = {
            "matchup": matchup,
            "dataset_id": dataset_id,
            "dataset_kind": dataset_kind,
            x_field: int(x_value),
            "num_runs": int(len(items)),
        }
        for metric in metrics:
            values = np.asarray([_safe_float(r.get(metric)) for r in items], dtype=np.float64)
            keep = values[np.isfinite(values)]
            agg[f"{metric}_mean"] = float(np.mean(keep)) if keep.size else float("nan")
            agg[f"{metric}_std"] = float(np.std(keep)) if keep.size else float("nan")
        out.append(agg)

    out.sort(key=lambda r: (str(r["matchup"]), int(r[x_field]), str(r["dataset_id"])))
    return out


def _ordered_matchups(rows: List[Dict[str, Any]], requested: List[str]) -> List[str]:
    present = sorted({str(r.get("matchup", "")) for r in rows if str(r.get("matchup", ""))})
    if requested:
        return [m for m in requested if m in present]
    ordered = [m for m in DEFAULT_MATCHUP_ORDER if m in present]
    for m in present:
        if m not in ordered:
            ordered.append(m)
    return ordered


def _plot_metric_by_matchup(
    *,
    agg_rows: List[Dict[str, Any]],
    metric: str,
    x_field: str,
    matchups: List[str],
    baseline_matchup: str,
    title_prefix: str,
    out_path: Path,
    dpi: int,
) -> None:
    if not matchups:
        raise ValueError("No matchups to plot.")

    baseline_rows = [r for r in agg_rows if str(r.get("matchup")) == baseline_matchup]
    baseline_val = None
    if baseline_rows:
        vals = np.asarray([_safe_float(r.get(f"{metric}_mean")) for r in baseline_rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            baseline_val = float(np.mean(vals))

    n = len(matchups)
    cols = min(2, n)
    rows_n = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(7.0 * cols, 4.4 * rows_n), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).flatten()

    for ax_idx, matchup in enumerate(matchups):
        ax = axes_arr[ax_idx]
        entries = [r for r in agg_rows if str(r.get("matchup")) == matchup]
        entries = sorted(entries, key=lambda r: int(r.get(x_field, 0)))

        xs = np.asarray([int(r.get(x_field, 0)) for r in entries], dtype=np.float64)
        ys = np.asarray([_safe_float(r.get(f"{metric}_mean")) for r in entries], dtype=np.float64)
        yerr = np.asarray([_safe_float(r.get(f"{metric}_std")) for r in entries], dtype=np.float64)
        kinds = [str(r.get("dataset_kind", "")) for r in entries]
        ids = [str(r.get("dataset_id", "")) for r in entries]

        if xs.size:
            ax.plot(xs, ys, color="#1f77b4", linewidth=2.0, alpha=0.9)
            ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor="#1f77b4", alpha=0.35, capsize=3)
            for x, y, kind, ds_id in zip(xs, ys, kinds, ids):
                marker = "D" if kind == "human" else "o"
                color = "#ff7f0e" if kind == "human" else "#1f77b4"
                ax.scatter([x], [y], marker=marker, s=58, color=color, edgecolor="black", linewidth=0.35, zorder=3)
                ax.annotate(
                    ds_id,
                    (x, y),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.85,
                )

            positive_x = xs[xs > 0]
            if positive_x.size >= 2 and positive_x.max() / max(1.0, positive_x.min()) > 4.0:
                ax.set_xscale("log")

        if baseline_val is not None and matchup != baseline_matchup:
            ax.axhline(
                baseline_val,
                color="#d62728",
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                label=f"{baseline_matchup} baseline",
            )
            ax.legend(loc="best", fontsize=8, frameon=True)

        ax.set_title(matchup)
        ax.set_xlabel(x_field)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25, linewidth=0.7)

    for k in range(len(matchups), len(axes_arr)):
        fig.delaxes(axes_arr[k])

    fig.suptitle(f"{title_prefix}: {metric}", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _write_aggregated_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    results_csv = args.results_csv.expanduser().resolve()
    if not results_csv.exists():
        raise FileNotFoundError(
            f"results csv not found: {results_csv}. Pass --results-csv or run the script from a data-wall output directory."
        )

    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else results_csv.parent / "plots" / "eval"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    if not metrics:
        raise ValueError("No metrics provided.")
    grid_metrics = [m.strip() for m in str(args.grid_metrics).split(",") if m.strip()]
    requested_matchups = [m.strip() for m in str(args.matchups).split(",") if m.strip()]

    rows = _read_rows(results_csv)
    if not rows:
        raise RuntimeError(f"no rows in {results_csv}")
    agg_rows = _aggregate_rows(rows=rows, metrics=metrics, x_field=str(args.x_field))
    matchups = _ordered_matchups(agg_rows, requested=requested_matchups)
    if not matchups:
        raise RuntimeError("No matching matchups to plot.")

    generated_plots: List[str] = []
    for metric in metrics:
        out_png = out_dir / f"{metric}_by_matchup.png"
        _plot_metric_by_matchup(
            agg_rows=agg_rows,
            metric=metric,
            x_field=str(args.x_field),
            matchups=matchups,
            baseline_matchup=str(args.baseline_matchup),
            title_prefix=str(args.title_prefix),
            out_path=out_png,
            dpi=int(args.dpi),
        )
        generated_plots.append(str(out_png))

    generated_grid_plots: List[str] = []
    generated_debiased_plots: List[str] = []
    generated_bias_plots: List[str] = []
    generated_debiased_csv: List[str] = []
    generated_bias_csv: List[str] = []
    if bool(args.plot_agent_grid):
        for metric in grid_metrics:
            agents, meta, raw_mat = _build_agent_metric_matrix(
                rows=rows,
                metric=metric,
            )
            skill_mat, bias_mat = _compute_debiased_and_bias(raw_mat)
            bias_note = _bias_summary_text(bias_mat)

            out_png = out_dir / f"agent_score_grid_{metric}.png"
            _plot_agent_matrix(
                matrix=raw_mat,
                agents=agents,
                meta=meta,
                metric_key=metric,
                metric_label=f"Average {metric}",
                title=f"{str(args.title_prefix)}: Agent Score Grid ({metric})",
                note_text=bias_note,
                out_path=out_png,
                dpi=int(args.dpi),
            )
            generated_grid_plots.append(str(out_png))

            if bool(args.plot_debiased_grid):
                skill_png = out_dir / f"agent_score_grid_debiased_{metric}.png"
                _plot_agent_matrix(
                    matrix=skill_mat,
                    agents=agents,
                    meta=meta,
                    metric_key=metric,
                    metric_label=f"De-biased {metric}",
                    title=f"{str(args.title_prefix)}: De-biased Skill Grid ({metric})",
                    note_text=bias_note,
                    out_path=skill_png,
                    dpi=int(args.dpi),
                )
                generated_debiased_plots.append(str(skill_png))

                bias_png = out_dir / f"agent_score_grid_bias_{metric}.png"
                _plot_agent_matrix(
                    matrix=bias_mat,
                    agents=agents,
                    meta=meta,
                    metric_key=metric,
                    metric_label=f"Side-bias residual ({metric})",
                    title=f"{str(args.title_prefix)}: Side-Bias Grid ({metric})",
                    note_text=bias_note,
                    out_path=bias_png,
                    dpi=int(args.dpi),
                )
                generated_bias_plots.append(str(bias_png))

                skill_csv = out_dir / f"agent_score_grid_debiased_{metric}.csv"
                _write_agent_matrix_csv(
                    matrix=skill_mat,
                    agents=agents,
                    meta=meta,
                    out_path=skill_csv,
                )
                generated_debiased_csv.append(str(skill_csv))

                bias_csv = out_dir / f"agent_score_grid_bias_{metric}.csv"
                _write_agent_matrix_csv(
                    matrix=bias_mat,
                    agents=agents,
                    meta=meta,
                    out_path=bias_csv,
                )
                generated_bias_csv.append(str(bias_csv))

    agg_csv = out_dir / "aggregated_matchup_metrics.csv"
    _write_aggregated_csv(agg_rows, agg_csv)

    manifest = {
        "results_csv": str(results_csv),
        "out_dir": str(out_dir),
        "metrics": metrics,
        "x_field": str(args.x_field),
        "baseline_matchup": str(args.baseline_matchup),
        "matchups": matchups,
        "plots": generated_plots,
        "grid_metrics": grid_metrics,
        "grid_plots": generated_grid_plots,
        "debiased_grid_plots": generated_debiased_plots,
        "bias_grid_plots": generated_bias_plots,
        "debiased_grid_csv": generated_debiased_csv,
        "bias_grid_csv": generated_bias_csv,
        "aggregated_csv": str(agg_csv),
    }
    manifest_path = out_dir / "plot_manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[out] {agg_csv}")
    for p in generated_plots:
        print(f"[out] {p}")
    for p in generated_grid_plots:
        print(f"[out] {p}")
    for p in generated_debiased_plots:
        print(f"[out] {p}")
    for p in generated_bias_plots:
        print(f"[out] {p}")
    for p in generated_debiased_csv:
        print(f"[out] {p}")
    for p in generated_bias_csv:
        print(f"[out] {p}")
    print(f"[out] {manifest_path}")


if __name__ == "__main__":
    main()
