#!/usr/bin/env python3
"""Plot data-wall summary figures from results_index.json.

Generates:
1) Scaling law (goal_diff_per_episode, BC only) vs fixed opponent(s)
2) Mean-env-reward scaling law (BC only) vs fixed opponent(s)
3) All-vs-all heatmap (goal_diff_per_episode)
4) Stacked termination decomposition vs fixed opponent
5) Optional behavior-quality plot vs fixed opponent

Expected input:
  - results_index.json produced by run_data_wall_experiment.py

Default output directory:
  - <results_dir>/plots/summary
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TERMINATION_KEYS = [
    "term_rate_goal_scored",
    "term_rate_goal_conceded",
    "term_rate_time_out",
    "term_rate_player_in_goal",
    "term_rate_opponent_in_goal",
]

TERMINATION_LABELS = {
    "term_rate_goal_scored": "goal_scored",
    "term_rate_goal_conceded": "goal_conceded",
    "term_rate_time_out": "time_out",
    "term_rate_player_in_goal": "player_in_goal",
    "term_rate_opponent_in_goal": "opponent_in_goal",
}

# Human BC was trained on the augmented training split (effective ~250k timesteps).
HUMAN_EFFECTIVE_TRAIN_STEPS = 250_000
HUMAN_DATASET_IDS = {"HUMAN_DEMO", "HUMAN_DEMO_V2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BC data-wall summary figures.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results_index.json"),
        help="Path to results_index.json (default: ./results_index.json)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Directory to save plots (default: <results_dir>/plots/summary)",
    )
    parser.add_argument(
        "--scaling-opponents",
        nargs="+",
        default=None,
        help=(
            "Opponent ids (or aliases) for scaling plots. "
            "Default: ppo_expert + HUMAN_DEMO_V2 (if available)."
        ),
    )
    parser.add_argument(
        "--failure-opponent",
        default="ppo_expert",
        help="Opponent id (or alias) for failure-mode plots (default: ppo_expert)",
    )
    parser.add_argument(
        "--skip-behavior-quality",
        action="store_true",
        help="Skip optional behavior-quality figure.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a top-level list.")
    return [row for row in data if isinstance(row, dict)]


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def aggregate_pairs(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        p = row.get("policy_id")
        o = row.get("opponent_id")
        if p is None or o is None:
            continue
        grouped[(str(p), str(o))].append(row)

    agg: dict[tuple[str, str], dict[str, float]] = {}
    for key, group in grouped.items():
        result: dict[str, float] = {}
        numeric_keys = [
            "goal_diff_per_episode",
            "goals_scored_per_episode",
            "goals_conceded_per_episode",
            "goal_steps_rate",
            "conceded_steps_rate",
            "mean_env_reward",
            "mean_episode_length",
            *TERMINATION_KEYS,
        ]
        for k in numeric_keys:
            vals = [safe_float(r.get(k)) for r in group]
            vals = [v for v in vals if not math.isnan(v)]
            result[k] = float(np.mean(vals)) if vals else float("nan")
        agg[key] = result
    return agg


def policy_metadata(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pid = row.get("policy_id")
        if pid is not None:
            by_policy[str(pid)].append(row)

    meta: dict[str, dict[str, Any]] = {}
    for pid, group in by_policy.items():
        first = group[0]
        steps = [safe_float(r.get("dataset_steps")) for r in group]
        steps = [int(s) for s in steps if not math.isnan(s)]
        dataset_steps = int(round(np.median(steps))) if steps else None
        dataset_id = str(first.get("dataset_id", ""))
        dataset_kind = str(first.get("dataset_kind", "")).lower()
        if dataset_id.upper() in HUMAN_DATASET_IDS or dataset_kind == "human":
            dataset_steps = HUMAN_EFFECTIVE_TRAIN_STEPS
        meta[pid] = {
            "policy_id": pid,
            "algorithm": first.get("algorithm"),
            "dataset_id": first.get("dataset_id"),
            "dataset_kind": first.get("dataset_kind"),
            "dataset_steps": dataset_steps,
        }
    return meta


def format_steps_short(steps: int | None) -> str:
    if steps is None:
        return "n/a"
    if steps >= 1_000_000:
        v = steps / 1_000_000
        return f"{int(v)}M" if v.is_integer() else f"{v:.1f}M"
    if steps >= 1_000:
        v = steps / 1_000
        return f"{int(v)}k" if v.is_integer() else f"{v:.1f}k"
    return str(steps)


def policy_label(meta: dict[str, Any]) -> str:
    pid = str(meta.get("policy_id"))
    if pid == "ppo_expert":
        return "ppo_expert"

    dataset_id = str(meta.get("dataset_id", ""))
    dataset_kind = str(meta.get("dataset_kind", "")).lower()
    steps = meta.get("dataset_steps")
    steps_txt = format_steps_short(steps)

    if dataset_id.upper() in HUMAN_DATASET_IDS or dataset_kind == "human":
        return f"Human ({steps_txt})"
    return f"Synth {steps_txt}"


def scaling_legend_label(meta: dict[str, Any]) -> str:
    pid = str(meta.get("policy_id"))
    if pid == "ppo_expert":
        return "ppo_expert"

    algo = str(meta.get("algorithm", "")).upper()
    dataset_id = str(meta.get("dataset_id", ""))
    dataset_kind = str(meta.get("dataset_kind", "")).lower()
    steps_txt = format_steps_short(meta.get("dataset_steps"))

    if algo == "BC":
        if dataset_id.upper() in HUMAN_DATASET_IDS or dataset_kind == "human":
            return f"Human BC ({steps_txt})"
        return f"Synthetic BC ({steps_txt})"

    return policy_label(meta)


def add_reversed_scaling_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    # Keep each opponent's line/triangle pair together, but show later opponents first.
    pair_size = 2
    if len(handles) % pair_size == 0:
        order: list[int] = []
        for start in range(len(handles) - pair_size, -1, -pair_size):
            order.extend(range(start, start + pair_size))
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]
    else:
        handles = list(reversed(handles))
        labels = list(reversed(labels))

    ax.legend(handles, labels, frameon=False)


def policy_sort_key(meta: dict[str, Any]) -> tuple[int, float]:
    pid = str(meta.get("policy_id"))
    if pid == "ppo_expert":
        return (2, float("inf"))

    dataset_id = str(meta.get("dataset_id", ""))
    dataset_kind = str(meta.get("dataset_kind", "")).lower()
    steps = safe_float(meta.get("dataset_steps"))
    steps = steps if not math.isnan(steps) else float("inf")

    if dataset_id.upper() in HUMAN_DATASET_IDS or dataset_kind == "human":
        return (0, steps)
    return (1, steps)


def sanitize_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)


# Back-compatibility with artifacts produced before the 2026 release rename
# (the PPO expert was internally called "maurus_best" / "DW_MAURUS_*").
# Safe to delete once all artifacts under artifacts/ have been regenerated.
_LEGACY_ID_ALIASES = {
    "maurus_best": "ppo_expert",
    "maurus_expert": "ppo_expert",
    "maurus_vs_maurus": "expert_vs_expert",
    "maurus_vs_bc": "expert_vs_bc",
    "bc_vs_maurus": "bc_vs_expert",
}


def _legacy_id_candidates(token: str) -> list[str]:
    """Both spellings of *token*: pre-rename and post-rename."""
    out = [token]
    low = token.lower()
    if low in _LEGACY_ID_ALIASES:
        out.append(_LEGACY_ID_ALIASES[low])
    for legacy, current in _LEGACY_ID_ALIASES.items():
        if low == current:
            out.append(legacy)
    if "MAURUS" in token.upper():
        out.append(re.sub("(?i)maurus", "EXPERT", token))
    if "EXPERT" in token.upper():
        out.append(re.sub("(?i)expert", "MAURUS", token))
    return out


def resolve_policy_id(token: str, meta_by_policy: dict[str, dict[str, Any]]) -> str:
    for candidate in _legacy_id_candidates(token):
        if candidate in meta_by_policy:
            return candidate

    upper = token.upper()
    if upper in HUMAN_DATASET_IDS:
        for pid, meta in meta_by_policy.items():
            if str(meta.get("dataset_id", "")).upper() in HUMAN_DATASET_IDS:
                return pid
    if upper == "PPO_EXPERT" and "ppo_expert" in meta_by_policy:
        return "ppo_expert"

    raise ValueError(
        f"Could not resolve policy/opponent '{token}'. Available ids: {', '.join(sorted(meta_by_policy))}"
    )


def get_bc_policies(meta_by_policy: dict[str, dict[str, Any]]) -> list[str]:
    bc_ids = [
        pid
        for pid, meta in meta_by_policy.items()
        if str(meta.get("algorithm", "")).upper() == "BC"
    ]
    bc_ids.sort(key=lambda pid: policy_sort_key(meta_by_policy[pid]))
    return bc_ids


def get_human_bc_policy(meta_by_policy: dict[str, dict[str, Any]]) -> str | None:
    for pid, meta in meta_by_policy.items():
        if str(meta.get("algorithm", "")).upper() != "BC":
            continue
        dataset_id = str(meta.get("dataset_id", "")).upper()
        dataset_kind = str(meta.get("dataset_kind", "")).lower()
        if dataset_id in HUMAN_DATASET_IDS or dataset_kind == "human":
            return pid
    return None


def get_synth_bc_policies(meta_by_policy: dict[str, dict[str, Any]]) -> list[str]:
    synth_ids = []
    for pid, meta in meta_by_policy.items():
        if str(meta.get("algorithm", "")).upper() != "BC":
            continue
        dataset_id = str(meta.get("dataset_id", "")).upper()
        dataset_kind = str(meta.get("dataset_kind", "")).lower()
        if dataset_id in HUMAN_DATASET_IDS or dataset_kind == "human":
            continue
        synth_ids.append(pid)
    synth_ids.sort(key=lambda pid: policy_sort_key(meta_by_policy[pid]))
    return synth_ids


def plot_scaling_law(
    outdir: Path,
    meta_by_policy: dict[str, dict[str, Any]],
    agg: dict[tuple[str, str], dict[str, float]],
    opponents: list[str],
) -> Path:
    synth_bc_ids = get_synth_bc_policies(meta_by_policy)
    human_bc_id = get_human_bc_policy(meta_by_policy)

    x_synth = [safe_float(meta_by_policy[pid].get("dataset_steps")) for pid in synth_bc_ids]
    labels_x = [policy_label(meta_by_policy[pid]) for pid in synth_bc_ids]
    human_x = (
        safe_float(meta_by_policy[human_bc_id].get("dataset_steps"))
        if human_bc_id is not None
        else float("nan")
    )

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.8), constrained_layout=True)

    for opp in opponents:
        opp_meta = meta_by_policy.get(opp)
        opp_label = scaling_legend_label(opp_meta) if opp_meta is not None else opp
        y_synth = []
        for pid in synth_bc_ids:
            row = agg.get((pid, opp), {})
            gd_ep = safe_float(row.get("goal_diff_per_episode"))
            y_synth.append(gd_ep)

        (line,) = ax.plot(
            x_synth,
            y_synth,
            marker="o",
            linewidth=2.0,
            label=f"Synthetic BC vs {opp_label}",
        )

        # Plot the human-BC point separately so no line segment is drawn from Synth 80k to Human 250k.
        if human_bc_id is not None and not math.isnan(human_x):
            row_h = agg.get((human_bc_id, opp), {})
            human_y = safe_float(row_h.get("goal_diff_per_episode"))
            if not math.isnan(human_y):
                ax.scatter(
                    [human_x],
                    [human_y],
                    marker="^",
                    s=88,
                    facecolor=line.get_color(),
                    edgecolor="black",
                    linewidth=0.7,
                    zorder=5,
                    label=f"{scaling_legend_label(meta_by_policy[human_bc_id])} vs {opp_label}",
                )

    ax.set_xscale("log")
    ax.set_xticks(x_synth)
    ax.set_xticklabels(labels_x, rotation=25, ha="right")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.set_xlabel("Dataset size (BC training steps)")
    ax.set_ylabel("Goal diff / episode")
    add_reversed_scaling_legend(ax)

    fig.suptitle("Scaling law: BC performance vs dataset size", fontsize=13)
    path = outdir / "scaling_law.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_mean_env_reward_scaling(
    outdir: Path,
    meta_by_policy: dict[str, dict[str, Any]],
    agg: dict[tuple[str, str], dict[str, float]],
    opponents: list[str],
) -> Path:
    synth_bc_ids = get_synth_bc_policies(meta_by_policy)
    human_bc_id = get_human_bc_policy(meta_by_policy)

    x_synth = [safe_float(meta_by_policy[pid].get("dataset_steps")) for pid in synth_bc_ids]
    labels_x = [policy_label(meta_by_policy[pid]) for pid in synth_bc_ids]
    human_x = (
        safe_float(meta_by_policy[human_bc_id].get("dataset_steps"))
        if human_bc_id is not None
        else float("nan")
    )

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.8), constrained_layout=True)

    for opp in opponents:
        opp_meta = meta_by_policy.get(opp)
        opp_label = scaling_legend_label(opp_meta) if opp_meta is not None else opp
        y_synth = []
        for pid in synth_bc_ids:
            row = agg.get((pid, opp), {})
            y_synth.append(safe_float(row.get("mean_env_reward")))

        (line,) = ax.plot(
            x_synth,
            y_synth,
            marker="o",
            linewidth=2.0,
            label=f"Synthetic BC vs {opp_label}",
        )

        # Plot the human-BC point separately so no line segment is drawn between Synth and Human.
        if human_bc_id is not None and not math.isnan(human_x):
            row_h = agg.get((human_bc_id, opp), {})
            human_y = safe_float(row_h.get("mean_env_reward"))
            if not math.isnan(human_y):
                ax.scatter(
                    [human_x],
                    [human_y],
                    marker="^",
                    s=88,
                    facecolor=line.get_color(),
                    edgecolor="black",
                    linewidth=0.7,
                    zorder=5,
                    label=f"{scaling_legend_label(meta_by_policy[human_bc_id])} vs {opp_label}",
                )

    ax.set_xscale("log")
    ax.set_xticks(x_synth)
    ax.set_xticklabels(labels_x, rotation=25, ha="right")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.set_xlabel("Dataset size (BC training steps)")
    ax.set_ylabel("Mean env reward")
    add_reversed_scaling_legend(ax)

    fig.suptitle("Scaling law: mean env reward vs dataset size", fontsize=13)
    path = outdir / "mean_env_reward_scaling.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_all_vs_all_heatmap(
    outdir: Path,
    meta_by_policy: dict[str, dict[str, Any]],
    agg: dict[tuple[str, str], dict[str, float]],
) -> Path:
    policies = sorted(meta_by_policy.keys(), key=lambda pid: policy_sort_key(meta_by_policy[pid]))
    n = len(policies)
    raw_matrix = np.full((n, n), np.nan, dtype=float)
    for i, p in enumerate(policies):
        for j, o in enumerate(policies):
            raw_matrix[i, j] = safe_float(agg.get((p, o), {}).get("goal_diff_per_episode"))

    # Debias by removing symmetric component and projecting to an anti-symmetric matrix:
    # A = (M - M^T) / 2, with diagonal forced to 0.
    matrix = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        matrix[i, i] = 0.0

    pair_bias_values = []
    for i in range(n):
        for j in range(i + 1, n):
            a = raw_matrix[i, j]
            b = raw_matrix[j, i]
            if math.isnan(a) or math.isnan(b):
                continue
            debiased = 0.5 * (a - b)
            matrix[i, j] = debiased
            matrix[j, i] = -debiased
            pair_bias_values.append(0.5 * (a + b))

    pair_bias_arr = np.array(pair_bias_values, dtype=float)
    mean_abs_pair_bias = float(np.nanmean(np.abs(pair_bias_arr))) if pair_bias_arr.size else 0.0
    max_abs_pair_bias = float(np.nanmax(np.abs(pair_bias_arr))) if pair_bias_arr.size else 0.0
    diag_bias = np.diag(raw_matrix)
    finite_diag_bias = diag_bias[~np.isnan(diag_bias)]
    mean_abs_diag_bias = (
        float(np.nanmean(np.abs(finite_diag_bias))) if finite_diag_bias.size else 0.0
    )

    finite_vals = matrix[~np.isnan(matrix)]
    vmax = float(np.max(np.abs(finite_vals))) if finite_vals.size else 1.0
    vmax = vmax if vmax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(8.6, 7.2), constrained_layout=True)
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("Goal diff / episode")

    labels = [policy_label(meta_by_policy[pid]) for pid in policies]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Policy")
    ax.set_title("All-vs-all score grid")

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if math.isnan(val):
                txt = "n/a"
                color = "black"
            else:
                txt = f"{val:+.3f}"
                color = "white" if abs(val) > 0.45 * vmax else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    bias_note = (
        "Debiasing: A = (M - M^T)/2, diag(A)=0. "
        f"Removed symmetric bias |(M+M^T)/2| mean={mean_abs_pair_bias:.4f}, "
        f"max={max_abs_pair_bias:.4f}; raw |diag(M)| mean={mean_abs_diag_bias:.4f}."
    )
    fig.text(0.01, -0.045, bias_note, ha="left", va="bottom", fontsize=8, color="dimgray")

    path = outdir / "all_vs_all_heatmap_goal_diff_per_episode.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def policies_for_fixed_opponent(
    fixed_opp: str,
    meta_by_policy: dict[str, dict[str, Any]],
    agg: dict[tuple[str, str], dict[str, float]],
) -> list[str]:
    policies = []
    for pid in meta_by_policy:
        if pid == fixed_opp:
            continue
        if (pid, fixed_opp) in agg:
            policies.append(pid)
    policies.sort(key=lambda pid: policy_sort_key(meta_by_policy[pid]))
    return policies


def plot_termination_decomposition(
    outdir: Path,
    fixed_opp: str,
    meta_by_policy: dict[str, dict[str, Any]],
    agg: dict[tuple[str, str], dict[str, float]],
) -> Path:
    policies = policies_for_fixed_opponent(fixed_opp, meta_by_policy, agg)
    if not policies:
        raise ValueError(f"No matchup rows found for * vs {fixed_opp}.")

    x = np.arange(len(policies))
    width = 0.72
    fig, ax = plt.subplots(figsize=(12.2, 5.2), constrained_layout=True)

    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    bottom = np.zeros(len(policies), dtype=float)
    for k, color in zip(TERMINATION_KEYS, colors):
        vals = np.array([safe_float(agg[(pid, fixed_opp)].get(k)) for pid in policies], dtype=float)
        vals = np.nan_to_num(vals, nan=0.0)
        ax.bar(x, vals, width=width, bottom=bottom, label=TERMINATION_LABELS[k], color=color, alpha=0.92)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([policy_label(meta_by_policy[pid]) for pid in policies], rotation=25, ha="right")
    ax.set_ylabel("Termination rate")
    opp_label = policy_label(meta_by_policy[fixed_opp]) if fixed_opp in meta_by_policy else fixed_opp
    ax.set_title(f"Failure-mode decomposition vs {opp_label}")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=False,
        borderaxespad=0.0,
    )

    path = outdir / f"termination_decomposition_vs_{sanitize_name(fixed_opp)}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def plot_behavior_quality(
    outdir: Path,
    fixed_opp: str,
    meta_by_policy: dict[str, dict[str, Any]],
    agg: dict[tuple[str, str], dict[str, float]],
) -> Path:
    policies = policies_for_fixed_opponent(fixed_opp, meta_by_policy, agg)
    if not policies:
        raise ValueError(f"No matchup rows found for * vs {fixed_opp}.")

    labels = [policy_label(meta_by_policy[pid]) for pid in policies]
    x = np.arange(len(policies))

    mean_ep_len = np.array(
        [safe_float(agg[(pid, fixed_opp)].get("mean_episode_length")) for pid in policies],
        dtype=float,
    )
    goals_per_1k = np.array(
        [1000.0 * safe_float(agg[(pid, fixed_opp)].get("goal_steps_rate")) for pid in policies],
        dtype=float,
    )
    conceded_per_1k = np.array(
        [1000.0 * safe_float(agg[(pid, fixed_opp)].get("conceded_steps_rate")) for pid in policies],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)

    axes[0].bar(x, np.nan_to_num(mean_ep_len, nan=0.0), color="#4c78a8")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_ylabel("Mean episode length (steps)")
    axes[0].set_title("Episode length")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)

    w = 0.36
    axes[1].bar(x - w / 2, np.nan_to_num(goals_per_1k, nan=0.0), width=w, color="#1b9e77", label="Goals / 1000 steps")
    axes[1].bar(
        x + w / 2,
        np.nan_to_num(conceded_per_1k, nan=0.0),
        width=w,
        color="#d95f02",
        label="Conceded / 1000 steps",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_ylabel("Event rate")
    axes[1].set_title("Goal events per 1000 steps")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)
    axes[1].legend(frameon=False)

    opp_label = policy_label(meta_by_policy[fixed_opp]) if fixed_opp in meta_by_policy else fixed_opp
    fig.suptitle(f"Behavior quality vs {opp_label}", fontsize=13)

    path = outdir / f"behavior_quality_vs_{sanitize_name(fixed_opp)}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    results_path = args.results.expanduser().resolve()
    if not results_path.exists():
        raise FileNotFoundError(
            f"results index not found: {results_path}. Pass --results or run the script from a data-wall output directory."
        )

    rows = load_rows(results_path)
    meta_by_policy = policy_metadata(rows)
    agg = aggregate_pairs(rows)

    outdir = (
        args.outdir.expanduser().resolve()
        if args.outdir is not None
        else results_path.parent / "plots" / "summary"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    scaling_tokens = args.scaling_opponents
    if scaling_tokens is None:
        scaling_tokens = ["ppo_expert", "HUMAN_DEMO_V2"]

    scaling_opps = []
    for tok in scaling_tokens:
        try:
            pid = resolve_policy_id(tok, meta_by_policy)
        except ValueError:
            print(f"[WARN] Could not resolve scaling opponent '{tok}', skipping.")
            continue
        if pid not in scaling_opps:
            scaling_opps.append(pid)
    if not scaling_opps:
        raise ValueError("No valid scaling opponents were resolved.")
    fixed_opp = resolve_policy_id(args.failure_opponent, meta_by_policy)

    saved = []
    saved.append(plot_scaling_law(outdir, meta_by_policy, agg, scaling_opps))
    saved.append(plot_mean_env_reward_scaling(outdir, meta_by_policy, agg, scaling_opps))
    saved.append(plot_all_vs_all_heatmap(outdir, meta_by_policy, agg))
    saved.append(plot_termination_decomposition(outdir, fixed_opp, meta_by_policy, agg))
    if not args.skip_behavior_quality:
        saved.append(plot_behavior_quality(outdir, fixed_opp, meta_by_policy, agg))

    manifest_path = outdir / "plot_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "results_index": str(results_path),
                "out_dir": str(outdir),
                "scaling_opponents": scaling_opps,
                "failure_opponent": fixed_opp,
                "skip_behavior_quality": bool(args.skip_behavior_quality),
                "plots": [str(path) for path in saved],
            },
            f,
            indent=2,
        )

    print("Saved plots:")
    for path in saved:
        print(f" - {path}")
    print(f" - {manifest_path}")


if __name__ == "__main__":
    main()
