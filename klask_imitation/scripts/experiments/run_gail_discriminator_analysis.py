#!/usr/bin/env python3
"""Run discriminator overfitting + reward-map analysis for official GAIL runs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAINED_MODELS_DIR = Path(
    os.environ.get("KLASKTRON_TRAINED_MODELS_DIR", _REPO_ROOT / "trained_models")
)
_ARTIFACTS_DIR = Path(
    os.environ.get("KLASKTRON_ARTIFACTS_DIR", _REPO_ROOT / "artifacts")
)
DEFAULT_GAIL_RUNS_DIR = str(_TRAINED_MODELS_DIR / "data_wall_experiments" / "gail_runs")
DEFAULT_OUTPUT_ROOT = str(_ARTIFACTS_DIR / "data_wall" / "analysis" / "gail_discriminator")
# Override with the ISAACLAB_ROOT environment variable.
DEFAULT_ISAACLAB_CWD = str(Path(os.environ.get("ISAACLAB_ROOT", Path.home() / "IsaacLab")))
DEFAULT_OFFICIAL_TAGS = [
    "GAIL_Human",
    "GAIL_DW_EXPERT_80k",
    "GAIL_DW_EXPERT_250k",
    "GAIL_DW_EXPERT_1Mio",
    "GAIL_DW_EXPERT_5Mio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve latest official GAIL runs and execute discriminator overfitting + "
            "reward-map analysis in one batch."
        )
    )
    parser.add_argument("--gail-runs-dir", type=str, default=DEFAULT_GAIL_RUNS_DIR)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--official-tags",
        type=str,
        default=",".join(DEFAULT_OFFICIAL_TAGS),
        help="Comma-separated run_tag values to resolve as latest runs.",
    )

    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--eval-script", type=str, default=None)
    parser.add_argument("--isaaclab-cwd", type=str, default=DEFAULT_ISAACLAB_CWD)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--num-envs", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-demos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to keep demo-heavy plotting inside evaluate_gail_results (default: disabled).",
    )

    parser.add_argument("--reward-map-target", type=str, choices=["peg1", "peg2", "ball"], default="peg1")
    parser.add_argument("--reward-map-bins", type=int, default=80)
    parser.add_argument("--reward-map-context-samples", type=int, default=4096)
    parser.add_argument(
        "--reward-map-legal-region-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--disc-generated-samples", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parse_tags(raw: str) -> List[str]:
    tags = [t.strip() for t in str(raw).split(",") if t.strip()]
    if not tags:
        raise ValueError("No official tags specified.")
    return tags


def _resolve_eval_script(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"evaluate_gail_results.py not found: {p}")
        return p
    p = Path(__file__).resolve().parent.parent / "eval" / "evaluate_gail_results.py"
    if not p.exists():
        raise FileNotFoundError(f"evaluate_gail_results.py not found: {p}")
    return p


def _build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        out = Path(args.output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out
    root = Path(args.output_root).expanduser().resolve()
    run_id = args.run_id or f"gail_disc_analysis_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = root / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _tail_text(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    try:
        txt = path.read_text(errors="replace")
    except Exception:
        return ""
    arr = txt.splitlines()
    return "\n".join(arr[-lines:])


def _parse_run_name(name: str) -> Optional[Tuple[str, str]]:
    m = re.match(r"^klask_gail_(\d{8}_\d{6})_(.+)$", name)
    if not m:
        return None
    return m.group(1), m.group(2)


def _resolve_latest_runs(gail_runs_dir: Path, tags: List[str]) -> List[Dict[str, str]]:
    if not gail_runs_dir.exists():
        raise FileNotFoundError(f"GAIL runs directory not found: {gail_runs_dir}")
    run_dirs = [p for p in gail_runs_dir.iterdir() if p.is_dir()]
    resolved: List[Dict[str, str]] = []
    for tag in tags:
        best_ts = ""
        best_dir: Optional[Path] = None
        for d in run_dirs:
            parsed = _parse_run_name(d.name)
            if parsed is None:
                continue
            ts, run_tag = parsed
            if run_tag != tag:
                continue
            if ts > best_ts:
                best_ts = ts
                best_dir = d
        if best_dir is None:
            raise RuntimeError(f"Could not resolve latest run for tag={tag!r}.")
        resolved.append(
            {
                "run_tag": str(tag),
                "run_name": str(best_dir.name),
                "run_dir": str(best_dir.resolve()),
                "timestamp": str(best_ts),
            }
        )
    return resolved


def _build_eval_command(
    *,
    args: argparse.Namespace,
    eval_script: Path,
    run_dir: Path,
    run_out_dir: Path,
) -> List[str]:
    cmd = [
        str(args.python_bin),
        str(eval_script),
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(run_out_dir),
        "--analysis-output-dir",
        str(run_out_dir),
        "--episodes",
        str(int(args.episodes)),
        "--num-envs",
        str(int(args.num_envs)),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
        "--run-discriminator-analysis",
        "--reward-map-target",
        str(args.reward_map_target),
        "--reward-map-bins",
        str(int(args.reward_map_bins)),
        "--reward-map-context-samples",
        str(int(args.reward_map_context_samples)),
        "--reward-map-legal-region-only" if bool(args.reward_map_legal_region_only) else "--no-reward-map-legal-region-only",
        "--disc-generated-samples",
        str(int(args.disc_generated_samples)),
    ]
    if not bool(args.include_demos):
        cmd.append("--no-demos")
    return cmd


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
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


def _format_plot_label(run_tag: str) -> str:
    tag = str(run_tag)
    m = re.match(r"^GAIL_DW_EXPERT_([^_]+)_BC_BOOTSTRAP.*$", tag)
    if m:
        return f"GAIL_{m.group(1)}_BC_BOOTSTRAP"
    m = re.match(r"^GAIL_DW_EXPERT_([^_]+).*$", tag)
    if m:
        return f"GAIL_{m.group(1)}"
    return tag


def _plot_overfit_gap(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    labels = [_format_plot_label(str(r["run_tag"])) for r in rows]
    gap_expert = np.asarray([float(r["expert_positive_acc_train_minus_val"]) for r in rows], dtype=np.float64)
    gap_balanced = np.asarray([float(r["balanced_acc_total_train_minus_val"]) for r in rows], dtype=np.float64)
    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(x - w / 2, gap_expert, w, label="expert-positive acc gap")
    ax.bar(x + w / 2, gap_balanced, w, label="balanced total-acc gap")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("train - val")
    ax.set_title("Discriminator overfitting gap by dataset")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_peg1_ball_proximity_gap(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    labels = [_format_plot_label(str(r["run_tag"])) for r in rows]
    gap = np.asarray(
        [float(r["peg1_ball_proximity_near_minus_far_mean_logit"]) for r in rows],
        dtype=np.float64,
    )
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(x, gap, width=0.62, color="tab:green")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("near - far mean logit")
    ax.set_title("Peg1-ball proximity reward gap by dataset")
    ax.grid(True, axis="y", alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if int(args.episodes) <= 0:
        raise ValueError("--episodes must be > 0.")
    if int(args.num_envs) <= 0:
        raise ValueError("--num-envs must be > 0.")
    if int(args.reward_map_bins) <= 0:
        raise ValueError("--reward-map-bins must be > 0.")
    if int(args.reward_map_context_samples) <= 0:
        raise ValueError("--reward-map-context-samples must be > 0.")
    if int(args.disc_generated_samples) < 0:
        raise ValueError("--disc-generated-samples must be >= 0.")

    tags = _parse_tags(args.official_tags)
    gail_runs_dir = Path(args.gail_runs_dir).expanduser().resolve()
    eval_script = _resolve_eval_script(args.eval_script)
    isaaclab_cwd = Path(args.isaaclab_cwd).expanduser().resolve()
    if not isaaclab_cwd.exists():
        raise FileNotFoundError(f"IsaacLab cwd not found: {isaaclab_cwd}")

    output_dir = _build_output_dir(args)
    runs_dir = output_dir / "runs"
    agg_dir = output_dir / "aggregate"
    runs_dir.mkdir(parents=True, exist_ok=True)
    agg_dir.mkdir(parents=True, exist_ok=True)

    resolved_runs = _resolve_latest_runs(gail_runs_dir, tags)
    resolved_config = {
        "created_at": dt.datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "gail_runs_dir": str(gail_runs_dir),
        "eval_script": str(eval_script),
        "isaaclab_cwd": str(isaaclab_cwd),
        "official_tags": tags,
        "episodes": int(args.episodes),
        "num_envs": int(args.num_envs),
        "device": str(args.device),
        "seed": int(args.seed),
        "include_demos": bool(args.include_demos),
        "reward_map_target": str(args.reward_map_target),
        "reward_map_bins": int(args.reward_map_bins),
        "reward_map_context_samples": int(args.reward_map_context_samples),
        "reward_map_legal_region_only": bool(args.reward_map_legal_region_only),
        "disc_generated_samples": int(args.disc_generated_samples),
        "resume": bool(args.resume),
        "force": bool(args.force),
        "dry_run": bool(args.dry_run),
        "resolved_runs": resolved_runs,
    }
    _write_json(output_dir / "resolved_config.json", resolved_config)

    run_index: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(resolved_runs, start=1):
        run_name = str(item["run_name"])
        run_tag = str(item["run_tag"])
        run_dir = Path(str(item["run_dir"])).resolve()
        out = runs_dir / run_name
        out.mkdir(parents=True, exist_ok=True)
        metrics_path = out / "discriminator_metrics.json"
        summary_path = out / "analysis_summary.json"
        log_path = out / "run.log"
        entry: Dict[str, Any] = {
            "index": int(idx),
            "run_name": run_name,
            "run_tag": run_tag,
            "run_dir": str(run_dir),
            "output_dir": str(out),
            "metrics_path": str(metrics_path),
            "summary_path": str(summary_path),
            "log_path": str(log_path),
        }

        if bool(args.resume) and metrics_path.exists() and summary_path.exists() and not bool(args.force):
            print(f"[job {idx}/{len(resolved_runs)}] cached {run_name}")
            entry["status"] = "cached"
            entry["return_code"] = 0
        else:
            cmd = _build_eval_command(
                args=args,
                eval_script=eval_script,
                run_dir=run_dir,
                run_out_dir=out,
            )
            entry["command"] = cmd
            if args.dry_run:
                print(f"[job {idx}/{len(resolved_runs)}] dry-run {run_name}")
                entry["status"] = "dry_run"
                entry["return_code"] = None
            else:
                print(f"[job {idx}/{len(resolved_runs)}] run {run_name}")
                print(f"[cmd] {' '.join(cmd)}")
                with log_path.open("w") as log_fh:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(isaaclab_cwd),
                        stdout=log_fh,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                entry["return_code"] = int(proc.returncode)
                if proc.returncode != 0:
                    tail = _tail_text(log_path, lines=100)
                    raise RuntimeError(
                        f"Analysis failed for {run_name} (rc={proc.returncode}). Log: {log_path}\n"
                        f"---- log tail ----\n{tail}"
                    )
                entry["status"] = "ok"

        run_index.append(entry)
        if args.dry_run:
            continue
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing discriminator_metrics.json for {run_name}: {metrics_path}")
        metrics = _load_json(metrics_path)
        cfg = _load_json(run_dir / "config.json")
        dataset_stats = cfg.get("dataset_stats", {})
        peg1_ball_proximity_summary = dict(metrics.get("peg1_ball_proximity", {}).get("summary", {}))
        row = {
            "run_tag": run_tag,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "dataset_steps_raw": int(dataset_stats.get("n_steps_raw", 0)),
            "dataset_trajectories_raw": int(dataset_stats.get("n_traj_raw", 0)),
            "expert_positive_acc_train": float(metrics["expert_positive_accuracy"]["train"]),
            "expert_positive_acc_val": float(metrics["expert_positive_accuracy"]["val"]),
            "expert_positive_acc_train_minus_val": float(
                metrics["expert_positive_accuracy"]["train_minus_val"]
            ),
            "balanced_acc_total_train": float(metrics["balanced_metrics"]["train"]["accuracy_total"]),
            "balanced_acc_total_val": float(metrics["balanced_metrics"]["val"]["accuracy_total"]),
            "balanced_acc_total_train_minus_val": float(
                metrics["balanced_metrics"]["gaps_train_minus_val"]["accuracy_total_train_minus_val"]
            ),
            "balanced_acc_expert_train": float(metrics["balanced_metrics"]["train"]["accuracy_expert"]),
            "balanced_acc_expert_val": float(metrics["balanced_metrics"]["val"]["accuracy_expert"]),
            "balanced_acc_expert_train_minus_val": float(
                metrics["balanced_metrics"]["gaps_train_minus_val"]["accuracy_expert_train_minus_val"]
            ),
            "balanced_acc_generated_train": float(metrics["balanced_metrics"]["train"]["accuracy_generated"]),
            "balanced_acc_generated_val": float(metrics["balanced_metrics"]["val"]["accuracy_generated"]),
            "balanced_acc_generated_train_minus_val": float(
                metrics["balanced_metrics"]["gaps_train_minus_val"]["accuracy_generated_train_minus_val"]
            ),
            "peg1_ball_proximity_near_distance": float(
                peg1_ball_proximity_summary.get("near_distance", float("nan"))
            ),
            "peg1_ball_proximity_far_distance": float(
                peg1_ball_proximity_summary.get("far_distance", float("nan"))
            ),
            "peg1_ball_proximity_near_minus_far_mean_logit": float(
                peg1_ball_proximity_summary.get("near_minus_far_mean_logit", float("nan"))
            ),
            "peg1_ball_proximity_best_distance": float(
                peg1_ball_proximity_summary.get("best_distance", float("nan"))
            ),
            "peg1_ball_proximity_best_mean_logit": float(
                peg1_ball_proximity_summary.get("best_mean_logit", float("nan"))
            ),
            "peg1_ball_proximity_spearman_distance_vs_mean_logit": float(
                peg1_ball_proximity_summary.get("spearman_distance_vs_mean_logit", float("nan"))
            ),
            "peg1_ball_proximity_monotone_nonincreasing_fraction": float(
                peg1_ball_proximity_summary.get("monotone_nonincreasing_fraction", float("nan"))
            ),
            "discriminator_metrics_path": str(metrics_path),
            "analysis_summary_path": str(summary_path),
        }
        aggregate_rows.append(row)

    _write_json(output_dir / "run_index.json", run_index)
    if not args.dry_run:
        # Keep rows in tag order.
        tag_to_row = {str(r["run_tag"]): r for r in aggregate_rows}
        ordered_rows = [tag_to_row[t] for t in tags if t in tag_to_row]
        _write_csv(ordered_rows, agg_dir / "official5_metrics.csv")
        _write_json(agg_dir / "official5_metrics.json", ordered_rows)
        _plot_overfit_gap(ordered_rows, agg_dir / "overfit_gap_by_dataset.png")
        _plot_peg1_ball_proximity_gap(ordered_rows, agg_dir / "peg1_ball_proximity_gap_by_dataset.png")

    print(f"[out] {output_dir / 'resolved_config.json'}")
    print(f"[out] {output_dir / 'run_index.json'}")
    if not args.dry_run:
        print(f"[out] {agg_dir / 'official5_metrics.csv'}")
        print(f"[out] {agg_dir / 'official5_metrics.json'}")
        print(f"[out] {agg_dir / 'overfit_gap_by_dataset.png'}")
        print(f"[out] {agg_dir / 'peg1_ball_proximity_gap_by_dataset.png'}")


if __name__ == "__main__":
    main()
