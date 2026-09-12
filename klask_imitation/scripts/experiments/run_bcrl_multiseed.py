#!/usr/bin/env python3
"""BC+RL 2x2 multi-seed sweep for the CoRL #1416 camera-ready (rebuttal promise #7).

Retrains {random, BC-init} x {sparse, dense} across seeds so that Fig. 5, the two
BC+RL rows of Table 2 and every number in 5.4 carry seed-level error bars. The
published runs cannot be pooled with these: they used a +/-1.0 m/s bound, and the two
dense checkpoints no longer exist.

Pre-registered before any result is seen (see --total-timesteps default):
  * ONE step budget for all four conditions. The survivors stopped at 17.3M and 18.9M,
    which is itself a confound; 18M brackets both and is an exact multiple of the 1.5M
    eval/checkpoint cadence. 50M would ask a different question, since the BC-init
    advantage tends to shrink with longer training.
  * ONE action bound for all four conditions. This script defaults to --action-limit
    1.0, which is what produced the published Fig. 5 and Sec. 5.4; pass
    --action-limit 0.2 for the deployment-faithful bound the physical robot enforces
    (that is the default of train_GAIL_BC_sparse_RL.py itself). The two are NOT
    poolable -- the run tag records the bound (A10 / A02) so they cannot be mixed.
    The BC bootstrap action head is rescaled by 1/limit, so the BC-init arm starts at
    the same commands under either bound; at 1.0 the rescale is a no-op.
  * Opponent refresh OFF. It never fired in the published runs (zero refresh
    checkpoints), so this reproduces the effective behaviour deterministically.

Retention: every run keeps its periodic checkpoints, final policy, config.json and
per-episode evaluation CSV. Do not prune these -- the original dense results are
unrecoverable precisely because only summary.json survived.

Stages: train | eval | aggregate
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import math
import os
import queue
import statistics
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HOME = Path.home()

# Same environment contract as the shell launchers in this directory: every path is
# overridable, and nothing is hardcoded to one machine.
KLASKTRON_ROOT = Path(os.environ.get("KLASKTRON_ROOT", _REPO_ROOT))
ISAACLAB_ROOT = os.environ.get("ISAACLAB_ROOT", str(_HOME / "IsaacLab"))
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python")
CONDA_SH = os.environ.get("CONDA_SH", str(_HOME / "miniconda3/etc/profile.d/conda.sh"))
CONDA_ENV = os.environ.get("CONDA_ENV", "env_isaaclab")
ROS_SETUP = os.environ.get("ROS_SETUP", str(_HOME / "ros2_ws/install/setup.bash"))
ISAACLAB_CWD = ISAACLAB_ROOT

TRAIN_SCRIPT = str(KLASKTRON_ROOT / "klask_imitation" / "scripts" / "training" / "train_GAIL_BC_sparse_RL.py")
EVAL_SCRIPT = str(KLASKTRON_ROOT / "klask_imitation" / "scripts" / "eval" / "evaluate_gail_results.py")

# The frozen PPO reference opponent (Elo 1283) every simulated metric is scored against.
PPO_EXPERT_DIR = os.environ.get(
    "PPO_EXPERT_DIR", f"{ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one")
PPO_EXPERT_CKPT = f"{PPO_EXPERT_DIR}/klask.pth"
PPO_EXPERT_CFG = f"{PPO_EXPERT_DIR}/agent.yaml"

DEFAULT_SAVE_ROOT = str(KLASKTRON_ROOT / "trained_models" / "bcrl_2x2_multiseed")
DEFAULT_OUTPUT_ROOT = str(KLASKTRON_ROOT / "artifacts" / "bcrl_2x2_multiseed")
DEVICE_PLACEHOLDER = "__DEVICE__"

# (reward_mode, learner_init_mode, short label) -- the 2x2 of Fig. 5.
CONDITIONS: Tuple[Tuple[str, str, str], ...] = (
    ("sparse", "random", "sparse_random"),
    ("sparse", "bootstrap", "sparse_bc"),
    ("dense", "random", "dense_random"),
    ("dense", "bootstrap", "dense_bc"),
)


@dataclasses.dataclass
class Job:
    reward_mode: str
    init_mode: str
    label: str
    seed: int
    action_limit: float = 1.0

    @property
    def _limit_tag(self) -> str:
        # e.g. 1.0 -> A10, 0.2 -> A02. Without this, runs at different action limits
        # share a tag and --resume silently treats them as the same experiment.
        return "A%02d" % int(round(float(self.action_limit) * 10))

    @property
    def job_id(self) -> str:
        return f"{self.label}_seed{self.seed}"

    @property
    def run_tag(self) -> str:
        return f"BCRL_2X2_{self._limit_tag}_{self.label.upper()}_SEED{self.seed}"


def _iso_now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def wrap_in_env(cmd: Sequence[str]) -> List[str]:
    inner = " ".join(f"'{c}'" if " " in c else c for c in cmd)
    script = (
        f"source {CONDA_SH} && conda activate {CONDA_ENV} && "
        f'export COLCON_TRACE="${{COLCON_TRACE:-0}}" && '
        f'export AMENT_TRACE_SETUP_FILES="${{AMENT_TRACE_SETUP_FILES:-}}" && '
        f"set +u && source {ROS_SETUP} && set -u && exec {inner}"
    )
    return ["bash", "-c", script]


def run_jobs(tasks: List[Tuple[str, List[str]]], *, devices: List[str], log_dir: Path,
             dry_run: bool) -> Dict[str, int]:
    """Workers pull from one queue and substitute their own device into the command."""
    log_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        for i, (jid, cmd) in enumerate(tasks):
            print(f"[dry-run] {jid} (e.g. {devices[i % len(devices)]})\n    "
                  + " ".join(c.replace(DEVICE_PLACEHOLDER, devices[i % len(devices)]) for c in cmd) + "\n")
        return {}

    pending: "queue.Queue[Tuple[str, List[str]]]" = queue.Queue()
    for t in tasks:
        pending.put(t)
    results: Dict[str, int] = {}
    lock = threading.Lock()

    def worker(device: str) -> None:
        while True:
            try:
                jid, template = pending.get_nowait()
            except queue.Empty:
                return
            cmd = [c.replace(DEVICE_PLACEHOLDER, device) for c in template]
            log_path = log_dir / f"{jid}.log"
            with lock:
                print(f"[{device}] start {jid} ({_iso_now()}) -> {log_path}")
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(f"# job={jid} device={device} started={_iso_now()}\n# {' '.join(cmd)}\n\n")
                fh.flush()
                rc = subprocess.run(cmd, cwd=ISAACLAB_CWD, stdout=fh, stderr=subprocess.STDOUT).returncode
            with lock:
                results[jid] = int(rc)
                print(f"[{device}] done  {jid} ({_iso_now()}) "
                      + ("ok" if rc == 0 else f"FAILED rc={rc}"))

    threads = [threading.Thread(target=worker, args=(d,), daemon=True) for d in devices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


# config.json is written BEFORE model.learn() (trainer line 1615 vs 1642), so it
# marks "started", not "finished". The final policy is the only completion marker.
COMPLETION_MARKER = "ppo_policy_sparse_rl_final.zip"


def find_run_dir(save_root: Path, run_tag: str, *, require_complete: bool = True) -> Optional[Path]:
    """Latest run directory carrying this exact tag; completed ones unless told otherwise."""
    if not save_root.is_dir():
        return None
    best: Optional[Tuple[str, Path]] = None
    for d in save_root.iterdir():
        if not d.is_dir() or run_tag not in d.name:
            continue
        marker = COMPLETION_MARKER if require_complete else "config.json"
        if not (d / marker).is_file():
            continue
        if best is None or d.name > best[0]:
            best = (d.name, d)
    return None if best is None else best[1]


def build_jobs(seeds: Sequence[int], labels: Optional[Sequence[str]],
               action_limit: float) -> List[Job]:
    jobs = []
    for reward_mode, init_mode, label in CONDITIONS:
        if labels and label not in labels:
            continue
        for s in seeds:
            jobs.append(Job(reward_mode, init_mode, label, int(s), float(action_limit)))
    return jobs


def stage_train(args, jobs: List[Job], out_dir: Path) -> None:
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    devices = [f"cuda:{g.strip()}" for g in args.gpus.split(",") if g.strip()]

    tasks, skipped = [], []
    for job in jobs:
        if args.resume and find_run_dir(save_root, job.run_tag) is not None:
            skipped.append(job.job_id)
            continue
        cmd = [
            PYTHON_BIN, TRAIN_SCRIPT,
            "--save-root", str(save_root),
            "--run-tag", job.run_tag,
            "--reward-mode", job.reward_mode,
            "--learner-init-mode", job.init_mode,
            "--seed", str(job.seed),
            "--action-limit", str(args.action_limit),
            "--no-enable-opponent-refresh",
            "--total-timesteps", str(int(args.total_timesteps)),
            "--train-num-envs", str(int(args.train_num_envs)),
            "--device", DEVICE_PLACEHOLDER,
            "--wandb-mode", args.wandb_mode,
        ]
        tasks.append((job.job_id, wrap_in_env(cmd)))

    if skipped:
        print(f"[train] resume: skipping {len(skipped)} done: {', '.join(skipped)}")
    if not tasks:
        print("[train] nothing to do.")
        return
    print(f"[train] {len(tasks)} run(s) x {int(args.total_timesteps):,} steps "
          f"over {len(devices)} GPU(s): {', '.join(devices)}")
    run_jobs(tasks, devices=devices, log_dir=out_dir / "logs" / "train", dry_run=args.dry_run)


def stage_eval(args, jobs: List[Job], out_dir: Path) -> None:
    save_root = Path(args.save_root)
    eval_root = out_dir / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    devices = [f"cuda:{g.strip()}" for g in args.gpus.split(",") if g.strip()]

    tasks = []
    for job in jobs:
        run_dir = find_run_dir(save_root, job.run_tag)
        if run_dir is None:
            print(f"[eval] no trained run for {job.job_id}; skipping.")
            continue
        policy = run_dir / "ppo_policy_sparse_rl_final.zip"
        if not policy.is_file():
            print(f"[eval] missing final policy for {job.job_id}; skipping.")
            continue
        odir = eval_root / job.job_id
        if args.resume and (odir / "summary.json").is_file():
            continue
        cmd = [
            PYTHON_BIN, EVAL_SCRIPT,
            "--run-dir", str(run_dir),
            "--policy-path", str(policy),
            "--output-dir", str(odir),
            "--episodes", str(int(args.eval_episodes)),
            "--num-envs", str(int(args.eval_num_envs)),
            "--device", DEVICE_PLACEHOLDER,
            "--seed", str(int(args.eval_seed)),
            # Promise #14: every simulated closed-loop number against the frozen PPO.
            "--peg1-only-fixed-opponent",
            "--opponent-policy-path", PPO_EXPERT_CKPT,
            "--opponent-policy-format", "rl_games",
            "--opponent-rl-games-config-path", PPO_EXPERT_CFG,
            "--deterministic-opponent",
            "--no-reward-net", "--no-demos",
            "--save-eval-outputs",   # writes episode_metrics.csv -- keep it
        ]
        tasks.append((job.job_id, wrap_in_env(cmd)))

    if not tasks:
        print("[eval] nothing to do.")
        return
    print(f"[eval] {len(tasks)} evaluation(s) vs frozen PPO.")
    run_jobs(tasks, devices=devices, log_dir=out_dir / "logs" / "eval", dry_run=args.dry_run)


_T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
           8: 2.306, 9: 2.262, 10: 2.228}


def _mean_ci95(v: Sequence[float]) -> Tuple[float, float, float, float]:
    n = len(v)
    m = statistics.fmean(v)
    if n < 2:
        return m, 0.0, m, m
    sd = statistics.stdev(v)
    half = _T_CRIT.get(n - 1, 1.96) * sd / math.sqrt(n)
    return m, sd, m - half, m + half


METRICS = ("goals_scored_per_episode", "goals_conceded_per_episode",
           "term_rate_player_in_goal", "term_rate_time_out", "mean_episode_length")


def stage_aggregate(args, jobs: List[Job], out_dir: Path) -> None:
    eval_root = out_dir / "eval"
    rows: List[Dict[str, Any]] = []
    for job in jobs:
        p = eval_root / job.job_id / "summary.json"
        if not p.is_file():
            continue
        s = json.loads(p.read_text())
        r: Dict[str, Any] = {"condition": job.label, "reward_mode": job.reward_mode,
                             "init_mode": job.init_mode, "seed": job.seed}
        for k in METRICS:
            r[k] = s.get(k)
        g, c = s.get("goals_scored_per_episode"), s.get("goals_conceded_per_episode")
        if isinstance(g, (int, float)) and isinstance(c, (int, float)):
            r["goal_diff_per_episode"] = float(g) - float(c)
        rows.append(r)

    if not rows:
        print("[aggregate] no evaluation summaries found.")
        return

    csv_path = out_dir / "results_per_run.csv"
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    metric_keys = [k for k in fields if k not in {"condition", "reward_mode", "init_mode", "seed"}]
    agg: Dict[str, Dict[str, Any]] = {}
    for _, _, label in CONDITIONS:
        sel = [r for r in rows if r["condition"] == label]
        if not sel:
            continue
        e: Dict[str, Any] = {"n_seeds": len(sel), "seeds": sorted(r["seed"] for r in sel)}
        for k in metric_keys:
            vals = [r[k] for r in sel if isinstance(r.get(k), (int, float))]
            if vals:
                m, sd, lo, hi = _mean_ci95(vals)
                e[k] = {"mean": m, "sd": sd, "ci95_low": lo, "ci95_high": hi,
                        "n": len(vals), "values": vals}
        agg[label] = e

    # The headline of 5.4 is the BC-prior effect within each reward regime.
    contrasts: Dict[str, Any] = {}
    for reward in ("sparse", "dense"):
        bc = [r for r in rows if r["condition"] == f"{reward}_bc"]
        rd = [r for r in rows if r["condition"] == f"{reward}_random"]
        if not bc or not rd:
            continue
        c: Dict[str, Any] = {}
        for k in metric_keys:
            a = [r[k] for r in bc if isinstance(r.get(k), (int, float))]
            b = [r[k] for r in rd if isinstance(r.get(k), (int, float))]
            if len(a) < 2 or len(b) < 2:
                continue
            diff = statistics.fmean(a) - statistics.fmean(b)
            se = math.sqrt(statistics.stdev(a) ** 2 / len(a) + statistics.stdev(b) ** 2 / len(b))
            t = _T_CRIT.get(min(len(a), len(b)) - 1, 1.96)
            lo, hi = diff - t * se, diff + t * se
            c[k] = {"diff_bc_minus_random": diff, "ci95_low": lo, "ci95_high": hi,
                    "excludes_zero": bool(lo > 0 or hi < 0)}
        contrasts[reward] = c

    out_json = out_dir / "results_aggregated.json"
    out_json.write_text(json.dumps(
        {"created_at": _iso_now(), "per_run": rows, "by_condition": agg,
         "bc_prior_contrasts": contrasts}, indent=2))

    print(f"[aggregate] {len(rows)} run rows -> {csv_path}")
    print(f"[aggregate] seed-level summary -> {out_json}\n")
    for label, e in agg.items():
        gd = e.get("goal_diff_per_episode")
        pig = e.get("term_rate_player_in_goal")
        line = f"  {label:<14s} n={e['n_seeds']}"
        if gd:
            line += f"  goal-diff {gd['mean']:+.4f} [{gd['ci95_low']:+.4f}, {gd['ci95_high']:+.4f}]"
        if pig:
            line += f"  PiG {pig['mean']:.3f} [{pig['ci95_low']:.3f}, {pig['ci95_high']:.3f}]"
        print(line)
    print("\n  BC prior effect (BC-init minus random), 95% CI on the difference:")
    for reward, c in contrasts.items():
        for k in ("goal_diff_per_episode", "term_rate_player_in_goal"):
            if k in c:
                d = c[k]
                verdict = "EXCLUDES 0" if d["excludes_zero"] else "includes 0"
                print(f"    {reward:6s} {k:<26s} {d['diff_bc_minus_random']:+.4f} "
                      f"[{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}]  {verdict}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="train", choices=["train", "eval", "aggregate", "all"])
    p.add_argument("--seeds", default="42,1,2,3,4")
    p.add_argument("--conditions", default=None,
                   help="Comma-separated subset of: " + ",".join(c[2] for c in CONDITIONS))
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--total-timesteps", type=int, default=18_000_000,
                   help="Pre-registered, identical for all conditions (default: %(default)s).")
    p.add_argument("--train-num-envs", type=int, default=4096)
    p.add_argument("--action-limit", type=float, default=1.0,
                   help="Per-axis command bound. 1.0 reproduces the published 2x2; "
                        "at 0.2 sparse RL does not learn within 18M steps (see archive).")
    p.add_argument("--eval-episodes", type=int, default=1000)
    p.add_argument("--eval-num-envs", type=int, default=300)
    p.add_argument("--eval-seed", type=int, default=42)
    p.add_argument("--save-root", default=DEFAULT_SAVE_ROOT)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-id", default=None)
    p.add_argument("--wandb-mode", default="disabled")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    labels = [s.strip() for s in args.conditions.split(",")] if args.conditions else None
    jobs = build_jobs(seeds, labels, float(args.action_limit))
    if not jobs:
        raise SystemExit("No jobs resolved.")

    for path, what in ((PPO_EXPERT_CKPT, "expert checkpoint"), (TRAIN_SCRIPT, "trainer"),
                       (EVAL_SCRIPT, "evaluator")):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {what}: {path}")

    run_id = args.run_id or f"camera_ready_{dt.datetime.now().strftime('%Y%m%d')}"
    out_dir = Path(args.output_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "provenance.json").write_text(json.dumps({
        "created_at": _iso_now(), "run_id": run_id, "seeds": seeds,
        "conditions": [c[2] for c in CONDITIONS],
        "total_timesteps": int(args.total_timesteps),
        "action_limit": float(args.action_limit),
        "opponent_refresh": False,
        "bootstrap_head_scale": 1.0 / float(args.action_limit),
        "eval_opponent": PPO_EXPERT_CKPT,
        "note": "Published runs used action_limit 1.0 and are NOT poolable with these.",
        "jobs": [dataclasses.asdict(j) for j in jobs],
    }, indent=2))

    print(f"[setup] {len(jobs)} job(s) = {len(set(j.label for j in jobs))} condition(s) "
          f"x {len(seeds)} seed(s), {int(args.total_timesteps):,} steps each")
    print(f"[setup] output: {out_dir}")

    stages = ["train", "eval", "aggregate"] if args.stage == "all" else [args.stage]
    for s in stages:
        {"train": stage_train, "eval": stage_eval, "aggregate": stage_aggregate}[s](args, jobs, out_dir)


if __name__ == "__main__":
    main()
