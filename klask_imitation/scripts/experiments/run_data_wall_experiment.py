#!/usr/bin/env python3
"""Run BC data-wall policy evaluation jobs (no training/retraining).

This script orchestrates matchup evaluations using `evaluate_policy_matchup.py`
and writes:
  - one `summary.json` per matchup job
  - a top-level `jobs.json` index
  - a flat `results_raw.csv` table for downstream aggregation
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fnmatch
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_TASK = "Isaac-Klask-v0"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAINED_MODELS_DIR = Path(
    os.environ.get("KLASKTRON_TRAINED_MODELS_DIR", _REPO_ROOT / "trained_models")
)
_ARTIFACTS_DIR = Path(
    os.environ.get("KLASKTRON_ARTIFACTS_DIR", _REPO_ROOT / "artifacts")
)
DEFAULT_BC_RUNS_DIR = str(_TRAINED_MODELS_DIR / "data_wall_experiments" / "bc_runs")
DEFAULT_OUTPUT_ROOT = str(_ARTIFACTS_DIR / "data_wall" / "eval_runs")
# Path to the Isaac Lab checkout that hosts the RL-Games expert checkpoints.
# Override with the ISAACLAB_ROOT environment variable.
_ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_ROOT", Path.home() / "IsaacLab"))
_EXPERT_AGENT_DIR = _ISAACLAB_ROOT / "logs/rl_games/klask/demo_agents/best_one"
DEFAULT_ISAACLAB_CWD = str(_ISAACLAB_ROOT)
DEFAULT_EXPERT_CHECKPOINT = str(_EXPERT_AGENT_DIR / "klask.pth")
DEFAULT_EXPERT_AGENT_CFG = str(_EXPERT_AGENT_DIR / "agent.yaml")
DEFAULT_MATCHUPS = "bc_vs_expert,bc_vs_bc,expert_vs_bc,expert_vs_expert"
DEFAULT_POLICY_SELECTION = "official"
DEFAULT_OFFICIAL_DW_DATASET_IDS = (
    "DW_EXPERT_80000",
    "DW_EXPERT_250000",
    "DW_EXPERT_1000000",
    "DW_EXPERT_5000000",
)
# Matches W&B run "klask_BC_Human (klask_bc_loss_20260301_202918)".
DEFAULT_OFFICIAL_HUMAN_RUN_NAME = "klask_bc_20260303_161251_HUMAN_DEMO_V2"

POLICY_FORMAT_SB3 = "sb3_ppo"
POLICY_FORMAT_RL_GAMES = "rl_games"
VALID_MATCHUPS = {
    "bc_vs_expert",
    "bc_vs_bc",
    "expert_vs_bc",
    "expert_vs_expert",
    "all_vs_all",
}


@dataclass(frozen=True)
class BcPolicy:
    policy_id: str
    checkpoint: str
    run_dir: str
    dataset_id: str
    dataset_kind: str
    dataset_steps: Optional[int]
    dataset_steps_used_train: Optional[int]
    train_raw_steps: Optional[int]


@dataclass(frozen=True)
class EvalJob:
    job_id: str
    matchup: str
    output_dir: str
    policy_id: str
    policy_checkpoint: str
    policy_format: str
    opponent_id: Optional[str]
    opponent_checkpoint: Optional[str]
    opponent_format: Optional[str]
    force_self_play: Optional[bool]
    rl_games_config_path: Optional[str]
    opponent_rl_games_config_path: Optional[str]
    bc_reference: Optional[BcPolicy]


@dataclass(frozen=True)
class PolicyAgent:
    agent_id: str
    checkpoint: str
    policy_format: str
    rl_games_config_path: Optional[str]
    bc_reference: Optional[BcPolicy]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate already-trained BC policies in policy-vs-policy Klask matchups "
            "(BC vs PPO expert / BC self-play / PPO expert vs BC / PPO expert vs PPO expert)."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config (overrides defaults).")
    parser.add_argument("--bc-runs-dir", type=str, default=DEFAULT_BC_RUNS_DIR)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=str, default=None, help="Explicit output directory (overrides output-root/run-id).")
    parser.add_argument("--run-id", type=str, default=None, help="Run name when output-dir is not provided.")
    parser.add_argument("--matchups", type=str, default=DEFAULT_MATCHUPS, help="Comma-separated list.")
    parser.add_argument(
        "--policy-selection",
        type=str,
        choices=["official", "all"],
        default=DEFAULT_POLICY_SELECTION,
        help="BC policy set selection. 'official' uses fixed DW+human runs; 'all' scans all BC runs.",
    )
    parser.add_argument("--include", action="append", default=[], help="Glob pattern(s) for BC run dir names to include.")
    parser.add_argument("--exclude", action="append", default=[], help="Glob pattern(s) for BC run dir names to exclude.")
    parser.add_argument("--bc-checkpoint", action="append", default=[], help="Explicit BC checkpoint path(s).")
    parser.add_argument("--max-policies", type=int, default=None, help="Optional cap for quick smoke runs.")
    parser.add_argument("--human-baseline-steps", type=int, default=81455)
    parser.add_argument(
        "--official-dw-dataset-id",
        action="append",
        default=[],
        help="Override/add official DW dataset id(s), e.g. DW_EXPERT_1000000.",
    )
    parser.add_argument(
        "--official-human-run-name",
        type=str,
        default=DEFAULT_OFFICIAL_HUMAN_RUN_NAME,
        help="Official BC human run directory name.",
    )

    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--eval-script", type=str, default=None, help="Path to evaluate_policy_matchup.py")
    parser.add_argument("--isaaclab-cwd", type=str, default=DEFAULT_ISAACLAB_CWD)

    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--num-envs", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--deterministic-opponent",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--summary-min-episode-length", type=int, default=3)
    parser.add_argument("--log-every-episodes", type=int, default=100)
    parser.add_argument(
        "--reward-profile",
        type=str,
        default="sparse_competitive",
        choices=["sparse_competitive", "sparse_plus_collision", "dense_default", "zero"],
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-fabric", action="store_true", default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)

    parser.add_argument("--expert-policy-path", type=str, default=DEFAULT_EXPERT_CHECKPOINT)
    parser.add_argument("--expert-rl-games-config-path", type=str, default=DEFAULT_EXPERT_AGENT_CFG)
    parser.add_argument("--expert-task", type=str, default=DEFAULT_TASK)
    return parser.parse_args()


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        if math.isnan(out):
            return float("nan")
        return out
    except Exception:
        return float("nan")


def _parse_scaled_number(token: str) -> Optional[int]:
    m = re.fullmatch(r"(?i)(\d+(?:\.\d+)?)([km]?)", token.strip())
    if not m:
        return None
    base = float(m.group(1))
    suffix = m.group(2).lower()
    if suffix == "k":
        base *= 1_000.0
    elif suffix == "m":
        base *= 1_000_000.0
    return int(round(base))


def _parse_dataset_steps_from_name(name: str) -> Optional[int]:
    patterns = [
        r"(?i)DW_[A-Z0-9]+_(\d+(?:\.\d+)?[kKmM]?)(?:_|$)",
        r"(?i)_(\d+(?:\.\d+)?[kKmM]?)(?:_|$)",
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, name):
            parsed = _parse_scaled_number(m.group(1))
            if parsed is not None:
                return parsed
    return None


def _dataset_id_from_run_name(run_name: str) -> str:
    m = re.match(r"^klask_bc_\d{8}_\d{6}_(.+)$", run_name)
    if m:
        return m.group(1)
    return run_name


def _load_train_stats(run_dir: Path) -> Dict[str, Any]:
    stats_path = run_dir / "train_stats.json"
    if not stats_path.exists():
        return {}
    with stats_path.open("r") as fh:
        return json.load(fh)


def _match_any(name: str, patterns: Iterable[str]) -> bool:
    pats = [p for p in patterns if p]
    if not pats:
        return True
    return any(fnmatch.fnmatch(name, p) for p in pats)


def _discover_bc_checkpoints(
    *,
    bc_runs_dir: Path,
    explicit_checkpoints: List[str],
    policy_selection: str,
    official_dw_dataset_ids: List[str],
    official_human_run_name: str,
    include_patterns: List[str],
    exclude_patterns: List[str],
    human_baseline_steps: int,
    max_policies: Optional[int],
) -> List[BcPolicy]:
    candidates: List[Path] = []
    if explicit_checkpoints:
        for ckpt in explicit_checkpoints:
            p = Path(ckpt).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(f"BC checkpoint not found: {p}")
            candidates.append(p)
    else:
        if not bc_runs_dir.exists():
            raise FileNotFoundError(f"BC runs directory not found: {bc_runs_dir}")
        candidates = sorted(bc_runs_dir.glob("*/ppo_policy_bc_final.zip"))
        if policy_selection == "official":
            dw_targets = [x for x in official_dw_dataset_ids if x]
            if not dw_targets:
                raise ValueError("Official selection requires at least one DW dataset id.")

            dw_hits: Dict[str, List[Path]] = {ds: [] for ds in dw_targets}
            human_hits: List[Path] = []
            for ckpt in candidates:
                run_name = ckpt.parent.name
                ds_id = _dataset_id_from_run_name(run_name)
                if ds_id in dw_hits:
                    dw_hits[ds_id].append(ckpt)
                if run_name == official_human_run_name:
                    human_hits.append(ckpt)

            missing: List[str] = []
            duplicate: List[str] = []
            selected: List[Path] = []
            for ds_id in dw_targets:
                hits = dw_hits[ds_id]
                if not hits:
                    missing.append(ds_id)
                    continue
                if len(hits) > 1:
                    duplicate.append(
                        f"{ds_id}: {[str(h.parent.name) for h in hits]}"
                    )
                    continue
                selected.append(hits[0])

            if not human_hits:
                missing.append(official_human_run_name)
            elif len(human_hits) > 1:
                duplicate.append(
                    f"{official_human_run_name}: {[str(h.parent.name) for h in human_hits]}"
                )
            else:
                selected.append(human_hits[0])

            if missing or duplicate:
                msgs: List[str] = []
                if missing:
                    msgs.append(f"missing official run(s): {missing}")
                if duplicate:
                    msgs.append(f"duplicate official run(s): {duplicate}")
                raise RuntimeError(
                    "Official BC policy selection failed: " + "; ".join(msgs)
                )
            candidates = selected

    policies: List[BcPolicy] = []
    for ckpt in candidates:
        run_dir = ckpt.parent
        run_name = run_dir.name
        if include_patterns and not _match_any(run_name, include_patterns):
            continue
        if exclude_patterns and _match_any(run_name, exclude_patterns):
            continue

        stats = _load_train_stats(run_dir)
        dataset_id = _dataset_id_from_run_name(run_name)
        upper = run_name.upper()
        dataset_kind = "human" if "HUMAN" in upper else "synthetic"
        parsed_steps = _parse_dataset_steps_from_name(dataset_id)
        if dataset_kind == "human":
            dataset_steps = int(human_baseline_steps) if int(human_baseline_steps) > 0 else parsed_steps
        else:
            dataset_steps = parsed_steps
        if dataset_steps is None:
            dataset_steps = _safe_int(stats.get("dataset_steps_used"))

        policy = BcPolicy(
            policy_id=run_name,
            checkpoint=str(ckpt.resolve()),
            run_dir=str(run_dir.resolve()),
            dataset_id=dataset_id,
            dataset_kind=dataset_kind,
            dataset_steps=dataset_steps,
            dataset_steps_used_train=_safe_int(stats.get("dataset_steps_used")),
            train_raw_steps=_safe_int(stats.get("train_raw_steps")),
        )
        policies.append(policy)

    policies.sort(key=lambda p: (p.dataset_steps if p.dataset_steps is not None else 10**18, p.policy_id))
    if max_policies is not None and max_policies > 0:
        policies = policies[: int(max_policies)]
    if not policies:
        raise RuntimeError("No BC checkpoints discovered after applying filters.")
    return policies


def _apply_bool_flag(cmd: List[str], flag: str, value: bool) -> None:
    if value:
        cmd.append(flag)
    else:
        cmd.append(f"--no-{flag.lstrip('-')}")


def _resolve_eval_script(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"evaluate script not found: {p}")
        return p
    p = Path(__file__).resolve().parent / "evaluate_policy_matchup.py"
    if not p.exists():
        raise FileNotFoundError(f"evaluate script not found: {p}")
    return p


def _build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        out = Path(args.output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out
    root = Path(args.output_root).expanduser().resolve()
    run_id = args.run_id or f"bc_eval_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = root / "eval_runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_jobs(
    *,
    matchups: List[str],
    bc_policies: List[BcPolicy],
    expert_policy_path: str,
    expert_rl_games_config_path: str,
    expert_task: str,
    output_dir: Path,
) -> List[EvalJob]:
    jobs: List[EvalJob] = []
    eval_root = output_dir / "matchups"

    def _all_agents() -> List[PolicyAgent]:
        agents: List[PolicyAgent] = []
        for bc in bc_policies:
            agents.append(
                PolicyAgent(
                    agent_id=bc.policy_id,
                    checkpoint=bc.checkpoint,
                    policy_format=POLICY_FORMAT_SB3,
                    rl_games_config_path=expert_rl_games_config_path,
                    bc_reference=bc,
                )
            )
        agents.append(
            PolicyAgent(
                agent_id="ppo_expert",
                checkpoint=expert_policy_path,
                policy_format=POLICY_FORMAT_RL_GAMES,
                rl_games_config_path=expert_rl_games_config_path,
                bc_reference=None,
            )
        )
        return agents

    for matchup in matchups:
        if matchup == "bc_vs_expert":
            for bc in bc_policies:
                out = eval_root / matchup / bc.policy_id
                jobs.append(
                    EvalJob(
                        job_id=f"{matchup}__{bc.policy_id}",
                        matchup=matchup,
                        output_dir=str(out),
                        policy_id=bc.policy_id,
                        policy_checkpoint=bc.checkpoint,
                        policy_format=POLICY_FORMAT_SB3,
                        opponent_id="ppo_expert",
                        opponent_checkpoint=expert_policy_path,
                        opponent_format=POLICY_FORMAT_RL_GAMES,
                        force_self_play=True,
                        rl_games_config_path=expert_rl_games_config_path,
                        opponent_rl_games_config_path=expert_rl_games_config_path,
                        bc_reference=bc,
                    )
                )
        elif matchup == "bc_vs_bc":
            for bc in bc_policies:
                out = eval_root / matchup / bc.policy_id
                jobs.append(
                    EvalJob(
                        job_id=f"{matchup}__{bc.policy_id}",
                        matchup=matchup,
                        output_dir=str(out),
                        policy_id=bc.policy_id,
                        policy_checkpoint=bc.checkpoint,
                        policy_format=POLICY_FORMAT_SB3,
                        opponent_id=bc.policy_id,
                        opponent_checkpoint=None,
                        opponent_format=None,
                        force_self_play=True,
                        rl_games_config_path=expert_rl_games_config_path,
                        opponent_rl_games_config_path=None,
                        bc_reference=bc,
                    )
                )
        elif matchup == "expert_vs_bc":
            for bc in bc_policies:
                out = eval_root / matchup / bc.policy_id
                jobs.append(
                    EvalJob(
                        job_id=f"{matchup}__{bc.policy_id}",
                        matchup=matchup,
                        output_dir=str(out),
                        policy_id="ppo_expert",
                        policy_checkpoint=expert_policy_path,
                        policy_format=POLICY_FORMAT_RL_GAMES,
                        opponent_id=bc.policy_id,
                        opponent_checkpoint=bc.checkpoint,
                        opponent_format=POLICY_FORMAT_SB3,
                        force_self_play=True,
                        rl_games_config_path=expert_rl_games_config_path,
                        opponent_rl_games_config_path=expert_rl_games_config_path,
                        bc_reference=bc,
                    )
                )
        elif matchup == "expert_vs_expert":
            out = eval_root / matchup / "ppo_expert"
            jobs.append(
                EvalJob(
                    job_id="expert_vs_expert__ppo_expert",
                    matchup=matchup,
                    output_dir=str(out),
                    policy_id="ppo_expert",
                    policy_checkpoint=expert_policy_path,
                    policy_format=POLICY_FORMAT_RL_GAMES,
                    opponent_id="ppo_expert",
                    opponent_checkpoint=expert_policy_path,
                    opponent_format=POLICY_FORMAT_RL_GAMES,
                    force_self_play=True,
                    rl_games_config_path=expert_rl_games_config_path,
                    opponent_rl_games_config_path=expert_rl_games_config_path,
                        bc_reference=None,
                    )
                )
        elif matchup == "all_vs_all":
            agents = _all_agents()
            for player in agents:
                for opponent in agents:
                    out = eval_root / matchup / player.agent_id / f"vs_{opponent.agent_id}"
                    player_is_bc = player.bc_reference is not None
                    opponent_is_bc = opponent.bc_reference is not None
                    bc_ref = (
                        player.bc_reference
                        if player_is_bc
                        else (opponent.bc_reference if opponent_is_bc else None)
                    )
                    jobs.append(
                        EvalJob(
                            job_id=f"all_vs_all__{player.agent_id}__vs__{opponent.agent_id}",
                            matchup=matchup,
                            output_dir=str(out),
                            policy_id=player.agent_id,
                            policy_checkpoint=player.checkpoint,
                            policy_format=player.policy_format,
                            opponent_id=opponent.agent_id,
                            opponent_checkpoint=opponent.checkpoint,
                            opponent_format=opponent.policy_format,
                            force_self_play=True,
                            rl_games_config_path=player.rl_games_config_path,
                            opponent_rl_games_config_path=(
                                opponent.rl_games_config_path
                                if opponent.policy_format == POLICY_FORMAT_RL_GAMES
                                else None
                            ),
                            bc_reference=bc_ref,
                        )
                    )
        else:
            raise ValueError(f"Unsupported matchup: {matchup}")

    # Ensure expert task path is kept in config-resolved for reproducibility.
    _ = expert_task
    # De-duplicate by job_id while preserving insertion order.
    dedup: Dict[str, EvalJob] = {}
    for job in jobs:
        dedup[job.job_id] = job
    return list(dedup.values())


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def _build_eval_command(
    *,
    args: argparse.Namespace,
    eval_script: Path,
    job: EvalJob,
) -> List[str]:
    cmd = [
        args.python_bin,
        str(eval_script),
        "--task",
        str(args.task),
        "--episodes",
        str(int(args.episodes)),
        "--num_envs",
        str(int(args.num_envs)),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
        "--summary-min-episode-length",
        str(int(args.summary_min_episode_length)),
        "--log-every-episodes",
        str(int(args.log_every_episodes)),
        "--reward-profile",
        str(args.reward_profile),
        "--output-dir",
        str(job.output_dir),
        "--matchup-id",
        str(job.job_id),
        "--checkpoint",
        str(job.policy_checkpoint),
        "--policy-format",
        str(job.policy_format),
    ]

    _apply_bool_flag(cmd, "--deterministic-policy", bool(args.deterministic_policy))
    _apply_bool_flag(cmd, "--deterministic-opponent", bool(args.deterministic_opponent))
    _apply_bool_flag(cmd, "--headless", bool(args.headless))

    if args.disable_fabric:
        cmd.append("--disable_fabric")
    if args.force:
        cmd.append("--force")
    if job.force_self_play is not None:
        _apply_bool_flag(cmd, "--force-self-play", bool(job.force_self_play))
    if job.rl_games_config_path:
        cmd.extend(["--rl-games-config-path", str(job.rl_games_config_path)])
    if job.opponent_checkpoint:
        cmd.extend(["--opponent-policy-path", str(job.opponent_checkpoint)])
    if job.opponent_format:
        cmd.extend(["--opponent-policy-format", str(job.opponent_format)])
    if job.opponent_rl_games_config_path:
        cmd.extend(["--opponent-rl-games-config-path", str(job.opponent_rl_games_config_path)])
    return cmd


def _tail_text(path: Path, lines: int = 60) -> str:
    if not path.exists():
        return ""
    try:
        txt = path.read_text(errors="replace")
    except Exception:
        return ""
    arr = txt.splitlines()
    return "\n".join(arr[-lines:])


def _run_eval_job(
    *,
    args: argparse.Namespace,
    eval_script: Path,
    isaaclab_cwd: Path,
    job: EvalJob,
) -> Dict[str, Any]:
    out_dir = Path(job.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    log_path = out_dir / "run.log"

    if args.resume and summary_path.exists() and not args.force:
        summary = _load_json(summary_path)
        return {
            "job_id": job.job_id,
            "matchup": job.matchup,
            "status": "cached",
            "summary_path": str(summary_path),
            "log_path": str(log_path),
            "return_code": 0,
            "summary": summary,
        }

    cmd = _build_eval_command(args=args, eval_script=eval_script, job=job)
    if args.dry_run:
        return {
            "job_id": job.job_id,
            "matchup": job.matchup,
            "status": "dry_run",
            "summary_path": str(summary_path),
            "log_path": str(log_path),
            "return_code": None,
            "command": cmd,
            "summary": None,
        }

    print(f"[run] {job.job_id}")
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
    if proc.returncode != 0:
        tail = _tail_text(log_path, lines=80)
        raise RuntimeError(
            f"Evaluation failed for {job.job_id} (rc={proc.returncode}). Log: {log_path}\n"
            f"---- log tail ----\n{tail}"
        )
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary output for {job.job_id}: {summary_path}")

    summary = _load_json(summary_path)
    return {
        "job_id": job.job_id,
        "matchup": job.matchup,
        "status": "ok",
        "summary_path": str(summary_path),
        "log_path": str(log_path),
        "return_code": int(proc.returncode),
        "summary": summary,
    }


def _pick_algorithm(job: EvalJob) -> str:
    if job.matchup == "all_vs_all":
        return "BC" if job.policy_format == POLICY_FORMAT_SB3 else "PPOExpert"
    if job.matchup.startswith("bc_"):
        return "BC"
    if job.matchup.startswith("expert_"):
        return "PPOExpert"
    return "unknown"


def _job_row(job: EvalJob, result: Dict[str, Any]) -> Dict[str, Any]:
    summary = result.get("summary") or {}
    bc = job.bc_reference
    row: Dict[str, Any] = {
        "job_id": job.job_id,
        "matchup": job.matchup,
        "algorithm": _pick_algorithm(job),
        "policy_id": job.policy_id,
        "opponent_id": job.opponent_id or "",
        "policy_checkpoint": job.policy_checkpoint,
        "opponent_checkpoint": job.opponent_checkpoint or "",
        "policy_format": job.policy_format,
        "opponent_policy_format": job.opponent_format or "",
        "dataset_id": bc.dataset_id if bc else "ppo_expert",
        "dataset_kind": bc.dataset_kind if bc else "synthetic",
        "dataset_steps": int(bc.dataset_steps) if (bc and bc.dataset_steps is not None) else 0,
        "dataset_steps_used_train": (
            int(bc.dataset_steps_used_train)
            if (bc and bc.dataset_steps_used_train is not None)
            else 0
        ),
        "train_raw_steps": int(bc.train_raw_steps) if (bc and bc.train_raw_steps is not None) else 0,
        "seed": _safe_int(summary.get("seed")) or 0,
        "episodes_total": _safe_int(summary.get("episodes_total")) or 0,
        "episodes_used_for_summary": _safe_int(summary.get("episodes_used_for_summary")) or 0,
        "mean_env_reward": _safe_float(summary.get("mean_env_reward")),
        "std_env_reward": _safe_float(summary.get("std_env_reward")),
        "mean_episode_length": _safe_float(summary.get("mean_episode_length")),
        "goals_scored_per_episode": _safe_float(summary.get("goals_scored_per_episode")),
        "goals_conceded_per_episode": _safe_float(summary.get("goals_conceded_per_episode")),
        "goal_diff_per_episode": _safe_float(summary.get("goal_diff_per_episode")),
        "goal_episode_rate": _safe_float(summary.get("goal_episode_rate")),
        "conceded_episode_rate": _safe_float(summary.get("conceded_episode_rate")),
        "goal_steps_rate": _safe_float(summary.get("goal_steps_rate")),
        "conceded_steps_rate": _safe_float(summary.get("conceded_steps_rate")),
        "term_rate_goal_scored": _safe_float(summary.get("term_rate_goal_scored")),
        "term_rate_goal_conceded": _safe_float(summary.get("term_rate_goal_conceded")),
        "term_rate_time_out": _safe_float(summary.get("term_rate_time_out")),
        "term_rate_player_in_goal": _safe_float(summary.get("term_rate_player_in_goal")),
        "term_rate_opponent_in_goal": _safe_float(summary.get("term_rate_opponent_in_goal")),
        "elapsed_sec": _safe_float(summary.get("elapsed_sec")),
        "num_envs": _safe_int(summary.get("num_envs")) or 0,
        "summary_path": result.get("summary_path", ""),
        "log_path": result.get("log_path", ""),
    }
    return row


def _write_results_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_csv.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_matchups(raw: str) -> List[str]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("No matchups specified.")
    invalid = [p for p in parts if p not in VALID_MATCHUPS]
    if invalid:
        raise ValueError(f"Invalid matchup(s): {invalid}. Valid: {sorted(VALID_MATCHUPS)}")
    return parts


def _load_config_file(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r") as fh:
        cfg = json.load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must be a JSON object: {p}")
    return cfg


def _apply_config_overrides(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    if not cfg:
        return
    # Shallow mapping for the fields needed in this eval-only orchestrator.
    if "output_root" in cfg and not args.output_dir:
        args.output_root = cfg["output_root"]
    if "human_baseline_steps" in cfg:
        args.human_baseline_steps = int(cfg["human_baseline_steps"])
    if "policy_selection" in cfg:
        args.policy_selection = str(cfg["policy_selection"])
    if "official_dw_dataset_ids" in cfg and isinstance(cfg["official_dw_dataset_ids"], list):
        args.official_dw_dataset_id = [str(x) for x in cfg["official_dw_dataset_ids"]]
    if "official_human_run_name" in cfg:
        args.official_human_run_name = str(cfg["official_human_run_name"])
    if "eval" in cfg and isinstance(cfg["eval"], dict):
        eval_cfg = cfg["eval"]
        args.episodes = int(eval_cfg.get("episodes", args.episodes))
        args.num_envs = int(eval_cfg.get("num_envs", args.num_envs))
        args.device = str(eval_cfg.get("device", args.device))
        args.seed = int(eval_cfg.get("seed", args.seed))
    if "bc_runs_dir" in cfg:
        args.bc_runs_dir = str(cfg["bc_runs_dir"])
    if "expert_policy_path" in cfg:
        args.expert_policy_path = str(cfg["expert_policy_path"])
    if "expert_rl_games_config_path" in cfg:
        args.expert_rl_games_config_path = str(cfg["expert_rl_games_config_path"])
    if "matchups" in cfg:
        if isinstance(cfg["matchups"], list):
            args.matchups = ",".join(str(x) for x in cfg["matchups"])
        else:
            args.matchups = str(cfg["matchups"])


def main() -> None:
    args = parse_args()
    cfg = _load_config_file(args.config)
    _apply_config_overrides(args, cfg)

    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be > 0")
    if args.log_every_episodes <= 0:
        raise ValueError("--log-every-episodes must be > 0")

    eval_script = _resolve_eval_script(args.eval_script)
    isaaclab_cwd = Path(args.isaaclab_cwd).expanduser().resolve()
    if not isaaclab_cwd.exists():
        raise FileNotFoundError(f"isaaclab cwd not found: {isaaclab_cwd}")

    expert_ckpt = Path(args.expert_policy_path).expanduser().resolve()
    expert_cfg = Path(args.expert_rl_games_config_path).expanduser().resolve()
    if not expert_ckpt.exists():
        raise FileNotFoundError(f"PPO expert checkpoint not found: {expert_ckpt}")
    if not expert_cfg.exists():
        raise FileNotFoundError(f"PPO expert config not found: {expert_cfg}")

    output_dir = _build_output_dir(args)
    matchups = _parse_matchups(args.matchups)
    official_dw_dataset_ids = (
        list(args.official_dw_dataset_id)
        if list(args.official_dw_dataset_id)
        else list(DEFAULT_OFFICIAL_DW_DATASET_IDS)
    )
    bc_policies = _discover_bc_checkpoints(
        bc_runs_dir=Path(args.bc_runs_dir).expanduser().resolve(),
        explicit_checkpoints=list(args.bc_checkpoint),
        policy_selection=str(args.policy_selection),
        official_dw_dataset_ids=official_dw_dataset_ids,
        official_human_run_name=str(args.official_human_run_name),
        include_patterns=list(args.include),
        exclude_patterns=list(args.exclude),
        human_baseline_steps=int(args.human_baseline_steps),
        max_policies=args.max_policies,
    )
    jobs = _build_jobs(
        matchups=matchups,
        bc_policies=bc_policies,
        expert_policy_path=str(expert_ckpt),
        expert_rl_games_config_path=str(expert_cfg),
        expert_task=str(args.expert_task),
        output_dir=output_dir,
    )

    resolved_cfg = {
        "created_at": dt.datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "eval_script": str(eval_script),
        "isaaclab_cwd": str(isaaclab_cwd),
        "task": str(args.task),
        "episodes": int(args.episodes),
        "num_envs": int(args.num_envs),
        "device": str(args.device),
        "seed": int(args.seed),
        "deterministic_policy": bool(args.deterministic_policy),
        "deterministic_opponent": bool(args.deterministic_opponent),
        "summary_min_episode_length": int(args.summary_min_episode_length),
        "log_every_episodes": int(args.log_every_episodes),
        "reward_profile": str(args.reward_profile),
        "headless": bool(args.headless),
        "disable_fabric": bool(args.disable_fabric),
        "resume": bool(args.resume),
        "force": bool(args.force),
        "dry_run": bool(args.dry_run),
        "matchups": matchups,
        "policy_selection": str(args.policy_selection),
        "official_dw_dataset_ids": official_dw_dataset_ids,
        "official_human_run_name": str(args.official_human_run_name),
        "bc_runs_dir": str(Path(args.bc_runs_dir).expanduser().resolve()),
        "human_baseline_steps": int(args.human_baseline_steps),
        "expert_policy_path": str(expert_ckpt),
        "expert_rl_games_config_path": str(expert_cfg),
        "expert_task": str(args.expert_task),
        "bc_policies_discovered": len(bc_policies),
        "jobs_planned": len(jobs),
    }
    _write_json(output_dir / "resolved_config.json", resolved_cfg)
    _write_json(output_dir / "discovered_bc_policies.json", [asdict(p) for p in bc_policies])
    _write_json(output_dir / "planned_jobs.json", [asdict(j) for j in jobs])

    results: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for idx, job in enumerate(jobs, start=1):
        print(f"[job {idx}/{len(jobs)}] {job.job_id}")
        result = _run_eval_job(
            args=args,
            eval_script=eval_script,
            isaaclab_cwd=isaaclab_cwd,
            job=job,
        )
        results.append(result)
        if result.get("summary") is not None:
            rows.append(_job_row(job, result))

    _write_json(output_dir / "jobs.json", results)
    _write_results_csv(rows, output_dir / "results_raw.csv")
    _write_json(output_dir / "results_index.json", rows)

    print(f"[out] {output_dir / 'resolved_config.json'}")
    print(f"[out] {output_dir / 'discovered_bc_policies.json'}")
    print(f"[out] {output_dir / 'planned_jobs.json'}")
    print(f"[out] {output_dir / 'jobs.json'}")
    print(f"[out] {output_dir / 'results_raw.csv'}")
    print(f"[out] {output_dir / 'results_index.json'}")
    plotting_dir = Path(__file__).resolve().parent.parent / "plotting"
    print("[next] Aggregate scaling-law report:")
    print(
        f"  python {Path(__file__).resolve().parent / 'aggregate_data_wall_results.py'} "
        f"--results-csv {output_dir / 'results_raw.csv'}"
    )
    print("[next] Plot detailed matchup grids:")
    print(
        f"  python {plotting_dir / 'plot_data_wall_eval_results.py'} "
        f"--results-csv {output_dir / 'results_raw.csv'}"
    )
    print("[next] Plot paper-style summary figures:")
    print(
        f"  python {plotting_dir / 'plot_data_wall_summary_figures.py'} "
        f"--results {output_dir / 'results_index.json'}"
    )


if __name__ == "__main__":
    main()
