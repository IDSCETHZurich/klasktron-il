# Target Repository Layout

## Current Constraints

The current codebase still reflects an internal research workspace, although the public paper scope has now been reduced to the scripts needed for:

- BC training
- BC sweep experiments
- GAIL training
- GAIL sweep experiments
- sparse RL fine-tuning from a bootstrapped policy
- synthetic dataset generation
- data-wall experiments and aggregation
- benchmark evaluation

The remaining cleanup concerns are:

- simulation code, baselines, experiments, and eval scripts live under one package root
- datasets and trained models exist as repo-local artifact directories
- several scripts depend on absolute machine-specific paths
- the human reestimation pipeline lives in a separate repository

## Target Public Layout

The target public layout for the GitHub release is:

```text
.
├── README.md
├── dataset_card.md
├── LICENSE
├── docs/
├── artifacts/
│   ├── evaluations/
│   ├── plots/
│   ├── data_wall/
│   └── optuna/
├── legacy_models/
├── configs/
│   └── training/                # BC, GAIL, sparse-RL release presets
├── trained_models/
│   ├── minimal active checkpoints only
│   └── data_wall_experiments/
│       ├── bc_runs/
│       └── gail_runs/
├── klask_imitation/
│   ├── my_isaaclab_tasks/          # Isaac Lab task definitions and wrappers
│   ├── klasktron/
│   │   ├── data/                   # Hugging Face download + dataset utilities
│   │   ├── reestimation/           # stable seam for future merge
│   │   └── paths.py                # repo-relative path resolution
│   └── scripts/
│       ├── training/               # BC, BC sweep, GAIL, GAIL sweep, sparse RL
│       ├── eval/                   # BC, GAIL, sparse-RL evaluation
│       ├── experiments/            # synthetic data + data-wall runners/aggregation
│       └── plotting/               # plot-only scripts for eval and paper figures
├── tests/
└── pyproject.toml
```

## Reestimation Seam

The future merge point for the human reestimation repo is:

```text
klask_imitation/klasktron/reestimation/
```

That layer should remain thin and stable:

- it accepts cleaned trajectory exports from the external repo
- it validates them against the benchmark schema
- it does not own the upstream estimation logic itself

## Migration Plan

### Phase 1

- add release-facing docs
- add Hugging Face dataset downloader
- centralize path defaults
- stop hardcoding workstation-specific paths in wrapper scripts

### Phase 2

- prune legacy algorithms and internal experiment scripts outside the paper scope
- move training defaults into `configs/training/*.json`
- split reusable utilities out of monolithic scripts
- remove archived and generated artifacts from the tracked tree

### Phase 3

- unify package naming around `klasktron`
- expose stable CLI entry points for training and evaluation
- merge or vendor the reestimation interface once the external repo is ready

## Checkpoint Policy

- `trained_models/data_wall_experiments/bc_runs/` is the canonical BC run bank for the paper
- the official human BC baseline is `klask_bc_20260303_161251_HUMAN_DEMO_V2`
- top-level `trained_models/` entries should stay minimal and only exist when a non-data-wall checkpoint is still needed for provenance, such as the retained sparse-RL bootstrap source
- superseded checkpoints should move to `legacy_models/` instead of staying alongside active release defaults
