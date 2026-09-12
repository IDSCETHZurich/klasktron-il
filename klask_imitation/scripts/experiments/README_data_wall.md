# Data Wall Experiment

This pipeline runs a synthetic-data scaling study for BC and GAIL.

## What It Does

1. Generates synthetic demonstrations from one or more expert checkpoints.
2. Trains BC and GAIL from scratch at each dataset size.
3. Evaluates all trained policies with a shared evaluator.
4. Aggregates scaling plots and computes a plateau-based data-wall estimate.

## Key Scripts

- `generate_synthetic_dataset.py`: build cached sim-native demonstration datasets.
- `run_data_wall_experiment.py`: orchestrate generate/train/eval/aggregate stages.
- `aggregate_data_wall_results.py`: produce plot + report + table from raw results.

Both `generate_synthetic_dataset.py` and `evaluate_policy_matchup.py` now derive sensible default output directories when `--output-dir` is omitted, and they use `--force` for explicit overwrite behavior instead of silently reusing arbitrary directories.

## Plotting Scripts

These live under `../plotting/` so experiment orchestration and figure generation stay separate:

- `plot_data_wall_eval_results.py`: generate matchup-wise evaluation grids and CSV manifests from `results_raw.csv`.
- `plot_data_wall_summary_figures.py`: generate paper-style summary figures from `results_index.json`, including scaling curves, all-vs-all heatmaps, behavior-quality plots, and termination decompositions.

## Config

Use `data_wall_config.example.json` as a template.

Important fields:
- `experts`: list of synthetic experts (`policy_path`, `policy_format`, `mode`, optional opponent policy).
- RL-Games experts: set `policy_format: "rl_games"` and optionally `rl_games_task` / `rl_games_config_path`.
- For PPO expert play-style self-play, set `mode: "peg1_self_play"` (opponent policy is ignored) and provide the matching RL-Games `agent.yaml` in `rl_games_config_path`.
- Opponent policy can also be RL-Games via `opponent_policy_format: "rl_games"` and optional opponent RL-Games task/config fields.
- `synthetic_scales`: target action-step budgets.
- `human_baseline_steps`: human cleaned data size used for baseline row.
- `aggregate.human_overhead_factor`: multiplier for collection + cleaning in hour conversion.

## Run

```bash
python run_data_wall_experiment.py \
  --config data_wall_config.example.json \
  --stage all
```

Use an explicit output directory (recommended for resume):

```bash
python run_data_wall_experiment.py \
  --config data_wall_config.example.json \
  --output-dir ${KLASKTRON_ROOT}/trained_models/data_wall_experiments/data_wall_manual_01 \
  --stage all \
  --resume
```

## Stage-Only Runs

```bash
python run_data_wall_experiment.py --config data_wall_config.example.json --output-dir <exp_dir> --stage generate --resume
python run_data_wall_experiment.py --config data_wall_config.example.json --output-dir <exp_dir> --stage train --resume
python run_data_wall_experiment.py --config data_wall_config.example.json --output-dir <exp_dir> --stage eval --resume
python run_data_wall_experiment.py --config data_wall_config.example.json --output-dir <exp_dir> --stage aggregate
```

## Outputs

In `<exp_dir>`:
- `dataset_index.json`: all dataset descriptors.
- `train_meta/*.json`: per-job training artifacts.
- `eval_meta/*.json`: per-job evaluation summaries.
- `results_raw.csv`: flattened run table.
- `aggregate/scaling_law_mean_env_reward.png`
- `aggregate/data_wall_report.json`
- `aggregate/data_wall_table.csv`

## Post-Processing

From the repository root:

```bash
python klask_imitation/scripts/experiments/aggregate_data_wall_results.py \
  --results-csv <exp_dir>/results_raw.csv

python klask_imitation/scripts/plotting/plot_data_wall_eval_results.py \
  --results-csv <exp_dir>/results_raw.csv

python klask_imitation/scripts/plotting/plot_data_wall_summary_figures.py \
  --results <exp_dir>/results_index.json
```

---
# Details
## Generate Synthetic datasets:
Paste these cli commands

### Self play of 80k BC policy:
cd ${ISAACLAB_ROOT} && python ${KLASKTRON_ROOT}/klask_imitation/scripts/experiments/generate_synthetic_dataset.py \
  --checkpoint ${KLASKTRON_ROOT}/trained_models/data_wall_experiments/bc_runs/klask_bc_20260303_161251_HUMAN_DEMO_V2/ppo_policy_bc_final.zip \
  --policy-format sb3_ppo \
  --force-self-play \
  --target-steps 80000 \
  --num_envs 256 \
  --device cuda:0 \
  --min-traj-len 1 \
  --output-dir ${KLASKTRON_ROOT}/artifacts/data_wall/datasets/bc_80k_selfplay_synthetic \
  --force

### Self play of 250k BC policy:
cd ${ISAACLAB_ROOT} && python ${KLASKTRON_ROOT}/klask_imitation/scripts/experiments/generate_synthetic_dataset.py \
  --checkpoint ${KLASKTRON_ROOT}/trained_models/data_wall_experiments/bc_runs/klask_bc_20260301_191609_DW_EXPERT_250000/ppo_policy_bc_final.zip \
  --policy-format sb3_ppo \
  --force-self-play \
  --target-steps 80000 \
  --num_envs 256 \
  --device cuda:3 \
  --min-traj-len 1 \
  --output-dir ${KLASKTRON_ROOT}/artifacts/data_wall/datasets/bc_250k_selfplay_80k \
  --force

### Self play of 1Mio BC policy:
cd ${ISAACLAB_ROOT} && python ${KLASKTRON_ROOT}/klask_imitation/scripts/experiments/generate_synthetic_dataset.py \
  --checkpoint ${KLASKTRON_ROOT}/trained_models/data_wall_experiments/bc_runs/klask_bc_20260301_195909_DW_EXPERT_1000000/ppo_policy_bc_final.zip \
  --policy-format sb3_ppo \
  --force-self-play \
  --target-steps 80000 \
  --num_envs 256 \
  --device cuda:0 \
  --min-traj-len 1 \
  --output-dir ${KLASKTRON_ROOT}/artifacts/data_wall/datasets/bc_1m_selfplay_80k \
  --force

### Self play of 5Mio BC policy:
cd ${ISAACLAB_ROOT} && python ${KLASKTRON_ROOT}/klask_imitation/scripts/experiments/generate_synthetic_dataset.py \
  --checkpoint ${KLASKTRON_ROOT}/trained_models/data_wall_experiments/bc_runs/klask_bc_20260302_142009_DW_EXPERT_5000000/ppo_policy_bc_final.zip \
  --policy-format sb3_ppo \
  --force-self-play \
  --target-steps 80000 \
  --num_envs 256 \
  --device cuda:0 \
  --min-traj-len 1 \
  --output-dir ${KLASKTRON_ROOT}/artifacts/data_wall/datasets/bc_5m_selfplay_80k \
  --force

--- 
### policy 80k vs expert best agent:
cd ${ISAACLAB_ROOT} && python ${KLASKTRON_ROOT}/klask_imitation/scripts/experiments/generate_synthetic_dataset.py \
  --checkpoint ${KLASKTRON_ROOT}/trained_models/data_wall_experiments/bc_runs/klask_bc_20260301_185423_DW_EXPERT_80000/ppo_policy_bc_final.zip \
  --policy-format sb3_ppo \
  --opponent-policy-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/klask.pth \
  --opponent-policy-format rl_games \
  --opponent-rl-games-config-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/agent.yaml \
  --target-steps 80000 \
  --num_envs 256 \
  --device cuda:3 \
  --min-traj-len 1 \
  --output-dir ${KLASKTRON_ROOT}/artifacts/data_wall/datasets/bc_80k_vs_expert_80k \
  --force \
  --rl-games-config-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/agent.yaml



### policy 250k vs expert best agent:
cd ${ISAACLAB_ROOT} && python ${KLASKTRON_ROOT}/klask_imitation/scripts/experiments/generate_synthetic_dataset.py \
  --checkpoint ${KLASKTRON_ROOT}/trained_models/data_wall_experiments/bc_runs/klask_bc_20260301_191609_DW_EXPERT_250000/ppo_policy_bc_final.zip \
  --policy-format sb3_ppo \
  --opponent-policy-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/klask.pth \
  --opponent-policy-format rl_games \
  --opponent-rl-games-config-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/agent.yaml \
  --target-steps 80000 \
  --num_envs 256 \
  --device cuda:0 \
  --min-traj-len 1 \
  --output-dir ${KLASKTRON_ROOT}/artifacts/data_wall/datasets/bc_250k_vs_expert_80k \
  --force \
  --rl-games-config-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/agent.yaml



### policy 1Mio vs expert best agent:
cd ${ISAACLAB_ROOT} && python ${KLASKTRON_ROOT}/klask_imitation/scripts/experiments/generate_synthetic_dataset.py \
  --checkpoint ${KLASKTRON_ROOT}/trained_models/data_wall_experiments/bc_runs/klask_bc_20260301_195909_DW_EXPERT_1000000/ppo_policy_bc_final.zip \
  --policy-format sb3_ppo \
  --opponent-policy-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/klask.pth \
  --opponent-policy-format rl_games \
  --opponent-rl-games-config-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/agent.yaml \
  --target-steps 80000 \
  --num_envs 256 \
  --device cuda:0 \
  --min-traj-len 1 \
  --output-dir ${KLASKTRON_ROOT}/artifacts/data_wall/datasets/bc_1m_vs_expert_80k \
  --force \
  --rl-games-config-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/agent.yaml

### policy 5Mio vs expert best agent:
cd ${ISAACLAB_ROOT} && python ${KLASKTRON_ROOT}/klask_imitation/scripts/experiments/generate_synthetic_dataset.py \
  --checkpoint ${KLASKTRON_ROOT}/trained_models/data_wall_experiments/bc_runs/klask_bc_20260302_142009_DW_EXPERT_5000000/ppo_policy_bc_final.zip \
  --policy-format sb3_ppo \
  --opponent-policy-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/klask.pth \
  --opponent-policy-format rl_games \
  --opponent-rl-games-config-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/agent.yaml \
  --target-steps 80000 \
  --num_envs 256 \
  --device cuda:0 \
  --min-traj-len 1 \
  --output-dir ${KLASKTRON_ROOT}/artifacts/data_wall/datasets/bc_5m_vs_expert_80k \
  --force \
  --rl-games-config-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/agent.yaml

--- 
### Human BC policy vs expert best agent:
cd ${ISAACLAB_ROOT} && python ${KLASKTRON_ROOT}/klask_imitation/scripts/experiments/generate_synthetic_dataset.py \
  --checkpoint ${KLASKTRON_ROOT}/trained_models/data_wall_experiments/bc_runs/klask_bc_20260303_161251_HUMAN_DEMO_V2/ppo_policy_bc_final.zip \
  --policy-format sb3_ppo \
  --opponent-policy-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/klask.pth \
  --opponent-policy-format rl_games \
  --opponent-rl-games-config-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/agent.yaml \
  --target-steps 80000 \
  --num_envs 256 \
  --device cuda:0 \
  --min-traj-len 1 \
  --output-dir ${KLASKTRON_ROOT}/artifacts/data_wall/datasets/bc_human_v2_vs_expert \
  --force \
  --rl-games-config-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/agent.yaml




  ### Human BC policy vs 5Mio BC policy:
cd ${ISAACLAB_ROOT} && python ${KLASKTRON_ROOT}/klask_imitation/scripts/experiments/generate_synthetic_dataset.py \
  --checkpoint ${KLASKTRON_ROOT}/trained_models/data_wall_experiments/bc_runs/klask_bc_20260303_161251_HUMAN_DEMO_V2/ppo_policy_bc_final.zip \
  --policy-format sb3_ppo \
  --opponent-policy-path ${KLASKTRON_ROOT}/trained_models/data_wall_experiments/bc_runs/klask_bc_20260302_142009_DW_EXPERT_5000000/ppo_policy_bc_final.zip \
  --opponent-policy-format sb3_ppo \
  --target-steps 80000 \
  --num_envs 256 \
  --device cuda:0 \
  --min-traj-len 1 \
  --output-dir ${KLASKTRON_ROOT}/artifacts/data_wall/datasets/bc_human_vs_bc_5Mio \
  --force \


---

### PPO expert vs PPO expert policy:
cd ${ISAACLAB_ROOT} && python ${KLASKTRON_ROOT}/klask_imitation/scripts/experiments/generate_synthetic_dataset.py \
  --checkpoint ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/klask.pth \
  --policy-format rl_games \
  --rl-games-config-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/agent.yaml \
  --opponent-policy-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/klask.pth \
  --opponent-policy-format rl_games \
  --opponent-rl-games-config-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/agent.yaml \
  --target-steps 80000 \
  --num_envs 256 \
  --device cuda:0 \
  --min-traj-len 1 \
  --output-dir ${KLASKTRON_ROOT}/artifacts/data_wall/datasets/expert_vs_expert_80k \
  --force


  ---
  # BC cloning experiment
  Ran on the synthetic datasets ensuring ca 312 steps/episode = 6.24 sec by adjusting num_env depending on desired dataset size (total number of timesteps)

  - For 80k: 256
  - For 250k: 800
  - For 1Mio: 3'200
  - For 5Mio: 16'000

  Saw that human data lead to way lower training vs validation loss scale than the synthetic data (-25 vs -5).
  This is likely due to data leakage: Human data didnt record the applied velocity commands but estimated the velocity based on the positon change/time.
  This creates data leakage! It simply learns that the action label given a state, i.e. the velocity applied at a given state (which it tries to predict) is interpolated from the positions.

  However this is not the case for the synthetic data, because of disturbances and actuator physics (modeled by a  neural network) where the velocity action label is not just pos_t+1 - pos_t / delta_t but a more complicated relationship. I.e. the observed velocities returned by the environment are not = to the applied velocity command (i.e. the label)
  Therefore the loss curves of human vs synthetic data BC is not directly comparable. It becomes a better comparison if we set the observation channels that correspond to velocities to zero for the human expert data.
  Now it has slightly less information than synthetic data, but the mismatch is much smaller (confirmed by testing BC on a synthetic dataset with and without velocity channel masking).

  ---
  

  ## Experiment Evaluation for BC policies
cd ${ISAACLAB_ROOT} && ${PYTHON_BIN} ${KLASKTRON_ROOT}/klask_imitation/scripts/experiments/run_data_wall_experiment.py \
  --output-dir ${KLASKTRON_ROOT}/artifacts/data_wall/eval_runs/bc_data_wall_official_sparse_plus_collision_20260304 \
  --policy-selection official \
  --matchups all_vs_all \
  --episodes 800 \
  --num-envs 300 \
  --device cuda:0 \
  --reward-profile sparse_plus_collision \
  --resume

# same but added +0.1 reward for ball collisions and used Human BC policy without masked vel. obs (better agent)
cd ${ISAACLAB_ROOT} && ${PYTHON_BIN} ${KLASKTRON_ROOT}/klask_imitation/scripts/experiments/run_data_wall_experiment.py \
  --output-dir ${KLASKTRON_ROOT}/artifacts/data_wall/eval_runs/bc_data_wall_official_sparse_plus_collision_20260304 \
  --policy-selection official \
  --matchups all_vs_all \
  --episodes 800 \
  --num-envs 300 \
  --device cuda:0 \
  --reward-profile sparse_plus_collision \
  --resume




# Note: with --resume, changed reward profiles require either a new --output-dir
# (as above) or adding --force to recompute existing summaries.



watch with: tail -f ${KLASKTRON_ROOT}/artifacts/data_wall/eval_runs/bc_data_wall_official_20260302/matchups/bc_vs_expert/klask_bc_20260301_185423_DW_EXPERT_80000/run.log




# GAIL
# Train on synthetic and human data:
# Fixed-budget protocol: all runs use the same total training timesteps.
# (No autoscaling.)

# Synthetic 80k (sim_native), no velocity masking
python ${KLASKTRON_ROOT}/klask_imitation/scripts/training/train_gail_sweep.py \
  --demo-base ${KLASKTRON_ROOT}/artifacts/data_wall/datasets/ppo_expert_synthetic_80000/trajectories \
  --demo-source-format sim_native \
  --run-tag GAIL_DW_EXPERT_80k \
  --total-timesteps 5000000 \
  --wandb --wandb-project klask-gail --wandb-mode online \
  --wandb-run-name GAIL_synthetic_80k


# Human demos (human_raw), augmented (default), no velocity masking (default)
python ${KLASKTRON_ROOT}/klask_imitation/scripts/training/train_gail_sweep.py \
  --demo-base ${KLASKTRON_ROOT}/data/expert_trajectories_cleaned \
  --demo-source-format human_raw \
  --augment \
  --run-tag GAIL_Human \
  --total-timesteps 5000000 \
  --wandb --wandb-project klask-gail --wandb-mode online \
  --wandb-run-name GAIL_human_augmented

# Human demos (human_raw), augmented, with velocity masking enabled
python ${KLASKTRON_ROOT}/klask_imitation/scripts/training/train_gail_sweep.py \
  --demo-base ${KLASKTRON_ROOT}/data/expert_trajectories_cleaned \
  --demo-source-format human_raw \
  --augment \
  --masked-velocities \
  --run-tag GAIL_Human_masked_vel \
  --total-timesteps 5000000 \
  --wandb --wandb-project klask-gail --wandb-mode online \
  --wandb-run-name GAIL_human_augmented_masked_vel

# Human demos non-augmented??:
python ${KLASKTRON_ROOT}/klask_imitation/scripts/training/train_gail_sweep.py \
  --demo-base ${KLASKTRON_ROOT}/data/expert_trajectories_cleaned \
  --demo-source-format human_raw \
  --run-tag GAIL_Human \
  --total-timesteps 5000000 \
  --wandb --wandb-project klask-gail --wandb-mode online \
  --wandb-run-name GAIL_human



# 250k
python ${KLASKTRON_ROOT}/klask_imitation/scripts/training/train_gail_sweep.py \
  --demo-base ${KLASKTRON_ROOT}/artifacts/data_wall/datasets/ppo_expert_synthetic_250000/trajectories \
  --demo-source-format sim_native \
  --run-tag GAIL_DW_EXPERT_250k \
  --total-timesteps 5000000 \
  --wandb --wandb-project klask-gail --wandb-mode online \
  --wandb-run-name GAIL_synthetic_250k \
  --device cuda:1 \

# 1Mio
python ${KLASKTRON_ROOT}/klask_imitation/scripts/training/train_gail_sweep.py \
  --demo-base ${KLASKTRON_ROOT}/artifacts/data_wall/datasets/ppo_expert_synthetic_1000000/trajectories \
  --demo-source-format sim_native \
  --run-tag GAIL_DW_EXPERT_1Mio \
  --total-timesteps 5000000 \
  --wandb --wandb-project klask-gail --wandb-mode online \
  --wandb-run-name GAIL_synthetic_1Mio

# 5Mio
python ${KLASKTRON_ROOT}/klask_imitation/scripts/training/train_gail_sweep.py \
  --demo-base ${KLASKTRON_ROOT}/artifacts/data_wall/datasets/ppo_expert_synthetic_5000000/trajectories \
  --demo-source-format sim_native \
  --run-tag GAIL_DW_EXPERT_5Mio \
  --total-timesteps 5000000 \
  --wandb --wandb-project klask-gail --wandb-mode online \
  --wandb-run-name GAIL_synthetic_5Mio_V2
  --n-disc-updates 1


# stable training using more rounds (fewer envs)
python ${KLASKTRON_ROOT}/klask_imitation/scripts/training/train_gail_sweep.py \
  --demo-base ${KLASKTRON_ROOT}/artifacts/data_wall/datasets/ppo_expert_synthetic_250000/trajectories \
  --demo-source-format sim_native \
  --run-tag GAIL_DW_EXPERT_250k \
  --total-timesteps 5000000 \
  --wandb --wandb-project klask-gail --wandb-mode online \
  --wandb-run-name GAIL_synthetic_250k \
  --device cuda:0 \
  --num-envs 256 \
  --n-steps 128 \
  --no-adaptive-disc-updates \
  --n-disc-updates 1


## replay and store trajectories
### policy vs expert best agent:
cd ${ISAACLAB_ROOT} && python ${KLASKTRON_ROOT}/klask_imitation/scripts/experiments/generate_synthetic_dataset.py \
  --checkpoint ${KLASKTRON_ROOT}/trained_models/data_wall_experiments/gail_runs/klask_gail_20260306_170752_GAIL_DW_EXPERT_250k_freeze_only/ppo_policy_gail_final.zip \
  --policy-format sb3_ppo \
  --opponent-policy-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/klask.pth \
  --opponent-policy-format rl_games \
  --opponent-rl-games-config-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/agent.yaml \
  --target-steps 80000 \
  --num_envs 256 \
  --device cuda:0 \
  --min-traj-len 1 \
  --output-dir ${KLASKTRON_ROOT}/artifacts/data_wall/datasets/synth_250k_gail_adaptive_ndisc_vs_expert\
  --force \
  --rl-games-config-path ${ISAACLAB_ROOT}/logs/rl_games/klask/demo_agents/best_one/agent.yaml


# gail policy self play:
cd ${ISAACLAB_ROOT} && python ${KLASKTRON_ROOT}/klask_imitation/scripts/experiments/generate_synthetic_dataset.py \
  --checkpoint ${KLASKTRON_ROOT}/trained_models/data_wall_experiments/gail_runs/klask_gail_20260303_152921_GAIL_DW_EXPERT_250k_stable_v2/ppo_policy_gail_final.zip \
  --policy-format sb3_ppo \
  --force-self-play \
  --target-steps 80000 \
  --num_envs 256 \
  --device cuda:0 \
  --min-traj-len 1 \
  --output-dir ${KLASKTRON_ROOT}/artifacts/data_wall/datasets/gail_250k_self_play \
  --force



## GAIL discriminator analysis (reward-map + overfitting)

### Split-aware GAIL training defaults
`train_gail_sweep.py` now supports validation splitting (default: file-level 90/10, seed=`--seed`) and writes `dataset_split.json` in each run directory.

Disable splitting only if needed:

```bash
python ${KLASKTRON_ROOT}/klask_imitation/scripts/training/train_gail_sweep.py \
  --demo-base ${KLASKTRON_ROOT}/data/expert_trajectories_cleaned \
  --demo-source-format human_raw \
  --run-tag GAIL_Human_no_split \
  --no-val-split
```

### Analyze a single GAIL run

```bash
cd ${ISAACLAB_ROOT} && ${PYTHON_BIN} \
  ${KLASKTRON_ROOT}/klask_imitation/scripts/eval/evaluate_gail_results.py \
  --run-dir ${KLASKTRON_ROOT}/trained_models/data_wall_experiments/gail_runs/klask_gail_20260306_175727_GAIL_DW_EXPERT_250k_freeze_only/ \
  --output-dir ${KLASKTRON_ROOT}/trained_models/data_wall_experiments/gail_runs/klask_gail_20260306_175727_GAIL_DW_EXPERT_250k_freeze_only/eval/ \
  --analysis-output-dir ${KLASKTRON_ROOT}/trained_models/data_wall_experiments/gail_runs/klask_gail_20260306_175727_GAIL_DW_EXPERT_250k_freeze_only/eval/ \
  --run-discriminator-analysis \
  --reward-map-target peg1 \
  --reward-map-bins 80 \
  --reward-map-context-samples 4096 \
  --disc-generated-samples 0 \
  --episodes 800 \
  --num-envs 300 \
  --device cuda:0
```





### Official 5-run batch (latest run per tag)

```bash
cd ${ISAACLAB_ROOT} && ${PYTHON_BIN} \
  ${KLASKTRON_ROOT}/klask_imitation/scripts/experiments/run_gail_discriminator_analysis.py \
  --gail-runs-dir ${KLASKTRON_ROOT}/trained_models/data_wall_experiments/gail_runs \
  --output-root ${KLASKTRON_ROOT}/artifacts/data_wall/analysis/gail_discriminator \
  --official-tags GAIL_Human,GAIL_DW_EXPERT_80k,GAIL_DW_EXPERT_250k,GAIL_DW_EXPERT_1Mio,GAIL_DW_EXPERT_5Mio \
  --reward-map-target peg1 \
  --reward-map-bins 80 \
  --reward-map-context-samples 4096 \
  --disc-generated-samples 0 \
  --episodes 800 \
  --num-envs 300 \
  --device cuda:0
```

### Output structure

- Per run split provenance:
  - `trained_models/data_wall_experiments/gail_runs/<run_name>/dataset_split.json`
- Batch analysis root:
  - `artifacts/data_wall/analysis/gail_discriminator/<analysis_id>/resolved_config.json`
  - `artifacts/data_wall/analysis/gail_discriminator/<analysis_id>/run_index.json`
  - `artifacts/data_wall/analysis/gail_discriminator/<analysis_id>/runs/<run_name>/discriminator_metrics.json`
  - `artifacts/data_wall/analysis/gail_discriminator/<analysis_id>/runs/<run_name>/analysis_summary.json`
  - `artifacts/data_wall/analysis/gail_discriminator/<analysis_id>/runs/<run_name>/reward_map_peg1.npy`
  - `artifacts/data_wall/analysis/gail_discriminator/<analysis_id>/runs/<run_name>/reward_map_peg1.png`
  - `artifacts/data_wall/analysis/gail_discriminator/<analysis_id>/aggregate/official5_metrics.csv`
  - `artifacts/data_wall/analysis/gail_discriminator/<analysis_id>/aggregate/official5_metrics.json`
  - `artifacts/data_wall/analysis/gail_discriminator/<analysis_id>/aggregate/overfit_gap_by_dataset.png`


/bin/bash -lc '
source "${CONDA_SH}"
conda activate env_isaaclab
export COLCON_TRACE="${COLCON_TRACE:-}"
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
set +u
source "${ROS_SETUP}"
set -u
${PYTHON_BIN} \
  ${KLASKTRON_ROOT}/klask_imitation/scripts/experiments/run_gail_discriminator_analysis.py \
  --python-bin ${PYTHON_BIN} \
  --official-tags GAIL_DW_EXPERT_80k_EVENT_AWARE,GAIL_DW_EXPERT_250k_EVENT_AWARE,GAIL_DW_EXPERT_1Mio_EVENT_AWARE,GAIL_DW_EXPERT_5Mio_EVENT_AWARE \
  --run-id gail_disc_analysis_event_aware_latest_20260312 \
  --device cuda:0
'
