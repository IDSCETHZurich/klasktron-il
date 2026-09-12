# KlaskTron-IL

Imitation-learning benchmark for the Klask tabletop game, built on Isaac Sim / Isaac Lab.

Code for *KlaskTron-IL: A Contact-Rich Adversarial Imitation Learning Benchmark
with a Validated Sim-to-Real Bridge* (CoRL 2026).

Dataset: <https://huggingface.co/datasets/KlaskLab/klasktron-il-benchmark>

## Setup

1. Install Isaac Lab: https://isaac-sim.github.io/IsaacLab/
2. Activate the Isaac Lab Python env, then:

```bash
pip install -e ./klask_imitation[dev,data]
```

## Dataset Download

Three tiers: `human` (~80k steps), `augmented` (~325k steps), `synthetic` (5M+ steps).

```bash
klasktron-download-data --repo-id KlaskLab/klasktron-il-benchmark --tier human
klasktron-download-data --repo-id KlaskLab/klasktron-il-benchmark --tier all
```

Datasets land in `./data` by default. Override with `--local-dir` or `KLASKTRON_DATA_DIR`.

See [dataset_card.md](dataset_card.md) for format details.

## Action bound and actuator model

Two training regimes ship side by side, and **every script states which one it is
using on startup**. They are not interchangeable: numbers from one cannot be pooled
with numbers from the other.

| | `--action-limit` | actuator model | what it is for |
|---|---|---|---|
| **Deployment (default)** | `0.2` | on | Matches the physical robot, which clips commands at 0.2 m/s and whose realised velocity is mediated by the learned actuator model. Policies trained this way share the interface the sim-to-real bridge was validated on. |
| **Paper reproduction** | `1.0` | off | Reproduces the published Table 2 / Fig. 5 / Sec. 5.4 numbers, which were produced before the interface was standardised. |

Which knob applies to which method:

| method | rolls out during training? | `--action-limit` | actuator model |
|---|---|---|---|
| BC (`train_BC_only.py`) | **no** — supervised on a fixed dataset | n/a (the policy's bound matters only at deployment) | n/a |
| GAIL (`train_gail_sweep.py`) | yes (generator) | yes | yes (`--actuator-model` / `--no-actuator-model`) |
| BC+RL (`train_GAIL_BC_sparse_RL.py`) | yes (PPO) | yes | always on |

Two details worth knowing:

- The bound is applied to the **frozen opponent as well as the learner**, so both
  sides of a match respect the limit the hardware enforces.
- When a run is bootstrapped from a BC policy, the copied action head is rescaled by
  `1 / action_limit`. A squashed policy emits `action_limit * tanh(h)` while the BC
  head emits raw m/s, so without the rescale a BC-initialised run at 0.2 would start
  at one fifth of the intended command. At `1.0` the rescale is a no-op, which is why
  the published configuration reproduces bit-identically.

Examples:

```bash
# deployment-faithful (both are the defaults, shown here for clarity)
python klask_imitation/scripts/training/train_gail_sweep.py --config configs/training/gail_baseline.json \
    --action-limit 0.2 --actuator-model

# paper reproduction
python klask_imitation/scripts/training/train_gail_sweep.py --config configs/training/gail_baseline.json \
    --action-limit 1.0 --no-actuator-model
```

**Known result:** under sparse reward, PPO from random initialisation does not learn
within 18M steps at `--action-limit 0.2` — goals per episode stay flat at the
untrained level, with or without a BC prior. The published sparse results in Sec. 5.4
depend on the `1.0` bound. Use `0.2` when the goal is a deployable policy, `1.0` when
the goal is reproducing the paper.

## Training

All scripts accept `--config` to load a JSON preset. CLI flags override config values.
wandb is disabled by default; pass `--wandb-mode online` to enable.

### Behavioral Cloning

```bash
python klask_imitation/scripts/training/train_BC_only.py --config configs/training/bc_baseline.json
```

### GAIL

Single run on human demos:

```bash
python klask_imitation/scripts/training/train_gail_sweep.py --config configs/training/gail_baseline.json --run-tag GAIL_HUMAN
```

Other dataset tiers (change `--demo-base` and `--demo-source-format`):

```bash
python klask_imitation/scripts/training/train_gail_sweep.py --config configs/training/gail_baseline.json --demo-base data/ppo_expert_synthetic_80000/trajectories --demo-source-format sim_native --run-tag GAIL_SYNTH_80k
python klask_imitation/scripts/training/train_gail_sweep.py --config configs/training/gail_baseline.json --demo-base data/ppo_expert_synthetic_250000/trajectories --demo-source-format sim_native --run-tag GAIL_SYNTH_250k
python klask_imitation/scripts/training/train_gail_sweep.py --config configs/training/gail_baseline.json --demo-base data/ppo_expert_synthetic_1000000/trajectories --demo-source-format sim_native --run-tag GAIL_SYNTH_1M
python klask_imitation/scripts/training/train_gail_sweep.py --config configs/training/gail_baseline.json --demo-base data/ppo_expert_synthetic_5000000/trajectories --demo-source-format sim_native --run-tag GAIL_SYNTH_5M
```

### BC Sweep (dataset-scale ablation)

```bash
python klask_imitation/scripts/training/train_bc_sweep.py
```

### Sparse RL (bootstrapped from BC/GAIL policy)

```bash
python klask_imitation/scripts/training/train_GAIL_BC_sparse_RL.py --config configs/training/sparse_rl_bootstrap.json
```

Random-init ablation:

```bash
python klask_imitation/scripts/training/train_GAIL_BC_sparse_RL.py --config configs/training/sparse_rl_random.json
```

### BC+RL 2x2 with error bars (Figure 5)

The published Fig. 5 reports a single training run per condition. The camera-ready
figure reports five seeds per condition with 95% confidence intervals, every policy
scored against the same frozen PPO reference opponent (Elo 1283).

```bash
# 4 conditions x 5 seeds = 20 runs; ~70 min each, one run per GPU
python klask_imitation/scripts/experiments/run_bcrl_multiseed.py --stage train --gpus 0,1,2,3 --resume
python klask_imitation/scripts/experiments/run_bcrl_multiseed.py --stage eval  --gpus 0,1,2,3 --resume
python klask_imitation/scripts/experiments/run_bcrl_multiseed.py --stage aggregate
python klask_imitation/scripts/plotting/make_fig5.py
```

This runner defaults to `--action-limit 1.0`, the bound behind the published figure.
Pass `--action-limit 0.2` for the deployment-faithful variant; the run tag records the
bound (`A10` / `A02`) so the two can never be pooled by `--resume`.

The shipped figure is `docs/figures/fig5_bcrl_2x2_seeds.pdf`.

### Synthetic Expert Generation

```bash
python klask_imitation/scripts/experiments/generate_synthetic_dataset.py --checkpoint <policy-checkpoint> --policy-format sb3_ppo --target-steps 5000000 --output-dir <output-dir>
```

## Playing / Visualization

Watch a trained policy play in the simulator:

```bash
python klask_imitation/scripts/tests/play.py --checkpoint <policy.zip> --num-envs 1
```

Against a fixed opponent (e.g. PPO expert agent):

```bash
python klask_imitation/scripts/tests/play.py --checkpoint <policy.zip> --opponent-policy-path <opponent.pth> --num-envs 1
```

Headless (no GUI, just prints episode stats):

```bash
python klask_imitation/scripts/tests/play.py --checkpoint <policy.zip> --headless --num-envs 16 --episodes 100
```

## Evaluation

```bash
python klask_imitation/scripts/eval/evaluate_gail_results.py --run-dir <trained-run-dir> --policy-path <checkpoint>
```

Other eval scripts: `evaluate_bc_results.py`, `evaluate_sparse_rl_results.py`.

Data-wall experiment scripts are under `klask_imitation/scripts/experiments/`.
Plotting scripts are under `klask_imitation/scripts/plotting/`.

## Reestimation Pipeline

The human state-reestimation pipeline (camera -> ball/peg tracking -> trajectories) is in a separate repo: [klask_reestimate](https://github.com/IDSCETHZurich/klask_reestimate).

Flow: `klask_reestimate` publishes `/ball_peg_states` -> `data/convert.py` converts bags to pkl -> training scripts load as `human_raw`.

Validate new exports before training:

```python
from klasktron.reestimation import validate_export_dir
result = validate_export_dir("data/expert_trajectories_cleaned")
print(result.valid, result.n_trajectories, result.n_steps)
```

See [docs/reestimation_integration.md](docs/reestimation_integration.md) for details.

## Repository Layout

```
configs/                    # training presets (JSON)
klask_imitation/
  my_isaaclab_tasks/        # Isaac Lab Klask env + wrappers
  klasktron/                # dataset utils, reestimation adapter
  scripts/training/         # BC, GAIL, sparse RL training
  scripts/eval/             # evaluation scripts
  scripts/experiments/      # synthetic data, data-wall
  scripts/plotting/         # paper figures (make_fig5.py = Fig. 5)
  scripts/play.py           # visualization / play script
data/                       # datasets (not in git)
trained_models/             # checkpoints
artifacts/                  # eval outputs, plots
docs/figures/               # camera-ready figures
```

## Citation

<!-- TODO(camera-ready): fill in the author list before making this repository public. -->

```bibtex
@inproceedings{klasktron2026,
  title     = {KlaskTron-IL: A Contact-Rich Adversarial Imitation Learning
               Benchmark with a Validated Sim-to-Real Bridge},
  author    = {FILL IN AUTHOR LIST},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```

## License

Source code in this repository is released under the Apache License 2.0 --
see [LICENSE](LICENSE).

Portions of `klask_imitation/my_isaaclab_tasks/` are derived from
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) and remain under the
BSD-3-Clause licence; see [NOTICE](NOTICE) for the file list and attribution.

The KlaskTron-IL dataset is distributed separately under CC-BY-4.0 --
see [dataset_card.md](dataset_card.md).
