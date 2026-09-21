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
The `--tier` name and the directory it creates differ for the augmented tier:

| `--tier` | lands in | `--demo-base` | `--demo-source-format` |
|---|---|---|---|
| `human` | `data/human/trajectories` | `data/human/trajectories` | `human_raw` |
| `augmented` | `data/human_augmented/trajectories` | `data/human_augmented/trajectories` | `sim_native` |
| `synthetic` | `data/synthetic/trajectories` | `data/synthetic/trajectories` | `sim_native` |

The default download also writes the dataset card over `data/README.md`; pass
`--no-docs` to keep the conversion-pipeline notes that ship in this repository.

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

Pass `--seed` once per seed to build a multi-seed sweep; the script trains one
configuration per invocation.

> **Figure 3 pre-dates this release.** The published Figure 3 (BC data scaling,
> five seeds per scale) was produced on 2026-05-27, before this repository was
> assembled, and its run artifacts are not included here. The script above
> reproduces the training side of that experiment, but the figure itself cannot be
> regenerated from the released code and checkpoints. The published values are the
> record of that run rather than an output of this repository.

### Sparse RL (bootstrapped from BC/GAIL policy)

Both variants need a BC checkpoint. It supplies the Peg1 network architecture, and
because `opponent_init_mode` is `from_pretrained` it also supplies the frozen
opponent -- so the random-init ablation needs it too. BC run directories are
timestamped, so train one first and pass its path:

```bash
python klask_imitation/scripts/training/train_BC_only.py --config configs/training/bc_baseline.json
# -> trained_models/klask_bc_<timestamp>_BC1/ppo_policy_bc_final.zip

python klask_imitation/scripts/training/train_GAIL_BC_sparse_RL.py --config configs/training/sparse_rl_bootstrap.json \
    --pretrained-policy-path trained_models/klask_bc_<timestamp>_BC1/ppo_policy_bc_final.zip
```

Random-init ablation:

```bash
python klask_imitation/scripts/training/train_GAIL_BC_sparse_RL.py --config configs/training/sparse_rl_random.json \
    --pretrained-policy-path trained_models/klask_bc_<timestamp>_BC1/ppo_policy_bc_final.zip
```

### BC+RL 2x2 with error bars (Figure 5)

The published Fig. 5 reports a single training run per condition. The camera-ready
figure reports five seeds per condition with 95% confidence intervals, every policy
scored against the same frozen PPO reference opponent (Elo 1283).

```bash
# 4 conditions x 5 seeds = 20 runs; ~70 min each, one run per GPU
# Pin one --run-id across all four stages: the runner otherwise names the sweep
# after today's date, and a sweep spanning midnight would be split in two.
RUN_ID=camera_ready_$(date +%Y%m%d)
python klask_imitation/scripts/experiments/run_bcrl_multiseed.py --stage train --gpus 0,1,2,3 --resume --run-id "$RUN_ID"
python klask_imitation/scripts/experiments/run_bcrl_multiseed.py --stage eval  --gpus 0,1,2,3 --resume --run-id "$RUN_ID"
python klask_imitation/scripts/experiments/run_bcrl_multiseed.py --stage aggregate --run-id "$RUN_ID"
python klask_imitation/scripts/plotting/make_fig5.py --run-id "$RUN_ID"
```

`run_bcrl_multiseed.py` shells out per job; set `ISAACLAB_ROOT`, and `CONDA_ENV`
if your Isaac Lab env is not named `env_isaaclab`. `make_fig5.py` defaults to the
most recent sweep when `--run-id` is omitted.

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

## Reproducing the paper

| Paper artifact | How to reproduce | Status |
|---|---|---|
| Table 1 (dataset tiers) | `klasktron-download-data --tier all` | reproducible |
| Sec. 5.2 BC hyperparameters | `train_BC_only.py --config configs/training/bc_baseline.json` | reproducible |
| Figure 2 (peg occupancy) | `evaluate_gail_results.py` writes `heatmap_agent_*.npy` | reproducible |
| Sec. 5.3 GAIL training | `train_gail_sweep.py --config configs/training/gail_baseline.json` | reproducible |
| Figure 4 (discriminator probe) | `run_gail_discriminator_analysis.py` | reproducible |
| Sec. 5.4 / Figure 5 (BC+RL 2x2) | `run_bcrl_multiseed.py` then `make_fig5.py` | reproducible |
| Appendix A / B hyperparameters | `configs/training/*.json` | reproducible |
| Actuator model, +-0.2 m/s identification range | deployed checkpoint ships in `klask_imitation/my_isaaclab_tasks/manager_based/klask/actuator_model/` (md5 `18812843f6f4cb45de2f12ebb879c7d6`) | artifact included; identification data and training code are not part of this release |
| Figure 3 (BC data scaling) | training side only, via `train_bc_sweep.py` | **not reproducible** |
| Table 2 / Table 3, BC and GAIL rows | retrain from the shipped configs | **not reproducible as published** |
| BC-5M hardware deployment | - | **not in this repository** |

### The frozen PPO expert

The PPO expert is inherited unchanged from the platform work and ships with this
repository at `klask_imitation/assets/ppo_expert/`, so evaluations against it run
without an external checkout. Override the location with `PPO_EXPERT_DIR` if you want
to point at your own copy.

`klask.pth` was saved from `cuda:2`; if you run under `CUDA_VISIBLE_DEVICES` masking it
will fail to deserialise, and `klask_cpu_mapped.pth` (identical weights) should be used
instead. See [the asset README](klask_imitation/assets/ppo_expert/README.md) for the
expert's 20-D observation interface and its `clip_actions: 0.2` command bound.

### What this repository cannot reproduce

- **Figure 3 (BC data scaling)** was produced on 2026-05-27, before this repository was
  assembled, and its run artifacts were not preserved. `train_bc_sweep.py` reproduces
  the training side; the published figure itself cannot be regenerated from this code.
- **The specific BC and GAIL checkpoints behind Tables 2 and 3** were not preserved.
  The shipped configs retrain equivalent policies, but per-seed numbers will differ from
  the published ones.
- **The BC-5M zero-shot hardware deployment** reported in the author response runs on the
  separate robot software stack and is outside this repository.
- Evaluations in the paper marked as self-play (Tables 2 and 3, BC and GAIL rows) were not
  re-run against the frozen PPO expert; the paper states this explicitly.

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
  assets/ppo_expert/        # frozen PPO expert (inherited, see [7])
  scripts/tests/play.py     # visualization / play script
data/                       # datasets (not in git)
trained_models/             # checkpoints (created at runtime)
artifacts/                  # eval outputs, plots (created at runtime)
docs/figures/               # camera-ready figures
```

## Citation

```bibtex
@inproceedings{klasktronil2026,
  title     = {KlaskTron-IL: A Contact-Rich Adversarial Imitation Learning
               Benchmark with a Validated Sim-to-Real Bridge},
  author    = {Sohm, Nicklas and Ramachandran, Aswin and D'Andrea, Raffaello},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```

The platform this benchmark builds on -- the hardware, the Isaac Lab environment, the
learned actuator model and the PPO expert, all inherited unchanged (see [NOTICE](NOTICE)):

```bibtex
@inproceedings{klasktron2026,
  title     = {KlaskTron: An Open-Source Platform for Physical Adversarial
               Multi-Agent RL},
  author    = {Ramachandran, Aswin and Schulz, Jona and Derungs, Maurus and
               Angelini, Carlo and Meier, Tobias and D'Andrea, Raffaello},
  booktitle = {Robotics: Science and Systems (RSS)},
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
