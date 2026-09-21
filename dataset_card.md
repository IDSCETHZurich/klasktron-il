# KlaskTron Dataset Card

## Dataset Summary

KlaskTron is a three-tier imitation-learning benchmark for the Klask tabletop game. The dataset is designed for benchmarking policy learning, adversarial imitation learning, and data-scaling behavior in a physically grounded Isaac Sim environment.

The benchmark exposes three training regimes:

- `human`: about 80k steps from real human play
- `human_augmented`: about 325k steps after augmentation of the human set (selected as `--tier augmented`)
- `synthetic`: 5M+ steps generated from synthetic expert rollouts in simulation

## Dataset Repository Layout

The intended Hugging Face dataset layout is:

```text
klasktron/
├── human/
│   ├── trajectories/
│   └── metadata.json
├── human_augmented/
│   ├── trajectories/
│   └── metadata.json
├── synthetic/
│   ├── trajectories/
│   └── metadata.json
├── docs/
│   └── schema.md
└── dataset_card.md
```

## Data Instances

Each stored trajectory is expected to serialize one or more episode records with the same layout already used in the training code:

- `obs`: float array with shape `[T + 1, 12]`
- `acts`: float array with shape `[T, 4]`
- `infos`: optional per-step metadata
- `terminal`: whether the trajectory terminates naturally

Observation layout in simulation coordinates:

```text
[ball_x, ball_y,
 ball_vx, ball_vy,
 peg1_x, peg1_y,
 peg2_x, peg2_y,
 peg1_vx, peg1_vy,
 peg2_vx, peg2_vy]
```

Action layout:

```text
[peg1_vx, peg1_vy, peg2_vx, peg2_vy]
```

## Data Fields

Recommended metadata fields per trajectory file or manifest entry:

- `tier`: one of `human`, `human_augmented`, `synthetic`
- `source_format`: one of `human_raw`, `sim_native`
- `collection_id`: session, run, or generator identifier
- `episode_count`: number of trajectories stored in the file
- `step_count`: number of action steps in the file
- `frame_rate_hz`: nominal control frequency
- `notes`: optional human-readable comments

## Collection Process

### Human Tier

- Recorded from real human Klask play
- Converted into benchmark trajectories through the external state-reestimation pipeline
- Manually cleaned before inclusion in the benchmark release

### Augmented Tier

- Derived from the human tier
- Includes symmetry-based augmentation and related benchmark preprocessing
- Intended to measure how far classical augmentation extends limited real data

### Synthetic Tier

- Generated in Isaac Sim / Isaac Lab from expert or expert-like policies
- Intended for large-scale imitation learning and data-scaling experiments
- Exported in the same simulation-coordinate schema used by the baselines

## Intended Uses

- Benchmarking BC and adversarial imitation learning
- Studying the effect of human versus synthetic demonstrations
- Evaluating transfer from simulated demonstrations to a standardized game environment

## Out-Of-Scope Uses

- Safety-critical robotics deployment
- Human-subject inference or identity analysis
- Claims about real-world skill outside the Klask benchmark protocol

## Known Limitations

- The human state-reestimation pipeline is maintained in a separate repository
- Observation quality depends on the reestimation and cleaning pipeline for the human tier
- Synthetic data can encode simulator and policy biases that do not occur in human play
- Exact split policies and release checksums should be finalized at dataset publication time

## Maintenance

- GitHub code repository: <https://github.com/IDSCETHZurich/klasktron-il>
- Hugging Face dataset repository: <https://huggingface.co/datasets/KlaskLab/klasktron-il-benchmark>
- Licence: dataset CC-BY-4.0; code Apache-2.0
- Contact: `FILL IN CONTACT`
