# Frozen PPO expert

The PPO self-play agent that generates the Tier 3 synthetic demonstrations and serves
as the fixed reference opponent for all closed-loop evaluations.

This agent is **inherited unchanged from the platform work** (reference [7] in the
paper); it is redistributed here so that the evaluation scripts in this repository run
without an external checkout.

## Files

| File | md5 | Notes |
|---|---|---|
| `klask.pth` | `b06db3851b0607b1789654f475a3c77b` | The checkpoint the scripts load by default. |
| `klask_cpu_mapped.pth` | `28187e5f151061fb198024bcd8d5f09a` | Same weights, re-saved with CPU storage tags. |
| `agent.yaml` | - | rl_games config. `clip_actions: 0.2` is the expert's action bound. |

## Interface

The expert consumes a **20-dimensional** observation: the 12-D benchmark state plus
eight engineered geometric features (striker-goal angles, ball-goal and ball-striker
distances). The IL baselines in this repository use the 12-D state only. Evaluation
code reconstructs the extra eight channels when driving the expert, which is why the
opponent network is built from the checkpoint's own dimensions rather than from the
environment's observation space.

Its physical command bound is `clip_actions: 0.2` from `agent.yaml`: rl_games emits a
normalised action and multiplies by that value, so the expert commands at most
+-0.2 m/s per axis. Any harness that bypasses the rl_games action path must reapply
this scaling itself.

## Loading on a different GPU layout

`klask.pth` was saved from `cuda:2`. Loading it in a process that cannot see that
device (for example under `CUDA_VISIBLE_DEVICES=0`) raises

```
Attempting to deserialize object on CUDA device 2 but torch.cuda.device_count() is 1
```

Use `klask_cpu_mapped.pth` in that case, or make all devices visible and select the
target device with the script's `--device` flag instead of masking.
