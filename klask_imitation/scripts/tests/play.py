"""Visualize a trained policy playing Klask in the Isaac Lab simulator.

Supports self-play (same policy controls both sides) or play against a fixed
opponent (e.g. the PPO expert agent in RL-Games format).

Usage examples:

  # Self-play with rendering (SB3 checkpoint)
  python klask_imitation/scripts/tests/play.py --checkpoint <policy.zip> --num-envs 1

  # Vs fixed RL-Games opponent
  python klask_imitation/scripts/tests/play.py --checkpoint <policy.zip> --opponent-policy-path <opp.pth> --num-envs 1

  # Headless (no GUI)
  python klask_imitation/scripts/tests/play.py --checkpoint <policy.zip> --headless --num-envs 16 --episodes 100
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Isaac Lab app must be launched before any other Isaac imports.
# We parse args early so --headless / --num-envs are available.
# ---------------------------------------------------------------------------


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize a trained Klask policy in Isaac Lab.",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    policy_group = parser.add_argument_group("Policy")
    policy_group.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to policy checkpoint (.zip for SB3, .pth for RL-Games).",
    )
    policy_group.add_argument("--policy-format", type=str, default=None)
    policy_group.add_argument("--rl-games-config-path", type=str, default=None)

    opponent_group = parser.add_argument_group("Opponent")
    opponent_group.add_argument(
        "--opponent-policy-path", type=str, default=None,
        help="Path to opponent checkpoint. If omitted, self-play is used.",
    )
    opponent_group.add_argument("--opponent-policy-format", type=str, default=None)
    opponent_group.add_argument("--opponent-rl-games-config-path", type=str, default=None)
    return parser


args = build_arg_parser().parse_args()

# Launch Isaac Lab app
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Now safe to import Isaac / gym / torch
# ---------------------------------------------------------------------------
import gymnasium as gym
import numpy as np
import torch
import yaml

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import my_isaaclab_tasks.manager_based.klask  # registers Isaac-Klask-v0

from klasktron.rl_games_obs import (  # noqa: E402
    _augment_local_obs_to_expected_dim,
    _build_env_info_for_dims,
    _infer_action_dim_from_rl_games_checkpoint,
    _infer_obs_dim_from_rl_games_checkpoint,
)
from my_isaaclab_tasks.manager_based.klask.env_cfg import EnvCfg
from my_isaaclab_tasks.manager_based.klask.env_wrapper import (
    ActionHistoryWrapper,
    KlaskAgentOpponentWrapper,
    KlaskCollisionAvoidanceWrapper,
    KlaskRandomOpponentWrapper,
    RlGamesGpuEnvSelfPlay,
    find_wrapper,
)
from my_isaaclab_tasks.manager_based.klask.actuator_model import ActuatorModelWrapper
from my_isaaclab_tasks.manager_based.klask.config.parameters import KLASK_PARAMS


POLICY_FORMAT_RL_GAMES = "rl_games"
POLICY_FORMAT_SB3_PPO = "sb3_ppo"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def ensure_2d(arr: Any, *, dtype: Any = None) -> np.ndarray:
    if dtype is None:
        dtype = np.float32
    out = to_numpy(arr).astype(dtype, copy=False)
    if out.ndim == 1:
        out = out[None, :]
    return out


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

class Sb3PpoPolicyAdapter:
    """Wrap an SB3 PPO model to expose the same interface as RL-Games players."""

    def __init__(self, model: Any, *, device: str, action_dim: Optional[int] = None):
        self.model = model
        self.device = torch.device(device)
        self.is_rnn = False
        self.states = None
        self.has_batch_dimension = True
        self.action_dim = action_dim
        self.obs_dim = None
        space = getattr(model, "observation_space", None)
        shape = getattr(space, "shape", None)
        if shape is not None and len(shape) > 0:
            self.obs_dim = int(shape[0])
        if self.action_dim is None:
            space = getattr(model, "action_space", None)
            shape = getattr(space, "shape", None)
            if shape is not None and len(shape) > 0:
                self.action_dim = int(shape[0])

    def reset(self) -> None:
        return

    def obs_to_torch(self, obs: Any) -> np.ndarray:
        obs_np = ensure_2d(obs, dtype=np.float32)
        if self.obs_dim is not None and obs_np.shape[1] > int(self.obs_dim):
            obs_np = obs_np[:, : int(self.obs_dim)]
        return obs_np

    def get_action(self, obs: Any, is_deterministic: bool = True) -> torch.Tensor:
        obs_np = ensure_2d(obs, dtype=np.float32)
        action_np, _ = self.model.predict(obs_np, deterministic=bool(is_deterministic))
        action_np = ensure_2d(action_np, dtype=np.float32)
        if self.action_dim is not None and action_np.shape[1] > int(self.action_dim):
            action_np = action_np[:, : int(self.action_dim)]
        return torch.as_tensor(action_np, dtype=torch.float32, device=self.device)


def resolve_policy_format(raw_fmt: Optional[str], checkpoint: str) -> str:
    if raw_fmt:
        fmt = raw_fmt.strip().lower()
        if fmt in {"sb3", "sb3_ppo", "ppo"}:
            return POLICY_FORMAT_SB3_PPO
        if fmt in {"rl_games", "rlgames", "rl"}:
            return POLICY_FORMAT_RL_GAMES
        raise ValueError(f"Unsupported --policy-format {raw_fmt!r}.")
    if checkpoint.lower().endswith(".zip"):
        inferred = POLICY_FORMAT_SB3_PPO
    else:
        inferred = POLICY_FORMAT_RL_GAMES
    print(f"[setup] inferred policy_format={inferred} from checkpoint path")
    return inferred


class RlGamesObsAdapter:
    """Feed an RL-Games player the observation width its checkpoint expects.

    The Klask env emits the 12-D benchmark state, but a checkpoint may have been
    trained on a wider observation -- the bundled PPO expert uses 20 dims: the
    12-D state plus eight engineered geometric channels. Rebuilding those eight
    is the only correct way to drive it; padding with zeros, or substituting a
    different 8-wide signal such as an action history, matches the shape while
    feeding the network values it never saw in training.
    """

    def __init__(
        self, player: Any, *, expected_obs_dim: int, action_dim: int, device: str
    ):
        self._player = player
        self.expected_obs_dim = int(expected_obs_dim)
        self.action_dim = int(action_dim)
        # The env and the player must agree on a device: rl_games can place a
        # restored player on its own default, which then fails when the action
        # is concatenated with the learner's on a different GPU.
        self.device = torch.device(device)

    def __getattr__(self, name: str) -> Any:
        # Everything not overridden here (reset, is_rnn, init_rnn,
        # has_batch_dimension, ...) delegates to the wrapped player.
        return getattr(self._player, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_player", "expected_obs_dim", "action_dim", "device"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._player, name, value)

    def obs_to_torch(self, obs: Any) -> torch.Tensor:
        obs_np = _augment_local_obs_to_expected_dim(
            ensure_2d(obs, dtype=np.float32), self.expected_obs_dim
        )
        return self._player.obs_to_torch(
            torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        )

    def get_action(self, obs: Any, is_deterministic: bool = True) -> torch.Tensor:
        acts = self._player.get_action(obs, is_deterministic=bool(is_deterministic))
        if not torch.is_tensor(acts):
            acts = torch.as_tensor(acts, dtype=torch.float32)
        if acts.ndim == 1:
            acts = acts.unsqueeze(0)
        return acts[:, : max(1, self.action_dim)].to(self.device)


def build_runner_player(
    *,
    agent_cfg: Dict[str, Any],
    checkpoint: str,
    num_envs: int,
    device: str,
) -> Tuple[Runner, BasePlayer]:
    agent_cfg = dict(agent_cfg)
    agent_cfg["params"] = dict(agent_cfg.get("params", {}))
    agent_cfg["params"]["config"] = dict(agent_cfg["params"].get("config", {}))
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint
    agent_cfg["params"]["config"]["num_actors"] = int(num_envs)
    agent_cfg["params"]["config"]["device"] = device
    if "device_name" in agent_cfg["params"]["config"]:
        agent_cfg["params"]["config"]["device_name"] = device

    # Size the network from the checkpoint rather than from the env. The env
    # reports 12 observations; a checkpoint such as the bundled PPO expert was
    # trained on 20. Building from the env would make restore() fail on a shape
    # mismatch, which is what used to make the expert unloadable here.
    ckpt_obs_dim = None
    ckpt_action_dim = None
    try:
        ckpt_obs_dim = int(_infer_obs_dim_from_rl_games_checkpoint(checkpoint))
        ckpt_action_dim = int(_infer_action_dim_from_rl_games_checkpoint(checkpoint))
        agent_cfg["params"]["config"]["env_info"] = _build_env_info_for_dims(
            observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(ckpt_obs_dim,), dtype=np.float32
            ),
            action_space=gym.spaces.Box(
                low=-1.0, high=1.0, shape=(ckpt_action_dim,), dtype=np.float32
            ),
            action_dim=ckpt_action_dim,
            obs_dim=ckpt_obs_dim,
        )
        agent_cfg["params"]["config"]["vec_env"] = None
        print(
            f"[setup] rl_games checkpoint expects obs_dim={ckpt_obs_dim}, "
            f"action_dim={ckpt_action_dim}"
        )
    except Exception as exc:
        print(f"[warn] could not infer checkpoint dims ({exc}); using env dims.")
        ckpt_obs_dim = None
        ckpt_action_dim = None

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    try:
        agent.restore(checkpoint)
    except Exception as exc:
        print(f"[warn] restore() failed ({exc}). Falling back to direct torch.load.")
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        if "model" not in ckpt:
            raise RuntimeError(f"Checkpoint missing 'model' key: {checkpoint}")
        agent.model.load_state_dict(ckpt["model"])
        if getattr(agent, "normalize_input", False) and "running_mean_std" in ckpt:
            agent.model.running_mean_std.load_state_dict(ckpt["running_mean_std"])
    agent.reset()
    agent.device = torch.device(device)
    agent.model.to(device)
    if ckpt_action_dim is not None:
        agent.actions_low = agent.actions_low[:ckpt_action_dim].to(device)
        agent.actions_high = agent.actions_high[:ckpt_action_dim].to(device)
    else:
        agent.actions_low = agent.actions_low.to(device)
        agent.actions_high = agent.actions_high.to(device)
    if ckpt_obs_dim is not None:
        # One explicit batch-size probe so batched observations are not treated
        # as a single unbatched sample.
        probe = torch.zeros(
            (max(1, int(num_envs)), int(ckpt_obs_dim)),
            dtype=torch.float32,
            device=agent.device,
        )
        agent.get_batch_size(probe, 1)
        agent = RlGamesObsAdapter(
            agent,
            expected_obs_dim=int(ckpt_obs_dim),
            action_dim=int(ckpt_action_dim),
            device=device,
        )
    return runner, agent


def build_sb3_player(*, checkpoint: str, device: str, action_dim: Optional[int] = None) -> Sb3PpoPolicyAdapter:
    from stable_baselines3 import PPO
    model = PPO.load(checkpoint, device=device)
    return Sb3PpoPolicyAdapter(model, device=device, action_dim=action_dim)


def load_rl_games_cfg(config_path: Optional[str]) -> Dict[str, Any]:
    if config_path is not None:
        with open(config_path, "r") as fh:
            cfg = yaml.safe_load(fh)
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid RL-Games config at {config_path}.")
        return cfg
    # Fall back to the bundled agent config
    agents_dir = Path(__file__).resolve().parents[2] / "my_isaaclab_tasks" / "manager_based" / "klask" / "agents"
    default_cfg = agents_dir / "rl_games_ppo_cfg.yaml"
    if default_cfg.exists():
        with open(default_cfg, "r") as fh:
            return yaml.safe_load(fh)
    raise FileNotFoundError(
        "No RL-Games config found. Pass --rl-games-config-path explicitly."
    )


def build_policy_player(
    *,
    policy_format: str,
    checkpoint: str,
    device: str,
    num_envs: int,
    action_dim: Optional[int],
    rl_games_config_path: Optional[str],
) -> Any:
    if policy_format == POLICY_FORMAT_SB3_PPO:
        return build_sb3_player(checkpoint=checkpoint, device=device, action_dim=action_dim)
    if policy_format == POLICY_FORMAT_RL_GAMES:
        rl_cfg = load_rl_games_cfg(rl_games_config_path)
        _, player = build_runner_player(
            agent_cfg=rl_cfg, checkpoint=checkpoint, num_envs=num_envs, device=device,
        )
        return player
    raise ValueError(f"Unsupported policy_format={policy_format!r}")


# ---------------------------------------------------------------------------
# Environment construction (follows test_policy.py pattern: direct EnvCfg)
# ---------------------------------------------------------------------------

def build_env(
    *,
    num_envs: int,
    device: str,
    headless: bool,
) -> Any:
    cfg = EnvCfg()
    cfg.scene.num_envs = num_envs

    env = gym.make("Isaac-Klask-v0", cfg=cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = ActuatorModelWrapper(env, device=device)

    if KLASK_PARAMS.get("action_history", 0) > 0:
        env = ActionHistoryWrapper(env, history_length=KLASK_PARAMS["action_history"])

    # Always use self-play wrapper so we can attach an opponent policy
    env = KlaskAgentOpponentWrapper(env)
    if hasattr(env.unwrapped, "single_action_space"):
        original_space = env.unwrapped.single_action_space
        if hasattr(original_space, "shape") and int(original_space.shape[0]) == 4:
            env.unwrapped._klask_original_single_action_space = original_space
            player_space = gym.spaces.Box(
                low=original_space.low[:2],
                high=original_space.high[:2],
                dtype=original_space.dtype,
            )
            env.unwrapped.single_action_space = player_space
            try:
                env.action_space = player_space
            except Exception:
                pass

    clip_obs = float(np.inf)
    clip_actions = float(np.inf)
    rl_env = RlGamesVecEnvWrapper(env, device, clip_obs=clip_obs, clip_actions=clip_actions)

    try:
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnvSelfPlay(
                config_name, num_actors, {}, **kwargs
            ),
        )
    except Exception:
        pass
    env_configurations.configurations["rlgpu"] = {
        "vecenv_type": "IsaacRlgWrapper",
        "env_creator": lambda **kwargs: rl_env,
    }
    return rl_env


# ---------------------------------------------------------------------------
# Main play loop
# ---------------------------------------------------------------------------

def play(
    *,
    env: Any,
    agent: Any,
    episodes: int,
    deterministic: bool,
) -> None:
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    obs_arr = ensure_2d(obs)
    num_envs = int(obs_arr.shape[0])

    if hasattr(agent, "get_batch_size"):
        _ = agent.get_batch_size(obs, 1)
    if getattr(agent, "is_rnn", False) and hasattr(agent, "init_rnn"):
        agent.init_rnn()

    self_play_wrapper = None
    try:
        self_play_wrapper = find_wrapper(env.env, KlaskAgentOpponentWrapper)
    except Exception:
        self_play_wrapper = None

    ep_reward = np.zeros((num_envs,), dtype=np.float64)
    ep_len = np.zeros((num_envs,), dtype=np.int64)
    episodes_completed = 0
    started = time.perf_counter()

    print(f"[play] running {episodes} episodes (num_envs={num_envs}, deterministic={deterministic})")

    while simulation_app.is_running() and episodes_completed < int(episodes):
        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs)
            action = agent.get_action(obs_t, is_deterministic=bool(deterministic))
            action = torch.clamp(action, -1.0, 1.0)

            next_obs, reward, dones, infos = env.step(action)
            if isinstance(next_obs, dict):
                next_obs = next_obs["obs"]

            done_np = to_numpy(dones).astype(bool).reshape(-1)
            rew_np = to_numpy(reward).astype(np.float64).reshape(-1)

            ep_reward += rew_np
            ep_len += 1

            if np.any(done_np):
                done_idx = np.where(done_np)[0]
                for i in done_idx:
                    episodes_completed += 1
                    print(
                        f"  episode {episodes_completed}/{episodes}: "
                        f"reward={ep_reward[i]:.3f}, length={ep_len[i]}"
                    )
                    if episodes_completed >= int(episodes):
                        break

                ep_reward[done_idx] = 0.0
                ep_len[done_idx] = 0

            # Reset RNN states on episode boundaries
            if getattr(agent, "is_rnn", False) and getattr(agent, "states", None) is not None:
                done_t = torch.as_tensor(done_np, dtype=torch.bool, device=agent.device)
                for s in agent.states:
                    s[:, done_t, :] = 0.0

            if self_play_wrapper is not None and self_play_wrapper.opponent is not None:
                opp = self_play_wrapper.opponent
                if getattr(opp, "is_rnn", False) and getattr(opp, "states", None) is not None:
                    done_t = torch.as_tensor(done_np, dtype=torch.bool, device=opp.device)
                    for s in opp.states:
                        s[:, done_t, :] = 0.0

            obs = next_obs

    elapsed = time.perf_counter() - started
    print(f"[play] done. {episodes_completed} episodes in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = args.device
    headless = args.headless

    checkpoint = str(args.checkpoint)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    policy_format = resolve_policy_format(args.policy_format, checkpoint)

    opponent_checkpoint = args.opponent_policy_path
    opponent_policy_format = None
    if opponent_checkpoint is not None:
        if not os.path.exists(opponent_checkpoint):
            raise FileNotFoundError(f"Opponent policy not found: {opponent_checkpoint}")
        opponent_policy_format = resolve_policy_format(args.opponent_policy_format, opponent_checkpoint)

    env = build_env(
        num_envs=int(args.num_envs),
        device=device,
        headless=headless,
    )

    runner: Optional[Runner] = None
    try:
        sb3_action_dim = None
        try:
            space = getattr(env, "action_space", None)
            shape = getattr(space, "shape", None)
            if shape is not None and len(shape) > 0:
                sb3_action_dim = int(shape[0])
        except Exception:
            pass

        print(f"[setup] loading policy: {checkpoint}")
        if policy_format == POLICY_FORMAT_RL_GAMES:
            rl_cfg = load_rl_games_cfg(args.rl_games_config_path)
            runner, agent = build_runner_player(
                agent_cfg=rl_cfg,
                checkpoint=checkpoint,
                num_envs=int(args.num_envs),
                device=device,
            )
        else:
            agent = build_sb3_player(
                checkpoint=checkpoint, device=device, action_dim=sb3_action_dim,
            )

        # Set up opponent
        if opponent_checkpoint is not None:
            print(f"[setup] loading fixed opponent: {opponent_checkpoint}")
            opponent = build_policy_player(
                policy_format=str(opponent_policy_format),
                checkpoint=opponent_checkpoint,
                device=device,
                num_envs=int(args.num_envs),
                action_dim=sb3_action_dim,
                rl_games_config_path=args.opponent_rl_games_config_path,
            )
        else:
            # Self-play: clone weights from the main policy
            print("[setup] no opponent provided; using self-play (same policy on both sides).")
            if policy_format == POLICY_FORMAT_RL_GAMES:
                if runner is None:
                    raise RuntimeError("RL-Games runner not initialized.")
                opponent = runner.create_player()
                opponent.device = torch.device(device)
                opponent.model.to(device)
                opponent.actions_low = agent.actions_low.to(device)
                opponent.actions_high = agent.actions_high.to(device)
                opponent.set_weights(agent.get_weights())
            else:
                opponent = build_sb3_player(
                    checkpoint=checkpoint, device=device, action_dim=sb3_action_dim,
                )

        self_play_wrapper = find_wrapper(env.env, KlaskAgentOpponentWrapper)
        self_play_wrapper.is_deterministic = bool(args.deterministic)
        self_play_wrapper.add_opponent(opponent)
        print(f"[setup] opponent attached (deterministic={bool(args.deterministic)})")

        play(
            env=env,
            agent=agent,
            episodes=int(args.episodes),
            deterministic=bool(args.deterministic),
        )
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Isaac Sim's shutdown path swallows the process exit status and can drop
    # buffered stdout, so flush and exit explicitly.
    import os as _os
    import sys as _sys
    import traceback as _tb

    try:
        main()
    except BaseException as exc:
        print(f"[error] {exc}")
        _tb.print_exc()
        # Flush and exit here rather than falling through to
        # simulation_app.close(): Isaac's shutdown terminates the process
        # itself, with status 0, which would report this failure as success
        # and would discard whatever is still in Python's stdout buffer.
        _sys.stdout.flush()
        _sys.stderr.flush()
        _os._exit(1)

    _sys.stdout.flush()
    _sys.stderr.flush()
    try:
        simulation_app.close()
    except Exception:
        pass
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(0)
