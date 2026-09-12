"""
test_policy.py

Simple script to load a trained SB3 PPO policy from one of the kept paper-scope
training runs and run it in a single Isaac/Omniverse environment so you can watch
the agent play.
Prints observation, action, reward, and step info for inspection.
"""

import os
import time
from pathlib import Path
from isaaclab.app import AppLauncher

# Launch Omniverse app immediately
app = AppLauncher(headless=False).app

import gymnasium as gym
import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from my_isaaclab_tasks.manager_based.klask.env_cfg import EnvCfg
from my_isaaclab_tasks.manager_based.klask.env_wrapper import (
    KlaskCollisionAvoidanceWrapper,
    ActionHistoryWrapper,
)
from my_isaaclab_tasks.manager_based.klask.actuator_model import ActuatorModelWrapper
from stable_baselines3 import PPO
from isaaclab_rl.sb3 import Sb3VecEnvWrapper


# >>> Added imports for non-blocking key read on Linux
import sys
import select
import termios
import tty


def build_env():
    """Construct a single env instance with same wrappers as training."""
    cfg = EnvCfg()
    cfg.scene.num_envs = 1
    ActuatorModelWrapper.DEBUG_ACTUATOR = True

    env = gym.make("Isaac-Klask-v0", cfg=cfg)
    env = ActuatorModelWrapper(env)
    # env = ActionHistoryWrapper(env, history_length=8)
    # env = KlaskCollisionAvoidanceWrapper(env, max_vel=1)
    env = Sb3VecEnvWrapper(env)

    return env


# >>> Helper for single-key non-blocking input (no Enter needed)
class NonBlockingKeyReader:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)  # cbreak mode: char-by-char
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def getch(self):
        # Return one character if available, else None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def rollout(model_path: str, episodes: int = 25, sleep: float = 0):
    env = build_env()

    print(f"Loading PPO policy from: {model_path}")
    device = (
        torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = PPO.load(model_path, device=device)

    print("Press 's' to skip the current episode (focus the terminal).")

    with NonBlockingKeyReader() as kb:
        for ep in range(1, episodes + 1):
            obs = env.reset()
            done = np.array([False])
            ep_rew = 0.0
            step = 0

            print(f"\n=== Starting Episode {ep} ===")
            while not done[0]:
                # Check for skip key
                ch = kb.getch()
                if ch in ("s", "S"):
                    print("↷ Skip requested: ending episode early.")
                    break

                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                ep_rew += reward[0]
                step += 1

                # Print detailed rollout info
                print(
                    f"[Step {step:3d}] obs={obs[0][:4]} action={action[0]} reward={reward[0]:.4f} done={done[0]}"
                )

                # Optional: slow down to watch
                time.sleep(sleep)

    print("✅ Rollout complete. Keeping Omniverse app open. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)  # keep app alive and responsive
    except KeyboardInterrupt:
        print("\n🛑 Exiting simulation...")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    trained_models_dir = Path(
        os.environ.get("KLASKTRON_TRAINED_MODELS_DIR", repo_root / "trained_models")
    )
    model_path_default = (
        trained_models_dir
        / "data_wall_experiments"
        / "bc_runs"
        / "klask_bc_20260303_161251_HUMAN_DEMO_V2"
        / "ppo_policy_bc_final.zip"
    )
    MODEL_PATH = os.environ.get("KLASKTRON_POLICY_PATH", str(model_path_default))

    # Check if file exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("Please update MODEL_PATH in the script.")
        exit(1)

    print(f"🎮 Testing policy: {MODEL_PATH}")
    rollout(
        model_path=MODEL_PATH,
        episodes=25,  # Number of episodes to run
        sleep=0,  # Seconds to sleep between steps (0 = as fast as possible)
    )
