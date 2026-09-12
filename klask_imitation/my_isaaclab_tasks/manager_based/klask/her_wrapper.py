"""
HER (Hindsight Experience Replay) wrapper for Klask environment.

Converts the Klask env to GoalEnv format required by SB3's HER:
- observation: {"observation": ..., "achieved_goal": ..., "desired_goal": ...}
- compute_reward(achieved_goal, desired_goal, info) method

Goal for Klask: Score in opponent's goal (ball position in goal area)
"""

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper
from typing import Dict, Any, Optional, Tuple, List, Union

from .config.parameters import KLASK_PARAMS


class KlaskSelfPlayGoalWrapper(VecEnvWrapper):
    """
    Self-play wrapper for Klask with 4D action space (both pegs).

    Uses HER with the goal being to get the ball into the opponent's goal.
    Since this is self-play, we use the player's perspective for rewards:
    - Player scores (ball in opponent goal): +1
    - Player concedes (ball in player goal): -1 (or 0 for pure sparse)

    The policy learns to play both sides, which creates emergent competitive behavior.
    """

    def __init__(
        self,
        venv,
        sparse_reward: bool = True,
        goal_tolerance: float = None,
        reward_scale: float = 1.0,
        penalize_peg_in_goal: float = -0.5,  # Penalty for peg falling in goal
    ):
        """
        Args:
            venv: Underlying SB3 VecEnv
            sparse_reward: If True, use sparse reward (+1 for goal)
            goal_tolerance: Distance threshold for goal achievement
            reward_scale: Scale factor for rewards
            penalize_peg_in_goal: Penalty when any peg falls into a goal
        """
        orig_obs_space = venv.observation_space
        orig_obs_dim = orig_obs_space.shape[0]
        goal_dim = 2  # Ball (x, y)

        # GoalEnv observation space
        observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(orig_obs_dim,), dtype=np.float32
                ),
                "achieved_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
                ),
                "desired_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
                ),
            }
        )

        super().__init__(venv, observation_space=observation_space)

        self.sparse_reward = sparse_reward
        self.goal_dim = goal_dim
        self.orig_obs_dim = orig_obs_dim
        self.reward_scale = reward_scale
        self.penalize_peg_in_goal = penalize_peg_in_goal

        # Get goal parameters from KLASK_PARAMS
        scene = KLASK_PARAMS["scene"]

        # Player goal (bottom, y < 0)
        player_goal = scene["player_goal"]
        self.player_goal_center = np.array(
            [float(player_goal[0]), float(player_goal[1])], dtype=np.float32
        )
        self.player_goal_radius = (
            float(player_goal[2]) if len(player_goal) == 3 else 0.02
        )

        # Opponent goal (top, y > 0)
        opp_goal = scene["opponent_goal"]
        self.opponent_goal_center = np.array(
            [float(opp_goal[0]), float(opp_goal[1])], dtype=np.float32
        )
        self.opponent_goal_radius = float(opp_goal[2]) if len(opp_goal) == 3 else 0.02

        # Allow override via argument
        if goal_tolerance is not None:
            self.player_goal_radius = goal_tolerance
            self.opponent_goal_radius = goal_tolerance

        # For HER: desired goal is opponent's goal (player wants to score there)
        self.desired_goal_center = self.opponent_goal_center.copy()
        self.goal_radius = self.opponent_goal_radius

        print(f"✅ KlaskSelfPlayGoalWrapper initialized:")
        print(f"   Original obs dim: {orig_obs_dim}")
        print(f"   Action dim: {venv.action_space.shape[0]} (controlling both pegs)")
        print(
            f"   Player goal: ({self.player_goal_center[0]:.4f}, {self.player_goal_center[1]:.4f}), r={self.player_goal_radius:.4f}"
        )
        print(
            f"   Opponent goal: ({self.opponent_goal_center[0]:.4f}, {self.opponent_goal_center[1]:.4f}), r={self.opponent_goal_radius:.4f}"
        )
        print(f"   Sparse reward: {sparse_reward}")
        print(f"   Peg-in-goal penalty: {penalize_peg_in_goal}")

    def _extract_ball_pos(self, obs: np.ndarray) -> np.ndarray:
        """Extract ball (x, y) position from observation (indices 8, 9 based on your obs structure)."""
        # Based on your observation manager:
        # 0-1: peg_1_pos (player)
        # 2: peg_1_x_vel
        # 3: peg_1_y_vel
        # 4-5: peg_2_pos (opponent)
        # 6: peg_2_x_vel
        # 7: peg_2_y_vel
        # 8-11: ball_state (x, y, vx, vy)
        if obs.ndim == 1:
            return obs[8:10].copy()  # Ball x, y
        return obs[:, 8:10].copy()

    def _extract_peg_positions(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract both peg positions."""
        if obs.ndim == 1:
            peg1 = obs[0:2].copy()  # Player peg
            peg2 = obs[4:6].copy()  # Opponent peg
        else:
            peg1 = obs[:, 0:2].copy()
            peg2 = obs[:, 4:6].copy()
        return peg1, peg2

    def _make_goal_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert flat observation to GoalEnv dict format."""
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        achieved_goal = self._extract_ball_pos(obs)
        desired_goal = np.tile(self.desired_goal_center, (obs.shape[0], 1))

        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

    def _make_single_goal_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert a single flat observation to GoalEnv dict format."""
        if obs is None:
            return None

        if obs.ndim > 1:
            obs = obs.squeeze()

        achieved_goal = obs[8:10].copy()  # Ball position

        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": self.desired_goal_center.copy(),
        }

    def reset(self, **kwargs):
        obs = self.venv.reset(**kwargs)
        return self._make_goal_obs(obs)

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()

        goal_obs = self._make_goal_obs(obs)

        # Compute HER-compatible reward
        her_reward = self.compute_reward(
            goal_obs["achieved_goal"], goal_obs["desired_goal"], info
        )

        # Process info for each environment
        for i, inf in enumerate(info):
            # === ADD THESE LINES FOR CALLBACK METRICS ===
            achieved = goal_obs["achieved_goal"][i]
            desired = goal_obs["desired_goal"][i]
            dist = np.linalg.norm(achieved - desired)

            # HER metrics (for callback logging)
            inf["peg_goal_distance"] = float(dist)
            inf["peg_goal_reached"] = float(
                self._is_success(achieved.reshape(1, -1), desired.reshape(1, -1))[0]
            )
            inf["her_reward"] = float(her_reward[i])
            inf["game_reward"] = (
                0.0  # Set appropriately if you have separate game rewards
            )

            # Game outcome flags (for callback logging)
            ball_pos = self._extract_ball_pos(obs[i : i + 1])
            inf["goal_scored"] = bool(self._ball_in_opponent_goal(ball_pos)[0])
            inf["goal_conceded"] = bool(self._ball_in_player_goal(ball_pos)[0])
            # === END OF ADDITIONS ===

            # Check if ball scored in opponent goal (success for HER)
            inf["is_success"] = self._is_success(
                goal_obs["achieved_goal"][i : i + 1],
                goal_obs["desired_goal"][i : i + 1],
            )[0]

            # Convert terminal_observation for HER
            if (
                "terminal_observation" in inf
                and inf["terminal_observation"] is not None
            ):
                terminal_obs = inf["terminal_observation"]
                inf["terminal_observation"] = self._make_single_goal_obs(terminal_obs)

        return goal_obs, her_reward, done, info

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Any,
    ) -> np.ndarray:
        """
        Compute reward for HER with shaped rewards to encourage progress.
        """
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal.reshape(1, -1)
        if desired_goal.ndim == 1:
            desired_goal = desired_goal.reshape(1, -1)

        dist = np.linalg.norm(achieved_goal - desired_goal, axis=1)

        if self.sparse_reward:
            success = (dist <= self.goal_radius).astype(np.float32)
            return (success * self.reward_scale).astype(np.float32)
        else:
            # Dense reward: bonus for scoring + distance shaping
            success = (dist <= self.goal_radius).astype(np.float32)

            # Normalized distance (assuming max distance ~0.4m on board)
            max_dist = 0.4
            shaped_reward = -dist / max_dist  # Range: [-1, 0]

            # Big bonus for actually scoring
            shaped_reward += success * 10.0

            return (shaped_reward * self.reward_scale).astype(np.float32)

    def _is_success(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> np.ndarray:
        """Check if ball is within goal radius of desired goal."""
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal.reshape(1, -1)
        if desired_goal.ndim == 1:
            desired_goal = desired_goal.reshape(1, -1)

        dist = np.linalg.norm(achieved_goal - desired_goal, axis=1)
        return (dist <= self.goal_radius).astype(np.float32)

    def _ball_in_player_goal(self, ball_pos: np.ndarray) -> np.ndarray:
        """Check if ball is in player's goal (conceded)."""
        if ball_pos.ndim == 1:
            ball_pos = ball_pos.reshape(1, -1)
        dist = np.linalg.norm(ball_pos - self.player_goal_center, axis=1)
        return dist <= self.player_goal_radius

    def _ball_in_opponent_goal(self, ball_pos: np.ndarray) -> np.ndarray:
        """Check if ball is in opponent's goal (scored)."""
        if ball_pos.ndim == 1:
            ball_pos = ball_pos.reshape(1, -1)
        dist = np.linalg.norm(ball_pos - self.opponent_goal_center, axis=1)
        return dist <= self.opponent_goal_radius

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Union[None, List[int], int] = None,
        **method_kwargs,
    ) -> List[Any]:
        """Required for HER to call compute_reward during replay buffer sampling."""
        if method_name == "compute_reward":
            result = self.compute_reward(*method_args, **method_kwargs)
            return [result]
        return self.venv.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )


class KlaskGoalVecEnvWrapper(VecEnvWrapper):
    """
    Wraps a Klask VecEnv to provide GoalEnv-compatible observations for HER.

    Purpose:
    Wraps the klask environment for single-agent learning where the agent controls only one peg
    (2D action space). The opponent is either absent or externally controlled (by another policy).
    Goal: score in opponent's goal.

    Goal definition for Klask:
    - desired_goal: opponent's goal position (fixed target)
    - achieved_goal: current ball position

    Reward:
    - Sparse: +1 if ball in opponent goal, 0 otherwise
    - Dense (optional): negative distance to goal
    """

    def __init__(
        self,
        venv,
        sparse_reward: bool = True,
        goal_tolerance: float = None,  # If None, use goal radius from params
    ):
        """
        Args:
            venv: Underlying SB3 VecEnv (after Sb3VecEnvWrapper)
            sparse_reward: If True, use sparse reward (+1 for goal, 0 otherwise)
            goal_tolerance: Distance threshold for goal achievement (overrides param)
        """
        # Get original spaces before modifying
        orig_obs_space = venv.observation_space
        orig_obs_dim = orig_obs_space.shape[0]

        # Goal is 2D: (x, y) position of opponent's goal
        # Achieved goal is 2D: (x, y) position of ball
        goal_dim = 2

        # Build GoalEnv observation space
        observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(orig_obs_dim,), dtype=np.float32
                ),
                "achieved_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
                ),
                "desired_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
                ),
            }
        )

        super().__init__(venv, observation_space=observation_space)

        self.sparse_reward = sparse_reward
        self.goal_dim = goal_dim
        self.orig_obs_dim = orig_obs_dim

        # Get goal parameters from KLASK_PARAMS
        scene = KLASK_PARAMS["scene"]
        opp_goal = scene["opponent_goal"]

        # Parse opponent_goal based on format
        if len(opp_goal) == 2:
            self.goal_center_x = float(opp_goal[0])
            self.goal_center_y = float(opp_goal[1])
            self.goal_radius = scene.get("goal_radius", 0.02)
        elif len(opp_goal) == 3:
            # Format: (x_center, y_center, radius)
            self.goal_center_x = float(opp_goal[0])
            self.goal_center_y = float(opp_goal[1])
            self.goal_radius = float(opp_goal[2])
        elif len(opp_goal) == 4:
            self.goal_center_x = (opp_goal[0] + opp_goal[1]) / 2
            self.goal_center_y = (opp_goal[2] + opp_goal[3]) / 2
            dx = (opp_goal[1] - opp_goal[0]) / 2
            dy = (opp_goal[3] - opp_goal[2]) / 2
            self.goal_radius = max(dx, dy)
        else:
            raise ValueError(f"Unexpected opponent_goal format: {opp_goal}")

        # Allow override via argument
        if goal_tolerance is not None:
            self.goal_radius = goal_tolerance

        # Desired goal center (where we want the ball to go)
        self.desired_goal_center = np.array(
            [self.goal_center_x, self.goal_center_y],
            dtype=np.float32,
        )

        # Compute rectangular bounds from center and radius
        self.goal_x_min = self.goal_center_x - self.goal_radius
        self.goal_x_max = self.goal_center_x + self.goal_radius
        self.goal_y_min = self.goal_center_y - self.goal_radius
        self.goal_y_max = self.goal_center_y + self.goal_radius

        print(f"✅ KlaskGoalVecEnvWrapper initialized:")
        print(f"   Original obs dim: {orig_obs_dim}")
        print(f"   Goal center: ({self.goal_center_x:.4f}, {self.goal_center_y:.4f})")
        print(
            f"   Goal radius: {self.goal_radius:.4f}m ({self.goal_radius * 1000:.1f}mm)"
        )
        print(
            f"   Goal bounds: x=[{self.goal_x_min:.4f}, {self.goal_x_max:.4f}], "
            f"y=[{self.goal_y_min:.4f}, {self.goal_y_max:.4f}]"
        )
        print(f"   Sparse reward: {sparse_reward}")

    def _extract_ball_pos(self, obs: np.ndarray) -> np.ndarray:
        """
        Extract ball (x, y) position from observation.
        Ball position is indices 0, 1.
        """
        if obs.ndim == 1:
            return obs[:2].copy()
        return obs[:, :2].copy()

    def _make_goal_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert flat observation to GoalEnv dict format."""
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        achieved_goal = self._extract_ball_pos(obs)
        desired_goal = np.tile(self.desired_goal_center, (obs.shape[0], 1))

        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

    def _make_single_goal_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert a single flat observation to GoalEnv dict format (no batch dim)."""

        if obs is None:
            # Return None if input is None (can happen with terminal_observation)
            return None

        if obs.ndim > 1:
            obs = obs.squeeze()

        achieved_goal = obs[:2].copy()

        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": self.desired_goal_center.copy(),
        }

    def reset(self, **kwargs):
        obs = self.venv.reset(**kwargs)
        return self._make_goal_obs(obs)

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()

        goal_obs = self._make_goal_obs(obs)

        # Recompute reward based on goal achievement
        new_reward = self.compute_reward(
            goal_obs["achieved_goal"], goal_obs["desired_goal"], info
        )

        # Process info for each environment
        for i, inf in enumerate(info):
            # Add success flag
            inf["is_success"] = self._is_success(
                goal_obs["achieved_goal"][i : i + 1],
                goal_obs["desired_goal"][i : i + 1],
            )[0]

            # Forward termination reasons if available from underlying env
            # These come from IsaacLab's termination manager
            log = inf.get("log")
            if log is not None:
                inf["goal_conceded"] = log.get("goal_conceded", False)
                inf["player_in_goal"] = log.get("player_in_goal", False)
                inf["opponent_in_goal"] = log.get("opponent_in_goal", False)

            # CRITICAL: Convert terminal_observation to goal-dict format for HER
            if (
                "terminal_observation" in inf
                and inf["terminal_observation"] is not None
            ):
                terminal_obs = inf["terminal_observation"]
                inf["terminal_observation"] = self._make_single_goal_obs(terminal_obs)

        return goal_obs, new_reward, done, info

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Any,
    ) -> np.ndarray:
        """
        Compute reward for HER.
        Called by HER during relabeling with substituted goals.
        """
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal.reshape(1, -1)
        if desired_goal.ndim == 1:
            desired_goal = desired_goal.reshape(1, -1)

        if self.sparse_reward:
            success = self._is_success(achieved_goal, desired_goal)
            return success.astype(np.float32)
        else:
            dist = np.linalg.norm(achieved_goal - desired_goal, axis=1)
            return -dist.astype(np.float32)

    def _is_success(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> np.ndarray:
        """Check if achieved_goal is within goal_radius of desired_goal."""
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal.reshape(1, -1)
        if desired_goal.ndim == 1:
            desired_goal = desired_goal.reshape(1, -1)

        dist = np.linalg.norm(achieved_goal - desired_goal, axis=1)
        return (dist <= self.goal_radius).astype(np.float32)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Union[None, List[int], int] = None,
        **method_kwargs,
    ) -> List[Any]:
        """
        Call a method on the wrapper or underlying environment.
        Required for HER to call compute_reward during replay buffer sampling.
        """
        if method_name == "compute_reward":
            # HER calls compute_reward with batched goals
            result = self.compute_reward(*method_args, **method_kwargs)
            # Return the full result array as a single-element list
            return [result]

        # Forward other methods to underlying venv
        return self.venv.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    def _is_success_rectangular(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> np.ndarray:
        """Alternative: Check if ball is within rectangular goal bounds."""
        ball_x = achieved_goal[:, 0]
        ball_y = achieved_goal[:, 1]

        in_x = (ball_x >= self.goal_x_min) & (ball_x <= self.goal_x_max)
        in_y = (ball_y >= self.goal_y_min) & (ball_y <= self.goal_y_max)

        return (in_x & in_y).astype(np.float32)


class KlaskGoalVecEnvWrapperWithOpponent(VecEnvWrapper):
    """
    Purpose:
    - Agent controls one peg (2D action space)
    - Opponent actions are AUTOMATICALLY injected from a pre-trained policy
    - Combines functionality of OpponentVecEnvWrapper + KlaskGoalVecEnvWrapper
    - Provides HER-compatible goal-conditioned observations

    Comparison to Alternatives:
        - vs KlaskGoalVecEnvWrapper: Automatically handles opponent actions
        - vs KlaskSelfPlayGoalWrapper: Uses fixed opponent instead of self-play
        - vs OpponentVecEnvWrapper + KlaskGoalVecEnvWrapper: Single wrapper, less overhead


    """

    def __init__(
        self,
        venv,
        opponent_policy,
        deterministic_opponent: bool = True,
        sparse_reward: bool = True,
        goal_tolerance: float = None,
    ):
        # Get original spaces
        orig_obs_space = venv.observation_space
        orig_obs_dim = orig_obs_space.shape[0]
        goal_dim = 2

        # GoalEnv observation space
        observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(orig_obs_dim,), dtype=np.float32
                ),
                "achieved_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
                ),
                "desired_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
                ),
            }
        )

        # Player only controls 2D actions
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        super().__init__(
            venv, observation_space=observation_space, action_space=action_space
        )

        self.opponent_policy = opponent_policy
        self.deterministic_opponent = deterministic_opponent
        self.sparse_reward = sparse_reward
        self.goal_dim = goal_dim
        self.orig_obs_dim = orig_obs_dim

        self._last_flat_obs = None

        # Get goal parameters from KLASK_PARAMS
        scene = KLASK_PARAMS["scene"]
        opp_goal = scene["opponent_goal"]

        # Parse opponent_goal based on format
        if len(opp_goal) == 2:
            self.goal_center_x = float(opp_goal[0])
            self.goal_center_y = float(opp_goal[1])
            self.goal_radius = scene.get("goal_radius", 0.02)
        elif len(opp_goal) == 3:
            # Format: (x_center, y_center, radius)
            self.goal_center_x = float(opp_goal[0])
            self.goal_center_y = float(opp_goal[1])
            self.goal_radius = float(opp_goal[2])
        elif len(opp_goal) == 4:
            self.goal_center_x = (opp_goal[0] + opp_goal[1]) / 2
            self.goal_center_y = (opp_goal[2] + opp_goal[3]) / 2
            dx = (opp_goal[1] - opp_goal[0]) / 2
            dy = (opp_goal[3] - opp_goal[2]) / 2
            self.goal_radius = max(dx, dy)
        else:
            raise ValueError(f"Unexpected opponent_goal format: {opp_goal}")

        if goal_tolerance is not None:
            self.goal_radius = goal_tolerance

        self.desired_goal_center = np.array(
            [self.goal_center_x, self.goal_center_y],
            dtype=np.float32,
        )

        self.goal_x_min = self.goal_center_x - self.goal_radius
        self.goal_x_max = self.goal_center_x + self.goal_radius
        self.goal_y_min = self.goal_center_y - self.goal_radius
        self.goal_y_max = self.goal_center_y + self.goal_radius

        print(f"✅ KlaskGoalVecEnvWrapperWithOpponent initialized:")
        print(f"   Goal center: ({self.goal_center_x:.4f}, {self.goal_center_y:.4f})")
        print(
            f"   Goal radius: {self.goal_radius:.4f}m ({self.goal_radius * 1000:.1f}mm)"
        )
        print(f"   Sparse reward: {sparse_reward}")

    def _extract_ball_pos(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 1:
            return obs[:2].copy()
        return obs[:, :2].copy()

    def _make_goal_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        achieved_goal = self._extract_ball_pos(obs)
        desired_goal = np.tile(self.desired_goal_center, (obs.shape[0], 1))
        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

    def _make_single_goal_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert a single flat observation to GoalEnv dict format (no batch dim)."""
        if obs is None:
            # Return None if input is None (can happen with terminal_observation)
            return None

        if obs.ndim > 1:
            obs = obs.squeeze()

        achieved_goal = obs[:2].copy()

        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": self.desired_goal_center.copy(),
        }

    def reset(self, **kwargs):
        obs = self.venv.reset(**kwargs)
        self._last_flat_obs = obs
        return self._make_goal_obs(obs)

    def step_async(self, player_actions):
        # Get opponent actions
        opp_actions, _ = self.opponent_policy.predict(
            self._last_flat_obs, deterministic=self.deterministic_opponent
        )
        full_actions = np.concatenate([player_actions, opp_actions], axis=1)
        self.venv.step_async(full_actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        self._last_flat_obs = obs

        goal_obs = self._make_goal_obs(obs)

        new_reward = self.compute_reward(
            goal_obs["achieved_goal"], goal_obs["desired_goal"], info
        )

        for i, inf in enumerate(info):
            inf["is_success"] = self._is_success(
                goal_obs["achieved_goal"][i : i + 1],
                goal_obs["desired_goal"][i : i + 1],
            )[0]

            # Forward termination reasons if available from underlying env
            log = inf.get("log")
            if log is not None:
                inf["goal_conceded"] = log.get("goal_conceded", False)
                inf["player_in_goal"] = log.get("player_in_goal", False)
                inf["opponent_in_goal"] = log.get("opponent_in_goal", False)

            # CRITICAL: Convert terminal_observation to goal-dict format for HER
            if (
                "terminal_observation" in inf
                and inf["terminal_observation"] is not None
            ):
                terminal_obs = inf["terminal_observation"]
                inf["terminal_observation"] = self._make_single_goal_obs(terminal_obs)

        return goal_obs, new_reward, done, info

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Any,
    ) -> np.ndarray:
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal.reshape(1, -1)
        if desired_goal.ndim == 1:
            desired_goal = desired_goal.reshape(1, -1)

        if self.sparse_reward:
            success = self._is_success(achieved_goal, desired_goal)
            return success.astype(np.float32)
        else:
            dist = np.linalg.norm(achieved_goal - desired_goal, axis=1)
            return -dist.astype(np.float32)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Union[None, List[int], int] = None,
        **method_kwargs,
    ) -> List[Any]:
        """
        Call a method on the wrapper or underlying environment.
        Required for HER to call compute_reward during replay buffer sampling.
        """
        if method_name == "compute_reward":
            # HER calls compute_reward with batched goals
            result = self.compute_reward(*method_args, **method_kwargs)
            # Return the full result array as a single-element list
            return [result]

        # Forward other methods to underlying venv
        return self.venv.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    def _is_success(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> np.ndarray:
        """Check if achieved_goal is within goal_radius of desired_goal."""
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal.reshape(1, -1)
        if desired_goal.ndim == 1:
            desired_goal = desired_goal.reshape(1, -1)

        dist = np.linalg.norm(achieved_goal - desired_goal, axis=1)
        return (dist <= self.goal_radius).astype(np.float32)


class KlaskSelfPlayPegGoalWrapper(VecEnvWrapper):
    """
    Self-play wrapper where HER learns PEG control, not ball control.

    Key insight: HER should learn goals the agent DIRECTLY controls.
    - achieved_goal = player peg position (directly controlled)
    - desired_goal = random peg position in player's half
    - Sparse scoring reward layered on top for strategic learning

    HER learns: "How to move my peg to any position"
    Scoring reward learns: "Which positions lead to goals"
    """

    def __init__(
        self,
        venv,
        sparse_reward: bool = True,
        goal_tolerance: float = 0.015,  # ~15mm tolerance for reaching peg goal
        score_reward: float = 10.0,  # Large reward for actually scoring
        concede_penalty: float = -5.0,  # Penalty for ball in own goal
        peg_in_goal_penalty: float = -2.0,  # Penalty for peg falling in goal
    ):
        orig_obs_space = venv.observation_space
        orig_obs_dim = orig_obs_space.shape[0]
        goal_dim = 2  # Peg (x, y) position

        observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(orig_obs_dim,), dtype=np.float32
                ),
                "achieved_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
                ),
                "desired_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
                ),
            }
        )

        super().__init__(venv, observation_space=observation_space)

        self.sparse_reward = sparse_reward
        self.goal_dim = goal_dim
        self.orig_obs_dim = orig_obs_dim
        self.goal_tolerance = goal_tolerance
        self.score_reward = score_reward
        self.concede_penalty = concede_penalty
        self.peg_in_goal_penalty = peg_in_goal_penalty

        # Player's reachable area (own half of the board)
        # Adjust these based on your actual board dimensions
        scene = KLASK_PARAMS["scene"]
        self.peg_x_min = -0.12  # Board x limits
        self.peg_x_max = 0.12
        self.peg_y_min = -0.18  # Player's half (y < 0)
        self.peg_y_max = -0.02  # Don't go past center

        # Goal positions for scoring detection
        player_goal = scene["player_goal"]
        self.player_goal_center = np.array(
            [float(player_goal[0]), float(player_goal[1])], dtype=np.float32
        )
        self.player_goal_radius = (
            float(player_goal[2]) if len(player_goal) == 3 else 0.02
        )

        opp_goal = scene["opponent_goal"]
        self.opponent_goal_center = np.array(
            [float(opp_goal[0]), float(opp_goal[1])], dtype=np.float32
        )
        self.opponent_goal_radius = float(opp_goal[2]) if len(opp_goal) == 3 else 0.02

        # Current desired goals (per env)
        self._desired_goals = None

        print(f"✅ KlaskSelfPlayPegGoalWrapper initialized:")
        print(f"   Goal type: PEG POSITION (directly controlled)")
        print(
            f"   Goal tolerance: {goal_tolerance:.4f}m ({goal_tolerance * 1000:.1f}mm)"
        )
        print(
            f"   Peg bounds: x=[{self.peg_x_min:.3f}, {self.peg_x_max:.3f}], y=[{self.peg_y_min:.3f}, {self.peg_y_max:.3f}]"
        )
        print(f"   Score reward: {score_reward}, Concede penalty: {concede_penalty}")

    def _extract_player_peg_pos(self, obs: np.ndarray) -> np.ndarray:
        """Extract player peg (x, y) position from observation."""
        # Based on obs structure: indices 0-1 are player peg position
        if obs.ndim == 1:
            return obs[0:2].copy()
        return obs[:, 0:2].copy()

    def _extract_ball_pos(self, obs: np.ndarray) -> np.ndarray:
        """Extract ball position for scoring detection."""
        if obs.ndim == 1:
            return obs[8:10].copy()
        return obs[:, 8:10].copy()

    def _sample_desired_goals(self, num_envs: int) -> np.ndarray:
        """Sample random peg positions within player's reachable area."""
        goals = np.random.uniform(
            low=[self.peg_x_min, self.peg_y_min],
            high=[self.peg_x_max, self.peg_y_max],
            size=(num_envs, 2),
        ).astype(np.float32)
        return goals

    def _make_goal_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert flat observation to GoalEnv dict format."""
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        # Achieved goal = where the peg actually is
        achieved_goal = self._extract_player_peg_pos(obs)

        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": self._desired_goals.astype(np.float32),
        }

    def _make_single_goal_obs(
        self, obs: np.ndarray, env_idx: int
    ) -> Dict[str, np.ndarray]:
        """Convert a single flat observation to GoalEnv dict format."""
        if obs is None:
            return None

        if obs.ndim > 1:
            obs = obs.squeeze()

        achieved_goal = obs[0:2].copy()  # Player peg position

        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": self._desired_goals[env_idx].copy(),
        }

    def reset(self, **kwargs):
        obs = self.venv.reset(**kwargs)

        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        num_envs = obs.shape[0]

        # Sample new random peg goals for all envs
        self._desired_goals = self._sample_desired_goals(num_envs)

        return self._make_goal_obs(obs)

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()

        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        goal_obs = self._make_goal_obs(obs)

        # Compute HER reward (sparse: did we reach the peg goal?)
        her_reward = self.compute_reward(
            goal_obs["achieved_goal"], goal_obs["desired_goal"], info
        )

        # Add game-outcome rewards on top
        ball_pos = self._extract_ball_pos(obs)
        game_reward = self._compute_game_reward(ball_pos, info)

        total_reward = her_reward + game_reward

        # Process info for each environment
        for i, inf in enumerate(info):
            # Check if ball scored (ultimate success metric)
            ball_in_opp_goal = self._ball_in_goal(
                ball_pos[i : i + 1],
                self.opponent_goal_center,
                self.opponent_goal_radius,
            )[0]
            inf["is_success"] = ball_in_opp_goal
            inf["goal_scored"] = ball_in_opp_goal

            # Check if conceded
            ball_in_own_goal = self._ball_in_goal(
                ball_pos[i : i + 1], self.player_goal_center, self.player_goal_radius
            )[0]
            inf["goal_conceded"] = ball_in_own_goal

            # Convert terminal_observation for HER
            if (
                "terminal_observation" in inf
                and inf["terminal_observation"] is not None
            ):
                terminal_obs = inf["terminal_observation"]
                inf["terminal_observation"] = self._make_single_goal_obs(
                    terminal_obs, i
                )

            # Resample goal for this env if episode ended
            if done[i]:
                self._desired_goals[i] = self._sample_desired_goals(1)[0]

        return goal_obs, total_reward, done, info

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Any,
    ) -> np.ndarray:
        """
        Sparse reward for reaching the desired PEG position.
        This is what HER uses for relabeling.
        """
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal.reshape(1, -1)
        if desired_goal.ndim == 1:
            desired_goal = desired_goal.reshape(1, -1)

        dist = np.linalg.norm(achieved_goal - desired_goal, axis=1)

        if self.sparse_reward:
            # +1 if within tolerance, 0 otherwise
            return (dist <= self.goal_tolerance).astype(np.float32)
        else:
            # Dense: negative distance (normalized)
            max_dist = 0.3  # Approximate max peg travel distance
            return (-dist / max_dist).astype(np.float32)

    def _compute_game_reward(self, ball_pos: np.ndarray, info: Any) -> np.ndarray:
        """
        Compute game-outcome reward (NOT used by HER relabeling).
        This provides the strategic signal for which peg positions are valuable.
        """
        rewards = np.zeros(ball_pos.shape[0], dtype=np.float32)

        # Big reward for scoring
        scored = self._ball_in_goal(
            ball_pos, self.opponent_goal_center, self.opponent_goal_radius
        )
        rewards += scored * self.score_reward

        # Penalty for conceding
        conceded = self._ball_in_goal(
            ball_pos, self.player_goal_center, self.player_goal_radius
        )
        rewards += conceded * self.concede_penalty

        return rewards

    def _ball_in_goal(
        self, ball_pos: np.ndarray, goal_center: np.ndarray, goal_radius: float
    ) -> np.ndarray:
        """Check if ball is within goal radius."""
        if ball_pos.ndim == 1:
            ball_pos = ball_pos.reshape(1, -1)
        dist = np.linalg.norm(ball_pos - goal_center, axis=1)
        return (dist <= goal_radius).astype(np.float32)

    def _is_success(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> np.ndarray:
        """Check if peg reached desired position."""
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal.reshape(1, -1)
        if desired_goal.ndim == 1:
            desired_goal = desired_goal.reshape(1, -1)

        dist = np.linalg.norm(achieved_goal - desired_goal, axis=1)
        return (dist <= self.goal_tolerance).astype(np.float32)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Union[None, List[int], int] = None,
        **method_kwargs,
    ) -> List[Any]:
        if method_name == "compute_reward":
            result = self.compute_reward(*method_args, **method_kwargs)
            return [result]
        return self.venv.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )
