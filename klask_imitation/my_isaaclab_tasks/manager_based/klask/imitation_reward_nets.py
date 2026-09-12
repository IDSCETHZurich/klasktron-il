from __future__ import annotations

from typing import Optional, Sequence, Type

import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn

from imitation.rewards.reward_nets import RewardNet, ShapedRewardNet
from imitation.util import networks


def _safe_norm(vec: th.Tensor, eps: float = 1e-6) -> th.Tensor:
    return th.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(eps)


def _safe_unit(vec: th.Tensor, eps: float = 1e-6) -> th.Tensor:
    return vec / _safe_norm(vec, eps=eps)


def _safe_cos(a: th.Tensor, b: th.Tensor, eps: float = 1e-6) -> th.Tensor:
    return th.sum(_safe_unit(a, eps=eps) * _safe_unit(b, eps=eps), dim=-1, keepdim=True)


class _KlaskRelativeFeatureMixin:
    """Build relative/tactical features from the 12D Klask observation layout."""

    state_feature_dim = 22
    transition_feature_dim = 32

    def __init__(
        self,
        *,
        observation_space: gym.Space,
        action_space: gym.Space,
    ) -> None:
        if not isinstance(observation_space, spaces.Box):
            raise TypeError(
                "KlaskRelativeShapedRewardNet expects Box observations, "
                f"got {type(observation_space).__name__}."
            )
        if not isinstance(action_space, spaces.Box):
            raise TypeError(
                "KlaskRelativeShapedRewardNet expects Box actions, "
                f"got {type(action_space).__name__}."
            )
        if len(observation_space.shape) != 1 or int(observation_space.shape[0]) < 12:
            raise ValueError(
                "KlaskRelativeShapedRewardNet expects flat observations with at least 12 dims, "
                f"got shape={observation_space.shape}."
            )
        if len(action_space.shape) != 1 or int(action_space.shape[0]) < 2:
            raise ValueError(
                "KlaskRelativeShapedRewardNet expects flat actions with at least 2 dims, "
                f"got shape={action_space.shape}."
            )
        self._obs_dim = int(observation_space.shape[0])
        self._action_dim = int(action_space.shape[0])

    @staticmethod
    def _goal_tensor(
        goal_xy: tuple[float, float],
        *,
        device: th.device,
        dtype: th.dtype,
    ) -> th.Tensor:
        return th.tensor(goal_xy, device=device, dtype=dtype).view(1, 2)

    def _split_state(self, state: th.Tensor) -> tuple[th.Tensor, ...]:
        if state.ndim != 2 or state.shape[1] < 12:
            raise ValueError(
                "Expected state tensor of shape [B, >=12], "
                f"got {tuple(state.shape)}."
            )
        s = state[:, :12]
        ball_xy = s[:, 0:2]
        ball_vel = s[:, 2:4]
        peg1_xy = s[:, 4:6]
        peg2_xy = s[:, 6:8]
        peg1_vel = s[:, 8:10]
        peg2_vel = s[:, 10:12]
        return ball_xy, ball_vel, peg1_xy, peg2_xy, peg1_vel, peg2_vel

    def _state_features(self, state: th.Tensor) -> th.Tensor:
        ball_xy, ball_vel, peg1_xy, peg2_xy, peg1_vel, _peg2_vel = self._split_state(state)
        goal_player = self._goal_tensor((0.0, -0.176215), device=state.device, dtype=state.dtype)
        goal_opponent = self._goal_tensor((0.0, 0.176215), device=state.device, dtype=state.dtype)

        peg1_to_ball = ball_xy - peg1_xy
        peg2_to_ball = ball_xy - peg2_xy
        ball_to_opp_goal = goal_opponent - ball_xy
        ball_to_own_goal = goal_player - ball_xy
        peg1_to_opp_goal = goal_opponent - peg1_xy
        peg1_to_own_goal = goal_player - peg1_xy

        dist_peg1_ball = _safe_norm(peg1_to_ball)
        dist_ball_opp_goal = _safe_norm(ball_to_opp_goal)
        dist_ball_own_goal = _safe_norm(ball_to_own_goal)
        peg1_ball_goal_alignment = _safe_cos(peg1_to_ball, ball_to_opp_goal)
        ball_progress_to_opp_goal = th.sum(
            ball_vel * _safe_unit(ball_to_opp_goal), dim=-1, keepdim=True
        )
        peg1_approach_to_ball = th.sum(
            peg1_vel * _safe_unit(peg1_to_ball), dim=-1, keepdim=True
        )

        return th.cat(
            [
                peg1_to_ball,
                peg2_to_ball,
                ball_to_opp_goal,
                ball_to_own_goal,
                peg1_to_opp_goal,
                peg1_to_own_goal,
                ball_vel,
                peg1_vel,
                dist_peg1_ball,
                dist_ball_opp_goal,
                dist_ball_own_goal,
                peg1_ball_goal_alignment,
                ball_progress_to_opp_goal,
                peg1_approach_to_ball,
            ],
            dim=-1,
        )

    def _transition_features(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        if action.ndim != 2 or action.shape[1] < 2:
            raise ValueError(
                "Expected action tensor of shape [B, >=2], "
                f"got {tuple(action.shape)}."
            )
        state_features = self._state_features(state)
        next_state_features = self._state_features(next_state)
        action_peg1 = action[:, :2]
        action_speed = _safe_norm(action_peg1)

        state_dist_peg1_ball = state_features[:, 16:17]
        state_dist_ball_opp_goal = state_features[:, 17:18]
        next_dist_peg1_ball = next_state_features[:, 16:17]
        next_dist_ball_opp_goal = next_state_features[:, 17:18]
        delta_dist_peg1_ball = next_dist_peg1_ball - state_dist_peg1_ball
        delta_dist_ball_opp_goal = next_dist_ball_opp_goal - state_dist_ball_opp_goal

        return th.cat(
            [
                state_features,
                next_state_features[:, 0:4],
                delta_dist_peg1_ball,
                delta_dist_ball_opp_goal,
                action_peg1,
                action_speed,
                done.to(state.dtype).reshape(-1, 1),
            ],
            dim=-1,
        )


class KlaskRelativeBaseRewardNet(_KlaskRelativeFeatureMixin, RewardNet):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        hid_sizes: Sequence[int] = (64, 64),
        normalize_input_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        RewardNet.__init__(self, observation_space, action_space)
        _KlaskRelativeFeatureMixin.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
        )
        self._net = networks.build_mlp(
            in_size=self.transition_feature_dim,
            hid_sizes=hid_sizes,
            squeeze_output=True,
            flatten_input=False,
            normalize_input_layer=normalize_input_layer,
        )

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        features = self._transition_features(state, action, next_state, done)
        return self._net(features)


class KlaskRelativePotential(nn.Module, _KlaskRelativeFeatureMixin):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        hid_sizes: Sequence[int] = (64, 64),
        normalize_input_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        nn.Module.__init__(self)
        _KlaskRelativeFeatureMixin.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
        )
        self._net = networks.build_mlp(
            in_size=self.state_feature_dim,
            hid_sizes=hid_sizes,
            squeeze_output=True,
            flatten_input=False,
            normalize_input_layer=normalize_input_layer,
        )

    def forward(self, state: th.Tensor) -> th.Tensor:
        return self._net(self._state_features(state))


class KlaskRelativeShapedRewardNet(ShapedRewardNet):
    """Shaped reward net over relative Klask features instead of raw coordinates."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        reward_hid_sizes: Sequence[int] = (64, 64),
        potential_hid_sizes: Sequence[int] = (64, 64),
        discount_factor: float = 0.99,
        normalize_input_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        base_reward_net = KlaskRelativeBaseRewardNet(
            observation_space=observation_space,
            action_space=action_space,
            hid_sizes=reward_hid_sizes,
            normalize_input_layer=normalize_input_layer,
        )
        potential_net = KlaskRelativePotential(
            observation_space=observation_space,
            action_space=action_space,
            hid_sizes=potential_hid_sizes,
            normalize_input_layer=normalize_input_layer,
        )
        super().__init__(
            base_reward_net,
            potential_net,
            discount_factor=float(discount_factor),
        )


__all__ = [
    "KlaskRelativeBaseRewardNet",
    "KlaskRelativePotential",
    "KlaskRelativeShapedRewardNet",
]
