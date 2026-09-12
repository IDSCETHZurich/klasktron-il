import random
import re
import shutil
from pathlib import Path

import gym
import numpy as np
import torch
import yaml
from gymnasium import Wrapper
from isaaclab_rl.rl_games import RlGamesGpuEnv
from rl_games.torch_runner import Runner
from gymnasium import spaces
from .mdp.rewards import ball_in_goal, ball_speed, ball_stationary, collision_player_ball, \
    distance_ball_goal, distance_player_ball_own_half, distance_to_wall, ball_in_own_half, \
    in_goal, peg_in_defense_line_with_rebounds
from isaaclab.managers import SceneEntityCfg
from .config.parameters import KLASK_PARAMS


def find_wrapper(env, wrapper_type):
    """Recursively searches for a wrapper of a given type."""
    while not isinstance(env, wrapper_type):
        env = env.env  # Move to the next layer
    if isinstance(env, wrapper_type):
        return env  # Found the wrapper
    return None  # Wrapper not found


class KlaskRandomOpponentWrapper(Wrapper):
    def step(self, actions, *args, **kwargs):
        # CAREFUL: Opponent actions
        if actions.size() == torch.Size([4096, 4]):
            actions[:, 2:] = 2 * torch.rand_like(actions)[:, :2] - 1
        else:
            opponent_actions = 2 * torch.rand_like(actions) - 1  # shape: [4096, 2]
            actions = torch.cat([actions, opponent_actions], dim=1)
        return self.env.step(actions, *args, **kwargs)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)


class OpponentObservationWrapper(Wrapper):
    def __init__(self, env, mode="train"):
        super().__init__(env)
        self.mode = mode

    def get_opponent_obs(self, obs):
        opponent_obs = obs.detach().clone()
        opponent_obs[:, :12] = -obs[:, :12]
        return opponent_obs

    def reset(self, *args, **kwargs):
        obs_dict, extras = self.env.reset(*args, **kwargs)
        if self.mode == "train":
            self.opponent_obs = self.get_opponent_obs(obs_dict["opponent"])
        else:
            obs_dict["opponent"] = self.get_opponent_obs(obs_dict["opponent"])
        return obs_dict, extras

    def step(self, actions, *args, **kwargs):
        obs_dict, rew, terminated, truncated, extras = self.env.step(
            actions, *args, **kwargs
        )
        if self.mode == "train":
            self.opponent_obs = self.get_opponent_obs(obs_dict["opponent"])
        else:
            obs_dict["opponent"] = self.get_opponent_obs(obs_dict["opponent"])
        return obs_dict, rew, terminated, truncated, extras


class RlGamesGpuEnvSelfPlay(RlGamesGpuEnv):
    def __init__(
        self,
        config_name,
        num_actors,
        config,
        training_curriculum=False,
        mode=None,
        folder=None,
        is_deterministic=False,
        **kwargs,
    ):
        self.agent = None
        self.config = config
        self.instance_device = config["params"]["config"]["device"]
        self.is_deterministic = is_deterministic
        self.sum_rewards = 0
        self.training_curriculum = training_curriculum

        if training_curriculum:
            self.mode = mode
            self.base_folder = Path(folder)
            self.current_checkpoint = None
            self.config_path = None
            self.counter = 0
        self.current_config = self.config
        super().__init__(config_name, num_actors, **kwargs)

    def reset(self):
        if self.training_curriculum:
            self.should_update_agents()
        if self.agent == None:
            self.create_agent()
        # if self.training_curriculum and new_file:

        obs = self.env.reset()
        # self.opponent_obs = self.get_opponent_obs(obs)
        self.opponent_obs = find_wrapper(
            self.env, OpponentObservationWrapper
        ).opponent_obs
        self.sum_rewards = 0
        return obs

    def create_agent(self):
        runner = Runner()
        from rl_games.common.env_configurations import get_env_info

        self.config["params"]["config"]["env_info"] = get_env_info(self.env)
        runner.load(self.current_config)

        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        restore_checkpoint = (
            self.training_curriculum and self.current_checkpoint is not None
        )
        self.agent = runner.create_player()
        if restore_checkpoint:
            self.agent.restore(self.current_checkpoint)

        self.agent.has_batch_dimension = True

    def should_update_agents(self, weights=None):
        if self.mode == 2 and self.counter > 15:
            use_old_opp = random.random() < 0.3
            if not use_old_opp and weights is not None:
                self.set_weights(indices=None, weigths=weights)
                self.counter = 1
                return

            base_folder = Path(self.base_folder)
            checkpoints = list(base_folder.glob("last*.pth"))

            if len(checkpoints) == 0:
                return
            checkpoints_sorted = sorted(checkpoints, key=lambda f: f.stat().st_ctime)
            decay = (
                0.9  # Closer to 1 → slower decay, closer to 0 → steeper bias to latest
            )
            n = len(checkpoints_sorted)
            weights = np.array(
                [decay ** (n - i - 1) for i in range(n)]
            )  # newest gets highest weight
            weights /= weights.sum()  # normalize to sum to 1

            # Randomly choose using the computed weights
            self.current_checkpoint = np.random.choice(checkpoints_sorted, p=weights)

            self.create_agent()
            self.counter = 0

        if self.mode == 1:
            matching_file = str(list(self.base_folder.glob("*"))[-1])
            if matching_file == self.current_checkpoint:
                return
            self.current_checkpoint = matching_file

            match = re.search(r"best_agent\((\d+)\)", self.current_checkpoint)
            if match:
                agent_number = match.group(1)
            self.config_path = (
                Path(self.base_folder) / f"klask_config_{agent_number}.yaml"
            )

        if self.mode == 0 and self.counter > 8:
            agent_folders = [f for f in self.base_folder.iterdir() if f.is_dir()]
            agent_folders_sorted = sorted(
                agent_folders, key=lambda f: f.stat().st_ctime
            )

            if len(agent_folders) > 4:  # if only the benchmark folder is in there
                folders_to_delete = agent_folders_sorted[1:-4]
                for folder in folders_to_delete:
                    shutil.rmtree(folder)

                use_self_play = random.random() < 0.5
                if use_self_play and weights is not None:
                    self.set_weights(indices=None, weigths=weights)
                    self.counter = 1
                    return
                # Choose one at random
                chosen_folder = random.choice(agent_folders)

                # Find the checkpoint and config file in the chosen folder
                checkpoint_path = next(
                    chosen_folder.glob("**/player_checkpoint.pth"), None
                )
                self.config_path = next(
                    chosen_folder.glob("**/player_config.yaml"), None
                )
                self.current_checkpoint = checkpoint_path

        if self.mode == 0 and self.counter > 8 and self.current_checkpoint is not None:
            with open(self.config_path, "r") as f:
                self.current_config = yaml.safe_load(f)
                self.current_config["params"]["config"]["device"] = self.instance_device
                self.current_config["params"]["config"]["device_name"] = (
                    self.instance_device
                )

            self.create_agent()
            self.counter = 0
        self.counter += 1

    def step(self, action, *args, **kwargs):
        opponent_obs = self.agent.obs_to_torch(self.opponent_obs)
        opponent_action = self.agent.get_action(opponent_obs, self.is_deterministic)
        full_action = torch.cat([action, -opponent_action], dim=1)
        obs, reward, dones, info = self.env.step(full_action, *args, **kwargs)
        # self.opponent_obs = self.get_opponent_obs(obs)
        self.opponent_obs = find_wrapper(
            self.env, OpponentObservationWrapper
        ).opponent_obs
        return obs, reward, dones, info

    def set_weights(self, indices, weigths):
        print("SETTING WEIGHTS")

        self.agent.set_weights(weigths)
        self.is_deterministic = True


class KlaskAgentOpponentWrapper(Wrapper):
    def __init__(self, env, is_deterministic=False):
        super().__init__(env)
        self.opponent = None
        self.is_deterministic = is_deterministic

    def add_opponent(self, opponent):
        self.opponent = opponent
        self.opponent.has_batch_dimension = True

    def get_opponent_obs(self, obs):
        opponent_obs = obs.detach().clone()
        opponent_obs[:, :12] = -obs[:, :12]
        return opponent_obs

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.opponent_obs = self.get_opponent_obs(obs["opponent"])
        return obs, info

    def step(self, action, *args, **kwargs):
        opponent_obs = self.opponent.obs_to_torch(self.opponent_obs)
        opponent_action = self.opponent.get_action(opponent_obs, self.is_deterministic)
        full_action = torch.cat([action, -opponent_action], dim=1)
        obs, reward, terminated, truncated, info = self.env.step(
            full_action, *args, **kwargs
        )

        self.opponent_obs = self.get_opponent_obs(obs["opponent"])
        return obs, reward, terminated, truncated, info


class KlaskCollisionAvoidanceWrapper(Wrapper):
    real_to_sim_factor_long_side = 0.0008285
    real_to_sim_factor_short_side = 1 / 1150
    DEACCELERATION_DISTANCE = 0.09
    PEG_RADIUS = 0.0075
    X_EDGE = (-0.16, 0.16)
    Y_EDGE_PLAYER = (-0.03, -0.22)
    Y_EDGE_OPPONENT = (-0.03, -0.22)
    board_dimensions = (0.32, 0.44)
    speed_limit_weight = 70.0

    def __init__(self, env, max_vel=0.2):
        super().__init__(env)

        self.x_min, self.x_max = 15.0, 360.0
        self.y_min_1, self.y_max_1 = 15.0, 235.0
        self.y_min_2, self.y_max_2 = 340.0, 530.0
        self.MAX_VEL = max_vel

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.state_1 = obs["policy"].clone()[:, :2]
        self.state_2 = obs["opponent"].clone()[:, :2]

        return obs, info

    def step(self, actions, *args, **kwargs):
        left_zone = (
            self.state_1[:, 0]
            <= self.X_EDGE[0] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS
        )
        soft_dist_left = self.state_1[left_zone, 0] - self.X_EDGE[0] - self.PEG_RADIUS
        actions[left_zone, 0] = torch.maximum(
            actions[left_zone, 0], -self.interpolate_vel(soft_dist_left)
        )

        left_zone_opp = (
            self.state_2[:, 0]
            <= self.X_EDGE[0] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS
        )
        soft_dist_left_opp = (
            self.state_2[left_zone_opp, 0] - self.X_EDGE[0] - self.PEG_RADIUS
        )
        actions[left_zone_opp, 2] = torch.maximum(
            actions[left_zone_opp, 2], -self.interpolate_vel(soft_dist_left_opp)
        )

        right_zone = (
            self.state_1[:, 0] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS
            >= self.X_EDGE[1]
        )
        actions[right_zone, 0] = torch.minimum(
            actions[right_zone, 0],
            self.interpolate_vel(
                self.X_EDGE[1] - self.state_1[right_zone, 0] - self.PEG_RADIUS
            ),
        )

        right_zone_opp = (
            self.state_2[:, 0] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS
            >= self.X_EDGE[1]
        )
        actions[right_zone_opp, 1] = torch.minimum(
            actions[right_zone_opp, 1],
            self.interpolate_vel(
                self.X_EDGE[1] - self.state_2[right_zone_opp, 0] - self.PEG_RADIUS
            ),
        )

        bottom_zone = (
            self.state_1[:, 1]
            <= self.Y_EDGE_PLAYER[0] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS
        )
        actions[bottom_zone, 1] = torch.maximum(
            actions[bottom_zone, 1],
            -self.interpolate_vel(
                self.state_1[:, 1][bottom_zone]
                - self.Y_EDGE_PLAYER[0]
                - self.PEG_RADIUS
            ),
        )

        bottom_zone_opp = (
            self.state_2[:, 1]
            <= self.Y_EDGE_OPPONENT[0] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS
        )
        actions[bottom_zone_opp, 1] = torch.maximum(
            actions[bottom_zone_opp, 1],
            -self.interpolate_vel(
                self.state_2[:, 1][bottom_zone_opp]
                - self.Y_EDGE_OPPONENT[0]
                - self.PEG_RADIUS
            ),
        )

        ### Y - TOP
        top_zone = (
            self.state_1[:, 1] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS
            >= self.Y_EDGE_PLAYER[1]
        )
        actions[top_zone, 1] = torch.minimum(
            actions[top_zone, 1],
            self.interpolate_vel(
                self.Y_EDGE_PLAYER[1] - self.state_1[:, 1][top_zone] + self.PEG_RADIUS
            ),
        )

        top_zone_opp = (
            self.state_2[:, 1] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS
            >= self.Y_EDGE_OPPONENT[1]
        )
        actions[top_zone_opp, 1] = torch.minimum(
            actions[top_zone_opp, 1],
            self.interpolate_vel(
                self.Y_EDGE_OPPONENT[1]
                - self.state_2[:, 1][top_zone_opp]
                + self.PEG_RADIUS
            ),
        )

        obs, rew, terminated, truncated, info = self.env.step(actions, *args, **kwargs)
        self.state_1 = obs["policy"].clone()[:, :2]
        self.state_2 = obs["opponent"].clone()[:, :2]

        return obs, rew, terminated, truncated, info

    def interpolate_vel(self, distance):
        return (self.MAX_VEL / self.DEACCELERATION_DISTANCE) * distance

'''
class ActionHistoryWrapper(Wrapper):
    def __init__(self, env, history_length):
        super().__init__(env)
        self.history_length = history_length
        self.history_x_player = torch.zeros(env.unwrapped.num_envs, history_length).to(
            env.unwrapped.device
        )
        self.history_y_player = torch.zeros(env.unwrapped.num_envs, history_length).to(
            env.unwrapped.device
        )
        self.history_x_opponent = torch.zeros(
            env.unwrapped.num_envs, history_length
        ).to(env.unwrapped.device)
        self.history_y_opponent = torch.zeros(
            env.unwrapped.num_envs, history_length
        ).to(env.unwrapped.device)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, actions, *args, **kwargs):
        self.history_x_player[:, :-1] = self.history_x_player.clone()[:, 1:]
        self.history_x_player[:, -1] = actions[:, 0]
        self.history_y_player[:, :-1] = self.history_y_player.clone()[:, 1:]
        self.history_y_player[:, -1] = actions[:, 1]
        self.history_x_opponent[:, :-1] = self.history_x_opponent.clone()[:, 1:]
        self.history_x_opponent[:, -1] = actions[:, 2]
        self.history_y_opponent[:, :-1] = self.history_y_opponent.clone()[:, 1:]
        self.history_y_opponent[:, -1] = actions[:, 3]

        obs, rew, terminated, truncated, info = self.env.step(actions, *args, **kwargs)
        obs["policy"][:, 12:] = torch.cat(
            [self.history_x_player, self.history_y_player], dim=1
        )
        obs["opponent"][:, 12:] = torch.cat(
            [self.history_x_opponent, self.history_y_opponent], dim=1
        )
        return obs, rew, terminated, truncated, info
'''


class ActionHistoryWrapper(Wrapper):
    """
    Adds last K action components to observations.
    - Dict mode: augments both keys ("policy", "opponent") by 2*K each.
    - Box mode: augments flat obs by 2*K (player lvx,lvy).
    Assumes actions: [lvx, lvy, rvx, rvy].
    """
    def __init__(self, env, history_length: int):
        super().__init__(env)
        self.history_length = int(history_length)
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        base_space = env.observation_space
        if isinstance(base_space, spaces.Dict):
            self.mode = "dict"
            pol_dim = int(np.prod(base_space["policy"].shape))
            opp_dim = int(np.prod(base_space["opponent"].shape))
            assert pol_dim == opp_dim, "policy/opponent dims must match"

            extra = 2 * self.history_length
            self.observation_space = spaces.Dict({
                "policy": spaces.Box(low=-np.inf, high=np.inf, shape=(pol_dim + extra,), dtype=np.float32),
                "opponent": spaces.Box(low=-np.inf, high=np.inf, shape=(opp_dim + extra,), dtype=np.float32),
            })

            # history buffers
            self.hx_pol = torch.zeros(self.num_envs, self.history_length, device=self.device)
            self.hy_pol = torch.zeros(self.num_envs, self.history_length, device=self.device)
            self.hx_opp = torch.zeros(self.num_envs, self.history_length, device=self.device)
            self.hy_opp = torch.zeros(self.num_envs, self.history_length, device=self.device)

        elif isinstance(base_space, spaces.Box):
            self.mode = "box"
            base_dim = int(np.prod(base_space.shape))
            extra = 2 * self.history_length
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(base_dim + extra,), dtype=np.float32
            )

            # player-only history buffers
            self.hx = torch.zeros(self.num_envs, self.history_length, device=self.device)
            self.hy = torch.zeros(self.num_envs, self.history_length, device=self.device)
        else:
            raise TypeError(f"Unsupported observation space: {type(base_space)}")

    def _roll_and_insert(self, actions: torch.Tensor):
        if self.history_length == 0:
            return
        if self.mode == "dict":
            # shift
            self.hx_pol[:, :-1] = self.hx_pol[:, 1:]
            self.hy_pol[:, :-1] = self.hy_pol[:, 1:]
            self.hx_opp[:, :-1] = self.hx_opp[:, 1:]
            self.hy_opp[:, :-1] = self.hy_opp[:, 1:]
            # newest
            # actions shape expected [N,4]; fallback if only [N,2]
            self.hx_pol[:, -1] = actions[:, 0]
            self.hy_pol[:, -1] = actions[:, 1]
            src2 = actions.shape[1] >= 4
            self.hx_opp[:, -1] = actions[:, 2] if src2 else 0.0
            self.hy_opp[:, -1] = actions[:, 3] if src2 else 0.0
        else:
            self.hx[:, :-1] = self.hx[:, 1:]
            self.hy[:, :-1] = self.hy[:, 1:]
            self.hx[:, -1] = actions[:, 0]
            self.hy[:, -1] = actions[:, 1]

    def reset(self, *args, **kwargs):
        # clear histories
        if self.history_length > 0:
            if self.mode == "dict":
                self.hx_pol.zero_(); self.hy_pol.zero_()
                self.hx_opp.zero_(); self.hy_opp.zero_()
            else:
                self.hx.zero_(); self.hy.zero_()

        obs, info = self.env.reset(*args, **kwargs)
        if self.mode == "dict":
            obs["policy"] = torch.cat([obs["policy"], self.hx_pol, self.hy_pol], dim=1)
            obs["opponent"] = torch.cat([obs["opponent"], self.hx_opp, self.hy_opp], dim=1)
        else:
            obs = torch.cat([obs, self.hx, self.hy], dim=1)
        return obs, info

    def step(self, actions, *args, **kwargs):
        self._roll_and_insert(actions)
        obs, rew, terminated, truncated, info = self.env.step(actions, *args, **kwargs)
        if self.mode == "dict":
            obs["policy"] = torch.cat([obs["policy"], self.hx_pol, self.hy_pol], dim=1)
            obs["opponent"] = torch.cat([obs["opponent"], self.hx_opp, self.hy_opp], dim=1)
            return obs, rew, terminated, truncated, info
        else:
            obs = torch.cat([obs, self.hx, self.hy], dim=1)
            return obs, rew, terminated, truncated, info


class BallNearPegResetWrapper(gym.Wrapper):
    """
    On each reset, move the ball to a position within `radius` of either the
    player (left) or opponent (right) peg (chosen randomly by default).

    Assumes flat observation layout used elsewhere in your code:
      obs = [bp_x, bp_y, bv_x, bv_y, lp_x, lp_y, rp_x, rp_y, lv_x, lv_y, rv_x, rv_y]
    """
    def __init__(self, env, radius: float = 0.02, which: str = "random"):
        super().__init__(env)
        self.radius = float(radius)
        self.which = which  # "random" | "player" | "opponent"
        self._rng = np.random.default_rng()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._place_ball_near_peg(obs)
        return obs, info

    def _extract_array(self, obs):
        if isinstance(obs, dict):
            for k in ("policy", "obs", "default"):
                if k in obs:
                    return obs[k]
            return next(iter(obs.values()))
        return obs

    def _place_ball_near_peg(self, obs):
        # Extract observation array
        if isinstance(obs, dict):
            arr = obs.get("policy", next(iter(obs.values())))
        else:
            arr = obs

        ball = self.env.unwrapped.scene["ball"]
        scene = self.env.unwrapped.scene

        # Use the asset buffers' device/dtype
        device = ball.data.root_pos_w.device
        dtype = ball.data.root_pos_w.dtype

        arr_t = torch.as_tensor(arr, dtype=torch.float32, device=device)
        if arr_t.ndim == 1:
            arr_t = arr_t.unsqueeze(0)
        n = arr_t.shape[0]
        env_ids = list(range(n))

        # Pick peg (env-frame xy from your obs layout)
        left = arr_t[:, 4:6]    # lp_x, lp_y
        right = arr_t[:, 6:8]   # rp_x, rp_y
        if self.which == "player":
            peg_xy = left
        elif self.which == "opponent":
            peg_xy = right
        else:
            mask = (torch.rand(n, 1, device=device) < 0.5)
            peg_xy = torch.where(mask, left, right)

        # Random offset uniform in disk of radius
        theta = 2 * torch.pi * torch.rand(n, 1, device=device)
        rad = torch.sqrt(torch.rand(n, 1, device=device)) * self.radius
        eps = torch.cat([rad * torch.cos(theta), rad * torch.sin(theta)], dim=1)

        # Convert env-frame target to world-frame using env origins
        target_xy_env = peg_xy + eps
        origins_xy = scene.env_origins[:, :2].to(device=device, dtype=torch.float32)[:n]
        target_xy_world = target_xy_env + origins_xy
        z_world = torch.full((n, 1), 0.032, device=device, dtype=torch.float32)  # table z

        pos_world = torch.cat([target_xy_world, z_world], dim=1).to(dtype)
        quat_world = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype).repeat(n, 1)

        # Write into buffers
        ball.data.root_pos_w[env_ids] = pos_world
        ball.data.root_quat_w[env_ids] = quat_world
        ball.data.root_lin_vel_w[env_ids] = 0.0
        ball.data.root_ang_vel_w[env_ids] = 0.0

        # Push to simulator
        if hasattr(ball, "write_data_to_sim"):
            ball.write_data_to_sim(env_ids=env_ids)
        elif hasattr(ball, "write_root_state_to_sim"):
            ball.write_root_state_to_sim(env_ids=env_ids)
        elif hasattr(ball, "write_root_pose_to_sim"):
            ball.write_root_pose_to_sim(env_ids=env_ids)



class RewardInfoWrapper(Wrapper):
    """
    Wrapper that recomputes individual reward components and exposes them via `info`.
    Also adds boolean flags for reward/termination diagnostics.
    """

    def __init__(self, env):
        super().__init__(env)

        base_env = self.env.unwrapped

        # Scene configs
        self.ball_cfg = SceneEntityCfg("ball")
        self.player_cfg = SceneEntityCfg("klask", body_names=["Peg_1"])
        self.opp_cfg = SceneEntityCfg("klask", body_names=["Peg_2"])

        scene = KLASK_PARAMS["scene"]
        self.player_goal = scene["player_goal"]
        self.opp_goal = scene["opponent_goal"]
        self.max_ball_vel_goal = scene["max_ball_vel_goal"]

        self.player_cfg.resolve(base_env.scene)
        self.opp_cfg.resolve(base_env.scene)


    def step(self, actions, *args, **kwargs):
        obs, rew, terminated, truncated, info = self.env.step(actions, *args, **kwargs)

        # underlying ManagerBasedRLEnv
        base_env = self.env.unwrapped

        # --- raw (unweighted) components as tensors (shape [num_envs]) ---
        r_goal_scored_raw = ball_in_goal(
            base_env, self.ball_cfg, self.opp_goal,
            max_ball_vel=self.max_ball_vel_goal, weight=None
        )
        r_goal_conceded_raw = ball_in_goal(
            base_env, self.ball_cfg, self.player_goal,
            max_ball_vel=self.max_ball_vel_goal, weight=None
        )

        r_ball_speed = ball_speed(base_env, self.ball_cfg)
        r_ball_stationary = ball_stationary(base_env, self.ball_cfg)
        r_coll = collision_player_ball(base_env, self.player_cfg, self.ball_cfg)
        r_dist_ball_opp = distance_ball_goal(base_env, self.ball_cfg, self.opp_goal)
        r_dist_ball_own = distance_ball_goal(base_env, self.ball_cfg, self.player_goal)
        r_dist_player_ball_own = distance_player_ball_own_half(
            base_env, self.player_cfg, self.ball_cfg
        )
        r_ball_own_half = ball_in_own_half(base_env, self.ball_cfg)
        r_dist_to_wall = distance_to_wall(
            base_env, self.player_cfg, KLASK_PARAMS["edge"]
        )
        r_block = peg_in_defense_line_with_rebounds(
            base_env, self.player_cfg, self.opp_cfg, self.ball_cfg
        )

        # --- write into info dict as components ---
        # (info is a dict of tensors for IsaacLab; Sb3VecEnvWrapper will
        #  split them per-env and convert to numpy for SB3.)
        info["rew_goal_scored_raw"] = r_goal_scored_raw
        info["rew_goal_conceded_raw"] = r_goal_conceded_raw
        info["rew_ball_speed"] = r_ball_speed
        info["rew_ball_stationary"] = r_ball_stationary
        info["rew_collision_player_ball"] = r_coll
        info["rew_distance_ball_opponent_goal"] = r_dist_ball_opp
        info["rew_distance_ball_own_goal"] = r_dist_ball_own
        info["rew_distance_player_ball_own_half"] = r_dist_player_ball_own
        info["rew_ball_in_own_half"] = r_ball_own_half
        info["rew_close_to_boundaries"] = r_dist_to_wall
        info["rew_player_strategically_positioned"] = r_block

        # geometric goal detection (do NOT rely on rewards)
        ball_in_opp_goal = ball_in_goal(
            base_env,
            self.ball_cfg,
            self.opp_goal,
            max_ball_vel=2.0,   # or whatever you actually want
            weight=None,
        ).bool()   # (num_envs,)

        ball_in_own_goal = ball_in_goal(
            base_env,
            self.ball_cfg,
            self.player_goal,
            max_ball_vel=2.0,
            weight=None,
        ).bool()

        player_in_own_goal = in_goal(
            base_env,
            self.player_cfg,
            self.player_goal,
        ).bool()
        opp_in_own_goal = in_goal(
            base_env,
            self.opp_cfg,
            self.opp_goal,
        ).bool()

        if isinstance(truncated, torch.Tensor):
            time_out = truncated.to(dtype=torch.bool)
        else:
            time_out = torch.as_tensor(np.asarray(truncated), dtype=torch.bool, device=ball_in_opp_goal.device)

        info["goal_scored"] = ball_in_opp_goal
        info["goal_conceded"] = ball_in_own_goal
        info["player_in_goal"] = player_in_own_goal
        info["opponent_in_goal"] = opp_in_own_goal
        info["time_out"] = time_out


        return obs, rew, terminated, truncated, info
