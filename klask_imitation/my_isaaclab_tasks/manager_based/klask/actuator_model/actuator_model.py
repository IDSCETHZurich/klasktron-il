print("[ActuatorModel MODULE] imported from", __file__)

import os

import torch
import torch.nn as nn
from gymnasium import Wrapper

from ..config import KLASK_PARAMS


class ActuatorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(ActuatorNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        # softsign activation: y = x/(1+|x|)
        self.activation = nn.Softsign()

    def forward(self, x_commands, x_states=None):
        if x_states is not None:
            x = torch.cat([x_commands, x_states], dim=1)
        else:
            x = torch.cat([x_commands], dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc_out(x)
        return x


class ActuatorModelWrapper(Wrapper):
    model_file = os.path.join(os.path.dirname(__file__), "model_odrive_new_estimator_history_10_interval_0.02_delay_0.0_horizon3_with_states.pt")
    num_history_steps = 10  # Number of past commands to include in input
    hidden_dim = 64
    delay = 0  # Time delay between most recent command included in the input and the current time
    include_states = True  # Whether to include history of peg states in input

    def __init__(
        self,
        env,
        device="cuda",
        model_file=None,
        apply_to_player=True,
        apply_to_opponent=True,
    ):
        super().__init__(env)

        num_envs = env.unwrapped.num_envs
        self.apply_to_player = bool(apply_to_player)
        self.apply_to_opponent = bool(apply_to_opponent)
        self.dT = KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"]
        input_dim = (
            self.num_history_steps * 2
            + self.include_states * (self.num_history_steps - 1) * 2
        )
        output_dim = 2
        self.model = ActuatorNetwork(
            input_dim, output_dim, hidden_dim=self.hidden_dim
        ).to(device)
        if model_file is None:
            model_file = self.model_file
        self.model.load_state_dict(torch.load(model_file, map_location=device))
        self.command_buffer_1 = torch.zeros(
            num_envs, 2 * self.num_history_steps, dtype=torch.float32
        ).to(device)
        self.command_buffer_2 = torch.zeros(
            num_envs, 2 * self.num_history_steps, dtype=torch.float32
        ).to(device)
        if self.include_states:
            self.state_buffer_1 = torch.zeros(
                num_envs, 2 * (self.num_history_steps - 1), dtype=torch.float32
            ).to(device)
            self.state_buffer_2 = torch.zeros(
                num_envs, 2 * (self.num_history_steps - 1), dtype=torch.float32
            ).to(device)
            self.position_player = torch.zeros(num_envs, 2, dtype=torch.float32).to(
                device
            )
            self.position_opponent = torch.zeros(num_envs, 2, dtype=torch.float32).to(
                device
            )

        self.model.eval()
        print(
            "[ActuatorModelWrapper] enabled sides:"
            f" player={self.apply_to_player} opponent={self.apply_to_opponent}"
        )

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)

        # Peg 1 history update:
        state_1 = obs["policy"][:, 2:4]

        if self.include_states:
            self.state_buffer_1[:, :] = state_1.repeat(1, self.num_history_steps - 1)
        self.command_buffer_1[:, :] = 0.0

        # Peg 2 history update:
        state_2 = obs["opponent"][
            :, 2:4
        ]  # here was a minus before but doesnt make sense

        if self.include_states:
            self.state_buffer_2[:, :] = state_2.repeat(1, self.num_history_steps - 1)
        self.command_buffer_2[:, :] = 0.0

        return obs, info

    def step(self, actions, *args, **kwargs):
        action_dim = int(actions.shape[1]) if actions.ndim >= 2 else 0

        # Peg 1 command history update:
        command_1 = actions[:, :2]

        prev_buffer = self.command_buffer_1.clone()
        self.command_buffer_1[:, 2:] = prev_buffer[:, :-2]
        self.command_buffer_1[:, :2] = command_1

        # Peg 2 command history update:
        command_2 = None
        if action_dim >= 4:
            command_2 = actions[:, 2:4]
            self.command_buffer_2[:, 2:] = self.command_buffer_2.clone()[:, :-2]
            self.command_buffer_2[:, :2] = command_2

        # Map commands to velocities using the actuator model:
        if self.include_states:
            states_input_1, states_input_2 = self.state_buffer_1, self.state_buffer_2
        else:
            states_input_1, states_input_2 = None, None

        with torch.no_grad():
            if self.apply_to_player:
                actions[:, :2] = self.model(self.command_buffer_1, states_input_1)
            if self.apply_to_opponent and action_dim >= 4:
                actions[:, 2:4] = self.model(self.command_buffer_2, states_input_2)
        obs, rew, terminated, truncated, info = self.env.step(actions, *args, **kwargs)

        if self.include_states:
            # Peg 1 state history update:
            self.position_player = obs["policy"][:, :2]
            state_1 = obs["policy"][:, 2:4]

            self.state_buffer_1[:, 2:] = self.state_buffer_1.clone()[:, :-2]
            self.state_buffer_1[:, :2] = state_1

            # Peg 2 state history update:
            self.position_opponent = obs["opponent"][:, :2]
            state_2 = obs["opponent"][:, 2:4]

            self.state_buffer_2[:, 2:] = self.state_buffer_2.clone()[:, :-2]
            self.state_buffer_2[:, :2] = state_2

        # Reset buffers for terminated envs:
        done = terminated | truncated
        self.command_buffer_1[done, :] = 0.0
        self.command_buffer_2[done, :] = 0.0
        if self.include_states:
            self.state_buffer_1[done, :] = state_1[done].repeat(
                1, self.num_history_steps - 1
            )
            self.state_buffer_2[done, :] = state_2[done].repeat(
                1, self.num_history_steps - 1
            )

        return obs, rew, terminated, truncated, info
