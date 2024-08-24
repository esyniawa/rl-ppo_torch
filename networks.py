import torch
import torch.nn as nn


class ActorLinear(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(ActorLinear, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state: torch.Tensor):
        action_probs = self.actor(state)
        return action_probs


class CriticLinear(nn.Module):
    def __init__(self, state_dim: int):
        super(CriticLinear, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state: torch.Tensor):
        state_value = self.critic(state)
        return state_value
