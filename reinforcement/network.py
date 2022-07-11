import torch
import torch.nn as nn
import torch.nn.functional as F


class A2CNetwork(nn.Module):
    def __init__(self, input_size=82, action_size=27, hidden_size=128, learning_rate=0.01):
        super(A2CNetwork, self).__init__()
        self.action_size = action_size
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.device = torch.device("cpu")

        self.actor_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

        self.critic_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        state_tensor = torch.FloatTensor(state).to(device=self.device)
        value = self.critic_net(state_tensor)

        policy_dist = F.softmax(self.actor_net(state_tensor), dim=-1)

        return value, policy_dist
