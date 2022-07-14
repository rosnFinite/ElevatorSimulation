import torch
import torch.nn as nn
import torch.nn.functional as F


class A2CNetwork(nn.Module):
    def __init__(self, input_size=82, action_size=27):
        super(A2CNetwork, self).__init__()
        self.action_size = action_size
        self.input_size = input_size

        self.shared_net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.actor_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

        self.critic_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        state_tensor = torch.FloatTensor(state)
        shared_out = self.shared_net(state_tensor)
        value = self.critic_net(shared_out)

        policy_dist = torch.distributions.Categorical(F.softmax(self.actor_net(shared_out), dim=-1))

        return value, policy_dist
