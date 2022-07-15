import torch
import torch.nn as nn
import torch.nn.functional as F


class A2CNetwork(nn.Module):
    def __init__(self, input_size=124, action_size=27):
        super(A2CNetwork, self).__init__()
        self.action_size = action_size
        self.input_size = input_size

        self.shared_net = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU()
        )

        self.actor_net = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_size)
        )

        self.critic_net = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

        self.shared_net.cuda()
        self.actor_net.cuda()
        self.critic_net.cuda()

    def forward(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).cuda()
        shared_out = self.shared_net(state_tensor).cuda()
        value = self.critic_net(shared_out).cuda()

        policy_dist = torch.distributions.Categorical(F.softmax(self.actor_net(shared_out), dim=-1))

        return value, policy_dist
