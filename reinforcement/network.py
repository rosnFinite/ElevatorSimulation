import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class A2CNetwork(nn.Module):
    def __init__(self, input_size=46, action_size=3, learning_rate=0.01, device="cpu"):
        super(A2CNetwork, self).__init__()
        self.device = device
        self.learning_rate = learning_rate

        self.shared_net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.actor_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        self.critic_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        torch.device(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def predict(self, state):
        shared_out = self.shared_net(torch.FloatTensor(state).to(device=self.device))
        probs = F.softmax(self.actor_net(shared_out), dim=-1)
        return probs, self.critic_net(shared_out)

    def get_next_action(self, state):
        probs = self.predict(state)[0].detach().numpy()
        action = np.random.choice([0, 1, 2], p=probs)
        return action

    def get_log_probs(self, state):
        shared_out = self.shared_net(torch.FloatTensor(state).to(device=self.device))
        log_probs = F.log_softmax(self.actor_net(shared_out), dim=-1)
        return log_probs
