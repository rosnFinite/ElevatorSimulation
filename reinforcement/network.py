import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class A2CNetwork(nn.Module):
    def __init__(self, input_size=82, action_size=27, learning_rate=0.01):
        super(A2CNetwork, self).__init__()
        self.action_size = action_size
        self.device = torch.device("cpu")
        self.learning_rate = learning_rate

        self.shared_net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        self.actor_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

        self.critic_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def predict(self, state):
        state_tensor = torch.FloatTensor(state).to(device=self.device)
        shared_out = self.shared_net(state_tensor)
        probs = F.softmax(self.actor_net(shared_out), dim=-1)
        return probs, self.critic_net(shared_out)

    def get_next_action(self, state):
        probs = self.predict(state)[0].detach().numpy()
        action = np.random.choice([x for x in range(self.action_size)], p=probs)
        return action

    def get_log_probs(self, state):
        shared_out = self.shared_net(torch.FloatTensor(state).to(device=self.device))
        log_probs = F.log_softmax(self.actor_net(shared_out), dim=-1)
        return log_probs
