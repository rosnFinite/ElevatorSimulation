import numpy as np
import torch
from network import CriticNetwork, ActorNetwork
from torch import optim
from torch.functional import F


class A2CAgent:
    def __init__(self, environment):
        self.env = environment
        self.gamma = 0.99
        self.action_size = 3
        self.observation_size = 46

        self.critic_network = CriticNetwork(self.observation_size, 32, 1)
        self.actor_network = ActorNetwork(self.observation_size, 32, 3)

        self.critic_network_optimizer = optim.Adam(self.critic_network.parameters(), lr=0.001)
        self.actor_network_optimizer = optim.Adam(self.actor_network.parameters(), lr=0.001)

    def _return_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), [next_value], axis=0)
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

    def training_on_batch(self, epochs, batch_size):
        episode_count = 0
        actions = np.empty((batch_size,), dtype=np.bool)
        dones = np.empty((batch_size,), dtype=np.bool)
        rewards, values = np.empty((2, batch_size), dtype=np.float)
        observations = np.empty((batch_size, self.observation_size), dtype=np.float)
        observation = self.env.get_state()

        for epoch in range(epochs):
            for i in range(batch_size):
                observations[i] = observation
                values[i] = self.critic_network(torch.tensor(observation, dtype=torch.float)).detach().numpy()
                policy = self.actor_network(torch.tensor(observation, dtype=torch.float))
                actions[i] = torch.multinomial(policy, 1).detach().numpy()
                observation, rewards[i], dones[i] = self.env.step(actions[i])

                if dones[i]:
                    observation = self.env.reset()

            if dones[-1]:
                next_value = 0
            else:
                next_value = self.critic_network(torch.tensor(observation, dtype=torch.float)).detach().numpy()

            episode_count += sum(dones)
            returns, advantages = self._return_advantages(rewards, dones, values, next_value)
            self.optimize_model(observations, actions, returns, advantages)

    def optimize_model(self, observations, actions, returns, advantages):
        actions = F.one_hot(torch.tensor(actions), self.action_size)
        returns = torch.tensor(returns[:, None], dtype=torch.float)
        advantages = torch.tensor(advantages, dtype=torch.float)
        observations = torch.tensor(observations, dtype=torch.float)


