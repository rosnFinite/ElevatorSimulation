import numpy as np
import torch
import json
from reinforcement.network import A2CNetwork
from skyscraper import Skyscraper

ACTION_ENCODING = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0],
                   [0, 2, 1], [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1],
                   [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0],
                   [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]]


class A2C:
    def __init__(self, env, network):
        self.env = env
        self.network = network
        self.action_space = np.arange(3)
        # Set up lists to log data
        self.ep_rewards = []
        self.policy_loss = []
        self.value_loss = []
        self.entropy_loss = []
        self.total_policy_loss = []
        self.total_loss = []

    def generate_episode(self, isTrain=True):
        states, actions, rewards, dones, next_states = [], [], [], [], []
        counter = 0
        done = False
        if not isTrain:
            self.s_0, _, _ = self.env.get_state()
            self.reward = 0
        while not done:
            action = self.network.get_next_action(self.s_0)
            self.env.schedule_action(ACTION_ENCODING[action])
            s_1, r, done = self.env.step()
            self.reward += r
            states.append(self.s_0)
            next_states.append(s_1)
            actions.append(action)
            rewards.append(r)
            dones.append(done)
            self.s_0 = s_1

            if done:
                self.ep_rewards.append(self.reward)
                self.s_0, _, _ = self.env.get_state()
                self.reward = 0
                if isTrain:
                    self.env = Skyscraper()
                    self.ep_counter += 1
            counter += 1
        return states, actions, rewards, dones, next_states

    def calc_rewards(self, batch):
        states, actions, rewards, dones, next_states = batch
        rewards = np.array(rewards)
        total_steps = len(rewards)
        state_values = self.network.predict(states)[1]
        next_state_values = self.network.predict(next_states)[1]
        done_mask = torch.ByteTensor(dones).to(self.network.device)
        next_state_values[done_mask] = 0.0
        state_values = state_values.detach().numpy().flatten()
        next_state_values = next_state_values.detach().numpy().flatten()

        G = np.zeros_like(rewards, dtype=np.float32)
        td_delta = np.zeros_like(rewards, dtype=np.float32)
        dones = np.array(dones)

        for t in range(total_steps):
            last_step = min(self.n_steps, total_steps - t)

            # Look for end of episode
            check_episode_completion = dones[t:t + last_step]
            if check_episode_completion.size > 0:
                if True in check_episode_completion:
                    next_ep_completion = np.where(check_episode_completion == True)[0][0]
                    last_step = next_ep_completion

            # Sum and discount rewards
            G[t] = sum([rewards[t + n:t + n + 1] * self.gamma ** n for
                        n in range(last_step)])

        if total_steps > self.n_steps:
            G[:total_steps - self.n_steps] += next_state_values[self.n_steps:] \
                                              * self.gamma ** self.n_steps
        td_delta = G - state_values
        return G, td_delta

    def train(self, n_steps=5, batch_size=10, num_episodes=5000,
              gamma=0.99, beta=1e-3, zeta=1e-3):
        self.n_steps = n_steps
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.beta = beta
        self.zeta = zeta
        self.batch_size = batch_size

        self.env = Skyscraper()
        self.s_0, _, _ = self.env.get_state()
        self.reward = 0
        self.ep_counter = 0
        while self.ep_counter < num_episodes:
            batch = self.generate_episode()
            G, td_delta = self.calc_rewards(batch)
            states = batch[0]
            actions = batch[1]

            self.update(states, actions, G, td_delta)

            print("\rMean Rewards: {:.6f} Episode: {:d}  Episode Rewards: {:.6f}  ".format(
                np.mean(self.ep_rewards[-100:]), self.ep_counter, self.ep_rewards[-1]))

            if self.ep_counter % 20 == 0:
                filepath = f'ep_{self.ep_counter}'
                torch.save(self.network, f'models/three_elevator/{filepath}_{self.ep_rewards[-1]:.2f}.pt')
                self.save_metrics(f'{filepath}.json')
                print("Saved Model and Metrics")

    def save_metrics(self, filepath):
        metrics = {
            "ep_rewards": self.ep_rewards,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy_loss": self.entropy_loss,
            "total_policy_loss": self.total_policy_loss,
            "total_loss": self.total_loss
        }
        json_object = json.dumps(metrics, indent=4)
        with open(f'models/three_elevator/metrics/{filepath}', "w") as outfile:
            outfile.write(json_object)

    def calc_loss(self, states, actions, rewards, advantages, beta=0.001):
        actions_t = torch.LongTensor(actions).to(self.network.device)
        rewards_t = torch.FloatTensor(rewards).to(self.network.device)
        advantages_t = torch.FloatTensor(advantages).to(self.network.device)

        log_probs = self.network.get_log_probs(states)
        log_prob_actions = advantages_t * log_probs[range(len(actions)), actions]
        policy_loss = -log_prob_actions.mean()

        action_probs, values = self.network.predict(states)
        entropy_loss = -self.beta * (action_probs * log_probs).sum(dim=1).mean()

        value_loss = self.zeta * torch.nn.MSELoss()(values.squeeze(-1), rewards_t)

        # Append values
        self.policy_loss.append(policy_loss.item())
        self.value_loss.append(value_loss.item())
        self.entropy_loss.append(entropy_loss.item())

        return policy_loss, entropy_loss, value_loss

    def update(self, states, actions, rewards, advantages):
        self.network.optimizer.zero_grad()
        policy_loss, entropy_loss, value_loss = self.calc_loss(states, actions, rewards, advantages)

        total_policy_loss = policy_loss - entropy_loss
        self.total_policy_loss.append(total_policy_loss.item())
        total_policy_loss.backward(retain_graph=True)

        value_loss.backward()

        total_loss = policy_loss + value_loss + entropy_loss
        self.total_loss.append(total_loss.item())
        self.network.optimizer.step()

"""T"""
net = A2CNetwork()
a2c = A2C(Skyscraper(), net)
a2c.train(n_steps=50, num_episodes=20000, beta=0.1, zeta=1e-3)


"""Test
net = torch.load("models/three_elevator/ep_4800_6.51.pt")
sky = Skyscraper()
a2c = A2C(sky, net)
states, actions, rewards, dones, next_states = a2c.generate_episode(isTrain=False)
print(actions)
"""