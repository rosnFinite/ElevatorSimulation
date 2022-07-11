import torch.optim
import numpy as np
from reinforcement.network import A2CNetwork
from skyscraper import Skyscraper

ACTION_ENCODING = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0],
                   [0, 2, 1], [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1],
                   [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0],
                   [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]]


def run_train(num_episodes, gamma=0.99):
    a2c_net = A2CNetwork()
    a2c_optimizer = torch.optim.Adam(a2c_net.parameters(), lr=a2c_net.learning_rate)
    rewards_log = []

    entropy_term = 0
    for episode in range(num_episodes):
        episode_rewards = []
        episode_values = []
        episode_log_probs = []

        environment = Skyscraper()
        state, _, done = environment.get_state()
        while not done:
            value, policy_dist = a2c_net.forward(state)
            value = value.detach().numpy()
            dist = policy_dist.detach().numpy()

            action = np.random.choice(len(ACTION_ENCODING), p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))

            environment.schedule_action(ACTION_ENCODING[action])
            state_1, reward, done = environment.step()
            state = state_1

            episode_rewards.append(reward)
            episode_values.append(value)
            episode_log_probs.append(log_prob)
            entropy_term += entropy

            if done:
                rewards_log.append(np.sum(episode_rewards))
                Qval, _ = a2c_net.forward(state_1)
                Qval = Qval.detach().numpy()

        Qvals = np.zeros_like(episode_values)
        for t in reversed(range(len(episode_rewards))):
            Qval = episode_rewards[t] + gamma * Qval
            Qvals[t] = Qval

        values = torch.FloatTensor(episode_values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(episode_log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        total_loss = actor_loss + critic_loss + 0.01 * entropy_term

        a2c_optimizer.zero_grad()
        total_loss.backward()
        a2c_optimizer.step()

        if episode % 10 == 0:
            state = {
                "episode": episode,
                "state_dict": a2c_net.state_dict(),
                "optimizer": a2c_optimizer.state_dict()
            }
            torch.save(state, f'models/three_elevator/checkpoint_{episode}.pt')

        print(f'Episode: {episode}, Reward: {rewards_log[-1]}')


def run_inference(save_filepath, environment):
    a2c_net = A2CNetwork()
    model_state = torch.load(save_filepath)
    a2c_net.load_state_dict(model_state["state_dict"])
    state, _, done = environment.get_state()
    while not done:
        value, policy_dist = a2c_net.forward(state)
        dist = policy_dist.detach().numpy()
        action = np.random.choice(len(ACTION_ENCODING), p=np.squeeze(dist))
        environment.schedule_action(ACTION_ENCODING[action])
        state_1, reward, done = environment.step()
        state = state_1


if __name__ == "__main__":
    run_train(num_episodes=1000)