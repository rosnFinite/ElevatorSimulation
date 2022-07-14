import datetime
import time
import torch.optim
import numpy as np
from reinforcement.network import A2CNetwork
from skyscraper import Skyscraper

# weights and biases is used to keep track of previous runs
# wandb.init(project="skyscraper")
# torch.set_num_threads(10)

ACTION_ENCODING = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0],
                   [0, 2, 1], [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1],
                   [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0],
                   [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]]


def run_train(num_episodes, gamma=0.99, ):
    print("Start Training")
    a2c_net = A2CNetwork()
    optimizer = torch.optim.Adam(a2c_net.actor_net.parameters(), lr=0.01)

    rewards_log = []

    for episode in range(num_episodes):
        episode_rewards = []
        episode_values = []
        episode_log_probs = []
        episode_actions = []
        episode_dones = []

        environment = Skyscraper()
        state, _, done = environment.get_state()
        start_time_episode = time.perf_counter()
        while not done:
            value, policy_dist = a2c_net.forward(state)

            action = policy_dist.sample()
            log_prob = policy_dist.log_prob(action).unsqueeze(0)

            environment.schedule_action(ACTION_ENCODING[action])
            state_1, reward, done = environment.step()
            state = state_1

            episode_rewards.append(reward)
            episode_values.append(value)
            episode_log_probs.append(log_prob)
            episode_actions.append(action)
            episode_dones.append(done)

            if done:
                rewards_log.append(np.sum(episode_rewards))

        end_time_episode = time.perf_counter()

        Qval = 0
        Qvals = []
        for t in reversed(range(len(episode_rewards))):
            Qval = episode_rewards[t] + gamma * Qval * episode_dones[t]
            Qvals.insert(0,Qval)

        values = torch.cat(episode_values)
        Qvals = torch.tensor(Qvals).detach()
        log_probs = torch.stack(episode_log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        total_loss = actor_loss + critic_loss

        start_time_optim = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        end_time_optim = time.perf_counter()

        # save model to restart training
        if episode % 10 == 0:
            state = {
                "episode": episode,
                "state_dict": a2c_net.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, f'models/checkpoint_{episode}_{rewards_log[-1]:.1f}.pt')

        """ Uncomment if weights and biases is used
        wandb.log({"total_loss": total_loss.item(),
                   "policy loss": actor_loss.item(),
                   "value loss": critic_loss.item(),
                   "reward": rewards_log[-1],
                   "num_transported": environment.num_transported_passengers})
        """

        mean_q = -1
        mean_t = -1
        if environment.num_transported_passengers != 0:
            mean_q = datetime.timedelta(seconds=environment.mean_queue_time)
            mean_t = datetime.timedelta(seconds=environment.mean_travel_time)

        print(f'Episode: {episode}\nActor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, '
              f'Total Loss: {total_loss.item():.4f}\nReward: {rewards_log[-1]:.4f} \n'
              f'Episode Time: {end_time_episode-start_time_episode:.3f}, '
              f'Optim Time: {end_time_optim-start_time_optim:.3f}\n'
              f'------------------------------SIM STATS------------------------------\n'
              f'Transported/Generated: {environment.num_transported_passengers} / {environment.num_generated_passengers}\n'
              f'Mean. waiting: {mean_q}\n'
              f'Mean. transport: {mean_t}\n'
              f'=====================================================================')


def run_inference(save_filepath, environment):
    a2c_net = A2CNetwork()
    model_state = torch.load(save_filepath)
    a2c_net.load_state_dict(model_state["state_dict"])
    state, _, done = environment.get_state()
    while not done:
        value, policy_dist = a2c_net.forward(state)
        action = policy_dist.sample()
        environment.schedule_action(ACTION_ENCODING[action])
        state_1, reward, done = environment.step()
        state = state_1


if __name__ == "__main__":
    run_train(num_episodes=15000)
