import datetime
import json
import time
import wandb
import torch.nn.functional as F
import torch.optim
import numpy as np
from reinforcement.network import A2CNetwork
from skyscraper import Skyscraper


wandb.init(project="skyscraper")
torch.set_num_threads(10)

ACTION_ENCODING = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0],
                   [0, 2, 1], [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1],
                   [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0],
                   [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]]


def run_train(num_episodes, gamma=0.99, ):
    print("Start Training")
    a2c_net = A2CNetwork()
    policy_optimizer = torch.optim.Adam(a2c_net.actor_net.parameters(), lr=0.001)
    value_optimizer = torch.optim.Adam(a2c_net.critic_net.parameters(), lr=0.001)

    rewards_log = []
    policy_log = []
    value_log = []
    num_transported_log = []
    num_created_log = []
    avg_q_time = []
    avg_t_time = []

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

        Qval, _ = a2c_net.forward(state_1)
        Qvals = []
        for t in reversed(range(len(episode_rewards))):
            Qval = episode_rewards[t] + gamma * Qval * episode_dones[t]
            Qvals.insert(0,Qval)

        values = torch.cat(episode_values)
        Qvals = torch.cat(Qvals).detach()
        log_probs = torch.stack(episode_log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        total_loss = actor_loss + critic_loss

        start_time_optim = time.perf_counter()
        policy_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        policy_optimizer.step()

        value_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        value_optimizer.step()
        end_time_optim = time.perf_counter()

        # save model
        if episode % 10 == 0:
            state = {
                "episode": episode,
                "state_dict": a2c_net.state_dict(),
                "actor_optimizer": policy_optimizer.state_dict(),
                "critic_optimizer": value_optimizer.state_dict()
            }
            torch.save(state, f'models/checkpoint_{episode}_{rewards_log[-1]:.1f}.pt')

        wandb.log({"total_loss": total_loss.item(),
                   "policy loss": actor_loss.item(),
                   "value loss": critic_loss.item(),
                   "reward": rewards_log[-1],
                   "num_transported": environment.num_transported_passengers})

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


def save_metrics(name, metrics):
    json_object = json.dumps(metrics, indent=4)
    with open(f"models/metrics/{name}", "w") as outfile:
        outfile.write(json_object)


def run_train_one_step(num_episodes, gamma=0.99):
    print("Start Training")
    a2c_net = A2CNetwork()
    # a2c_optimizer = torch.optim.Adam(a2c_net.parameters(), lr=a2c_net.learning_rate)

    for episode in range(num_episodes):
        environment = Skyscraper()
        s_0, _, done = environment.get_state()

        actions_log = []
        total_reward = 0
        total_loss = 0
        actor_loss = 0
        critic_loss = 0
        counter = 0
        while not done:
            value, policy_dist = a2c_net.forward(s_0)
            dist = policy_dist.cpu().detach().numpy()

            action = np.random.choice(len(ACTION_ENCODING), p=np.squeeze(dist))
            environment.schedule_action(ACTION_ENCODING[action])
            s_1, r, done = environment.step()
            total_reward += r

            # Logging
            actions_log.append(action)

            next_value, _ = a2c_net.forward(s_1)
            advantage = r + (1-done) * gamma * next_value - value

            zero_filtered_dist = np.where(dist > 0.0000000001, dist, -10)
            zero_filtered_dist = np.log(zero_filtered_dist, out=zero_filtered_dist, where=zero_filtered_dist > 0)
            entropy = -np.sum(np.mean(dist) * zero_filtered_dist)
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            actor_loss = -log_prob * advantage.detach() + 0.01 * entropy
            critic_loss = advantage.pow(2) + 0.01 * entropy
            total_loss = actor_loss + critic_loss + 0.01 * entropy

            a2c_net.actor_optimizer.zero_grad()
            actor_loss.backward()
            a2c_net.actor_optimizer.step()

            a2c_net.critic_optimizer.zero_grad()
            critic_loss.backward()
            a2c_net.critic_optimizer.step()

            s_0 = s_1
            counter += 1

        print(f'Episode {episode}, Rewards: {total_reward}')
        if episode % 10 == 0:
            state = {
                "episode": episode,
                "state_dict": a2c_net.state_dict(),
                "optimizer_actor": a2c_net.actor_optimizer.state_dict(),
                "optimizer_critic": a2c_net.critic_optimizer.state_dict()
            }
            torch.save(state, f'models/three_elevator/checkpoint_one_{episode}.pt')


def run_inference(save_filepath, environment):
    a2c_net = A2CNetwork()
    model_state = torch.load(save_filepath)
    a2c_net.load_state_dict(model_state["state_dict"])
    state, _, done = environment.get_state()
    while not done:
        value, policy_dist = a2c_net.forward(state)
        dist = policy_dist.cpu().detach().numpy()
        action = np.random.choice(len(ACTION_ENCODING), p=np.squeeze(dist))
        environment.schedule_action(ACTION_ENCODING[action])
        state_1, reward, done = environment.step()
        state = state_1

if __name__ == "__main__":
    run_train(num_episodes=15000)
