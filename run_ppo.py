import os.path
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, freeze_support
from torch.distributions import Categorical
# Set the number of threads PyTorch will use
torch.set_num_threads(4)

from networks import ActorLinear, CriticLinear
from env import GymEnvironment
from replay_buffer import ReplayBuffer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Ignore numpy deprecation warnings if numpy version > 1.24


# TODO: transfer the neural network of the Actor Critcs to the PPO class instead of initializing it in the class.
class PPO:
    def __init__(self,
                 env_name: str,
                 n_workers: int,
                 num_transitions: int,
                 num_samples: int,
                 k: int,
                 device: torch.device,
                 actor_lr: float = 0.0003,
                 critic_lr: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 0.2):

        self.env_name = env_name

        self.dummy_env = GymEnvironment(env_name, show_render=False)
        self.state_dim = self.dummy_env.state_shape[0]
        self.action_dim = self.dummy_env.num_actions

        # init networks
        self.device = device
        self.actor = ActorLinear(self.state_dim, self.action_dim).to(self.device)
        self.critic = CriticLinear(self.state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.critic.parameters(), lr=critic_lr)

        self.n_workers = n_workers
        self.T = num_transitions
        self.m = num_samples
        self.k = k
        self.gamma = gamma
        self.epsilon = epsilon

        self.replay_buffer = ReplayBuffer(capacity=self.T * self.n_workers,
                                          state_dim=self.state_dim,
                                          action_dim=self.action_dim,
                                          device=self.device)

    def worker(self):
        env = GymEnvironment(self.env_name)  # Create an instance of the environment
        state = env.reset()

        for _ in range(self.T):
            action_probs = self.actor(state)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()

            next_state, reward, done = env.step(action.item())
            self.replay_buffer.add(state, action, reward, next_state, done, action_dist.log_prob(action))

            if done:
                state = env.reset()
            else:
                state = next_state

        env.close()

    def collect_transitions(self, multiprocessing: bool = True):
        if multiprocessing:
            processes = []
            for i in range(self.n_workers):
                p = mp.Process(target=self.worker)
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        else:
            for _ in range(self.n_workers):
                self.worker()

    def compute_advantages(self, gae_lambda: float = 0.95):
        states, actions, rewards, next_states, dones, _ = self.replay_buffer.get_all()

        with torch.no_grad():
            state_values = self.critic(states)
            next_state_values = self.critic(next_states)

        # Compute TD errors
        td_errors = rewards + self.gamma * next_state_values * (1 - dones) - state_values

        # Compute advantages
        advantages = td_errors.clone()
        for t in reversed(range(len(self.replay_buffer) - 1)):
            advantages[t] += self.gamma * gae_lambda * advantages[t + 1] * (1 - dones[t])

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute returns
        returns = advantages + state_values

        self.replay_buffer.advantages[:len(self.replay_buffer)] = advantages
        self.replay_buffer.returns[:len(self.replay_buffer)] = returns

    def update(self, batch_size: int):
        self.compute_advantages()

        for _ in range(self.k):
            states, actions, _, _, _, old_action_probs, advantages, returns = self.replay_buffer.sample(batch_size)

            # Compute new actor and critic states
            new_action_probs = self.actor(states)
            new_action_dist = Categorical(new_action_probs)
            state_values = self.critic(states)

            # Actor loss
            ratio = torch.exp(new_action_dist.log_prob(new_action_dist.sample()) - old_action_probs)
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            # Critic loss
            critic_loss = nn.MSELoss()(state_values.squeeze(), returns.squeeze())

            # Update actor and critic
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def train(self, num_episodes: int):
        for episode in range(num_episodes):
            self.collect_transitions()
            self.update(self.m)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1} completed")

    def save_networks(self, path: str):
        if path[-1] != '/':
            path += '/'

        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.actor.state_dict(), path + 'ppo_actor.pth')
        torch.save(self.critic.state_dict(), path + 'ppo_critic.pth')

    def load_network(self, path: str):
        if path[-1] != '/':
            path += '/'

        self.actor.load_state_dict(torch.load(path + 'ppo_actor.pth'))
        self.critic.load_state_dict(torch.load(path + 'ppo_critic.pth'))

    def display_trained_agent(self,
                              num_episodes: int = 5,
                              max_steps: int = 1000):

        print("Displaying trained agent...")
        self.dummy_env = GymEnvironment(self.env_name, show_render=True)
        for episode in range(num_episodes):
            state = self.dummy_env.reset()
            total_reward = 0
            done = False
            step = 0

            while not done and step < max_steps:
                self.dummy_env.render()
                action_probs = self.actor(state)

                action_dist = Categorical(action_probs)
                action = action_dist.sample()

                state, reward, done = self.dummy_env.step(action.item())
                total_reward += reward
                step += 1

            print(f"Episode {episode + 1} finished with total reward: {total_reward}")

        self.dummy_env.close()


if __name__ == "__main__":
    # Example usage
    freeze_support()

    parser = argparse.ArgumentParser(description="PyTorch model with CUDA/CPU option")
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu',
                        help='Device to run the model on (cuda or cpu)')

    # Parse arguments
    args = parser.parse_args()

    # Set the device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    n_workers = 4
    T = 200
    m = 64
    k = 10

    # init model
    ppo = PPO("CartPole-v1", n_workers, T, m, k, device=device)
    ppo.train(num_episodes=500)

    # Display the trained agent
    ppo.display_trained_agent()

    # Save the model
    ppo.save_networks(f"results/params_n_workers_{n_workers}_T_{T}_m_{m}_k_{k}/ppo_model.pth")
