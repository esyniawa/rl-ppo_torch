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
torch.set_num_threads(8)

from dnn import ActorCriticLinear
from env import GymEnvironment

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Ignore numpy deprecation warnings if numpy version > 1.24

# TODO: Give a certain neural network architecture as argument to class
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

        self.device = device
        self.actor_critic = ActorCriticLinear(self.state_dim, self.action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_critic.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.actor_critic.critic.parameters(), lr=critic_lr)

        self.n_workers = n_workers
        self.T = num_transitions
        self.m = num_samples
        self.k = k
        self.gamma = gamma
        self.epsilon = epsilon

    @staticmethod
    def worker(env_name: str,
               actor_critic: nn.Module,
               device: str,
               T: int):

        env = GymEnvironment(env_name)  # Create an instance of the environment
        transitions = []
        state = env.reset()

        for _ in range(T):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs, state_value = actor_critic(state_tensor)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()

            next_state, reward, done = env.step(action.item())

            transitions.append(
                (state, action.item(),
                 reward, next_state,
                 done, action_probs.cpu().detach(),
                 state_value.cpu().detach())
            )

            if done:
                state = env.reset()
            else:
                state = next_state

        env.close()
        return transitions

    def collect_transitions(self, multiprocessing: bool = False):
        if multiprocessing:
            mp.set_start_method('spawn', force=True)  # This is important for CUDA tensors
            with Pool(self.n_workers) as p:
                results = p.starmap(self.worker, [(self.env_name, self.actor_critic, self.device, self.T)] * self.n_workers)

            all_transitions = []
            for worker_transitions in results:
                all_transitions.extend(worker_transitions)

            return all_transitions

        else:
            all_transitions = []
            for _ in range(self.n_workers):
                transitions = self.worker(self.env_name, self.actor_critic, self.device, self.T)
                all_transitions.extend(transitions)
            return all_transitions

    def compute_advantages(self, transitions: list):
        advantages = []
        returns = []

        for t in reversed(range(len(transitions))):
            state, _, reward, next_state, done, _, state_value = transitions[t]
            if t == len(transitions) - 1:
                next_value = 0 if done else \
                self.actor_critic(torch.FloatTensor(next_state).unsqueeze(0).to(self.device))[1].cpu().item()
            else:
                next_value = transitions[t + 1][6].item()

            td_error = reward + self.gamma * next_value * (1 - done) - state_value.item()
            advantages.insert(0, td_error)
            returns.insert(0, td_error + state_value.item())

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        return advantages, returns

    def update(self,
               transitions: list,
               advantages: torch.Tensor,
               returns: torch.Tensor):

        states = torch.FloatTensor(np.array([t[0] for t in transitions])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in transitions])).to(self.device)
        old_action_probs = torch.cat([t[5] for t in transitions]).to(self.device)

        for _ in range(self.k):
            indices = torch.randperm(len(transitions))[:self.m]

            sampled_states = states[indices]
            sampled_actions = actions[indices]
            sampled_advantages = advantages[indices]
            sampled_returns = returns[indices]
            sampled_old_action_probs = old_action_probs[indices]

            new_action_probs, state_values = self.actor_critic(sampled_states)
            new_action_dist = Categorical(new_action_probs)

            # Actor loss
            ratio = torch.exp(new_action_dist.log_prob(sampled_actions) - torch.log(
                sampled_old_action_probs.gather(1, sampled_actions.unsqueeze(1)).squeeze()))
            surrogate1 = ratio * sampled_advantages
            surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * sampled_advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            # Critic loss
            critic_loss = nn.MSELoss()(state_values.squeeze(), sampled_returns)

            # Update actor and critic
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def train(self, num_episodes: int):
        for episode in range(num_episodes):
            transitions = self.collect_transitions()
            advantages, returns = self.compute_advantages(transitions)
            self.update(transitions, advantages, returns)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1} completed")

    def save_network(self, path: str):
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder)

        torch.save(self.actor_critic.state_dict(), path)

    def load_network(self, path: str):
        self.actor_critic.load_state_dict(torch.load(path))

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
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_probs, _ = self.actor_critic(state_tensor)
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
    ppo.train(num_episodes=250)

    # Display the trained agent
    ppo.display_trained_agent()
