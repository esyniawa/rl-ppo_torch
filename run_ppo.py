import os.path
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, freeze_support
from torch.distributions import Categorical

from networks import ActorLinear, CriticLinear
from env import VectorizedEnvironment
from replay_buffer import QueuedReplayBuffer

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
                 epsilon: float = 0.2,
                 gae_lambda: float = 0.95):

        # Environment
        self.env_name = env_name
        self.vec_env = VectorizedEnvironment(env_name, n_workers)

        # init networks
        self.device = device
        self.actor = ActorLinear(self.vec_env.state_shape[0], self.vec_env.num_actions).to(self.device)
        self.critic = CriticLinear(self.vec_env.state_shape[0]).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.critic.parameters(), lr=critic_lr)

        self.n_workers = n_workers
        self.T = num_transitions
        self.m = num_samples
        self.k = k
        self.gamma = gamma
        self.epsilon = epsilon
        self.gae_lambda = gae_lambda

        # Initialize replay buffer
        self.replay_buffer = QueuedReplayBuffer(
            capacity=self.T * self.n_workers,
            obs_shape=self.vec_env.state_shape,
            action_shape=self.vec_env.action_shape,
        )

    def collect_transitions(self):
        states = self.vec_env.reset()

        for _ in range(self.T):
            # Convert states to tensor and move to device
            state_tensor = torch.FloatTensor(states).to(self.device)

            # Get action probabilities
            with torch.no_grad():
                action_probs = self.actor(state_tensor)

            # Sample actions
            action_dist = Categorical(action_probs)
            actions = action_dist.sample()

            log_probs = action_dist.log_prob(actions)

            # Take steps in environments
            next_states, rewards, dones = self.vec_env.step(actions.cpu().numpy())

            # Add transitions to replay buffer
            for state, action, reward, next_state, done, log_prob in zip(states, actions, rewards, next_states, dones, log_probs):
                self.replay_buffer.add(state, action, reward, next_state, done, log_prob.item())

            states = next_states

    def compute_advantages(self, states, rewards, next_states, dones):
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()

        # Calculate TD errors and advantages
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(rewards)

        running_advantage = 0
        for t in reversed(range(len(rewards))):
            running_advantage = deltas[t] + self.gamma * self.gae_lambda * running_advantage * (1 - dones[t])
            advantages[t] = running_advantage

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate returns
        returns = advantages + values

        return advantages, returns

    def update(self, batch_size):
        """
        Performs a PPO update. Calculating advantages for smaller batches with online updates.
        :param batch_size: int
        :return:
        """

        for _ in range(self.k):
            # Sample a batch from the replay buffer
            states, actions, rewards, next_states, dones, old_log_probs = self.replay_buffer.sample(batch_size)

            # Convert to tensors and move to device
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)

            # Compute advantages and returns for this batch
            advantages, returns = self.compute_advantages(states, rewards, next_states, dones)

            # Get current action probabilities and values
            action_probs = self.actor(states)
            current_values = self.critic(states).squeeze()

            # Calculate ratio and surrogate objectives
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            # Calculate value function loss
            value_loss = nn.MSELoss()(current_values, returns)

            # Update actor and critic
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

    def train(self, num_episodes):
        for episode in range(num_episodes):
            self.collect_transitions()
            self.update(self.m)  # m is the batch size for updates

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
        pass

def train_ppo(
    env_name: str,
    n_workers: int,
    num_transitions: int,
    num_samples: int,
    k: int,
    device: torch.device
):
    pass

def test_ppo(
    env_name: str,
    n_workers: int,
    num_transitions: int,
    num_samples: int,
    k: int,
    device: torch.device
):
    pass


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
    T = 1_000
    m = 64
    k = 100

    # init model
    ppo = PPO(env_name="MountainCar",
              n_workers=n_workers,
              num_transitions=T,
              num_samples=m,
              k=k,
              device=device)

    ppo.train(num_episodes=5)

    # Display the trained agent
    ppo.display_trained_agent()

    # Save the model
    ppo.save_networks(f"results/params_n_workers_{n_workers}_T_{T}_m_{m}_k_{k}/ppo_model.pth")
