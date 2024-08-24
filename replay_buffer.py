import torch
import torch.multiprocessing as mp
import numpy as np

torch.set_num_threads(4)  # Set the number of threads PyTorch will use


class ReplayBuffer:
    def __init__(self,
                 capacity: int,
                 state_dim: int | tuple,
                 action_dim: int | tuple,
                 device: torch.device):
        """
        Initializes a ReplayBuffer object with the given capacity, state dimension, action dimension, and device.

        Parameters:
            capacity (int): The maximum capacity of the replay buffer.
            state_dim (int | tuple): The dimension of the state space.
            action_dim (int | tuple): The dimension of the action space.
            device (torch.device): The device the sample of the data is sent to.

        Returns:
            None
        """

        assert capacity > 0, "Capacity must be greater than 0"
        self.capacity = capacity
        self.device = device

        if isinstance(state_dim, int):
            self.state_dim = (state_dim,)
        elif isinstance(state_dim, tuple):
            self.state_dim = state_dim
        else:
            raise ValueError("State dimension must be an integer or a tuple of integers")

        if isinstance(action_dim, int):
            self.action_dim = (action_dim,)
        elif isinstance(action_dim, tuple):
            self.action_dim = action_dim
        else:
            raise ValueError("Action dimension must be an integer or a tuple of integers")

        # Use shared memory for parallel access
        self.states = torch.zeros((capacity, *self.state_dim), dtype=torch.float32).share_memory_()
        self.actions = torch.zeros((capacity, *self.action_dim), dtype=torch.float32).share_memory_()
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32).share_memory_()
        self.next_states = torch.zeros((capacity, *self.state_dim), dtype=torch.float32).share_memory_()
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32).share_memory_()
        self.log_probs = torch.zeros((capacity, 1), dtype=torch.float32).share_memory_()

        # Add new tensors for advantages and returns
        self.advantages = torch.zeros((capacity, 1), dtype=torch.float32).share_memory_()
        self.returns = torch.zeros((capacity, 1), dtype=torch.float32).share_memory_()

        self.idx = mp.Value('i', 0)
        self.size = mp.Value('i', 0)
        self.lock = mp.Lock()

    def add(self,
            state: np.ndarray | torch.Tensor,
            action: np.ndarray | torch.Tensor,
            reward: float | np.ndarray | torch.Tensor,
            next_state: np.ndarray | torch.Tensor,
            done: bool | np.ndarray | torch.Tensor,
            log_prob: float):

        with self.lock:
            idx = self.idx.value
            self.states[idx] = torch.as_tensor(state, dtype=torch.float32)
            self.actions[idx] = torch.as_tensor(action, dtype=torch.float32)
            self.rewards[idx] = torch.as_tensor(reward, dtype=torch.float32)
            self.next_states[idx] = torch.as_tensor(next_state, dtype=torch.float32)
            self.dones[idx] = torch.as_tensor(done, dtype=torch.float32)
            self.log_probs[idx] = torch.as_tensor(log_prob, dtype=torch.float32)

            self.idx.value = (idx + 1) % self.capacity
            self.size.value = np.clip(self.size.value + 1, a_min=0, a_max=self.capacity)

    def get_all(self):

        # Transfer to specified device with non-blocking option
        states = self.states.to(self.device, non_blocking=True)
        next_states = self.next_states.to(self.device, non_blocking=True)

        actions = self.actions.to(self.device, non_blocking=True)
        log_probs = self.log_probs.to(self.device, non_blocking=True)

        rewards = self.rewards.to(self.device, non_blocking=True)
        dones = self.dones.to(self.device, non_blocking=True)

        if self.device == torch.device("cuda"):
            torch.cuda.synchronize()

        return states, actions, rewards, next_states, dones, log_probs

    def sample(self, batch_size: int):
        indices = torch.randint(0, len(self), (batch_size,))
        return (
            self.states[indices].to(self.device, non_blocking=True),
            self.actions[indices].to(self.device, non_blocking=True),
            self.rewards[indices].to(self.device, non_blocking=True),
            self.next_states[indices].to(self.device, non_blocking=True),
            self.dones[indices].to(self.device, non_blocking=True),
            self.log_probs[indices].to(self.device, non_blocking=True),
            self.advantages[indices].to(self.device, non_blocking=True),
            self.returns[indices].to(self.device, non_blocking=True)
        )

    def __len__(self):
        return self.size.value

# Example usage:
if __name__ == "__main__":
    import time

    state_dim = 4
    action_dim = 2
    buffer_capacity = 100000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim, device)

    # Adding transitions
    for _ in range(buffer_capacity + buffer_capacity):
        state = np.random.rand(state_dim)
        action = np.random.rand(action_dim)
        reward = np.random.rand()
        next_state = np.random.rand(state_dim)
        done = False
        action_prob = np.random.rand(action_dim)
        buffer.add(state, action, reward, next_state, done, action_prob)

    # Sampling from the buffer
    batch_size = 256
    start_time = time.time()
    states, actions, rewards, next_states, dones, actions_probs = buffer.get_all()
    end_time = time.time()

    print(f"Time taken to sample and transfer to {device}: {end_time - start_time:.4f} seconds")
    print(f"States shape: {states.shape}, device: {states.device}")
    print(f"Actions shape: {actions.shape}, device: {actions.device}")
    print(f"Rewards shape: {rewards.shape}, device: {rewards.device}")
    print(f"Next states shape: {next_states.shape}, device: {next_states.device}")
    print(f"Dones shape: {dones.shape}, device: {dones.device}")
    print(f"Actions probs shape: {actions_probs.shape}, device: {actions_probs.device}")

    print(len(buffer))
    # Ensure all operations are completed (important when using non_blocking=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()