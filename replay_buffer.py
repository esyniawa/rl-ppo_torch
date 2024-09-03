import numpy as np
import torch
from typing import List, Tuple
from queue import Queue
from threading import Thread


class QueuedReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...] | int, action_shape: Tuple[int, ...] | int):
        self.capacity = capacity

        if isinstance(obs_shape, int):
            self.obs_shape = (obs_shape,)
        elif isinstance(obs_shape, tuple):
            self.obs_shape = obs_shape
        else:
            raise ValueError("Observation dimension must be an integer or a tuple of integers")

        if isinstance(action_shape, int):
            self.action_shape = (action_shape,)
        elif isinstance(action_shape, tuple):
            self.action_shape = action_shape
        else:
            raise ValueError("Action dimension must be an integer or a tuple of integers")


        self.queue = Queue()

        self.observations = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, *self.action_shape), dtype=np.uint8)
        self.rewards = np.zeros(self.capacity, dtype=np.float16)
        self.next_observations = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.bool_)
        self.log_probs = np.zeros(self.capacity, dtype=np.float16)

        self.position = 0
        self.size = 0

        self.worker = Thread(target=self._process_queue, daemon=True)
        self.worker.start()

    def add(self, obs, action, reward, next_obs, done, log_prob):
        self.queue.put((obs, action, reward, next_obs, done, log_prob))

    def _process_queue(self):
        while True:
            obs, action, reward, next_obs, done, log_prob = self.queue.get()

            # Move tensors to CPU and convert to numpy arrays
            self.observations[self.position] = self._to_numpy(obs, dtype=np.float32)
            self.actions[self.position] = self._to_numpy(action, dtype=np.uint8)
            self.rewards[self.position] = self._to_numpy(reward, dtype=np.float16)
            self.next_observations[self.position] = self._to_numpy(next_obs, dtype=np.float32)
            self.dones[self.position] = self._to_numpy(done, dtype=np.bool_)
            self.log_probs[self.position] = self._to_numpy(log_prob, dtype=np.float16)

            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

            self.queue.task_done()

    @staticmethod
    def _to_numpy(x, dtype=np.float32):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy().astype(dtype)
        return np.array(x, dtype=dtype)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
            self.log_probs[indices]
        )
