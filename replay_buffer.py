import numpy as np
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
        self.actions = np.zeros((self.capacity, *self.action_shape), dtype=np.int8)
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

            self.observations[self.position] = obs
            self.actions[self.position] = action
            self.rewards[self.position] = reward
            self.next_observations[self.position] = next_obs
            self.dones[self.position] = done
            self.log_probs[self.position] = log_prob

            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

            self.queue.task_done()

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
