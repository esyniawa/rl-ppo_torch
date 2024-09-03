import gym
import numpy as np
from typing import List, Tuple


class VectorizedEnvironment:
    def __init__(self, env_name: str, num_envs: int):
        self.envs = [gym.make(env_name) for _ in range(num_envs)]
        self.num_envs = num_envs

        self.observations = self.reset()

    def reset(self) -> np.ndarray:
        results = [env.reset() for env in self.envs]
        states, _ = zip(*results)
        return np.array(states)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        new_obs, rewards, dones, _, _ = zip(*results)

        new_obs = np.array(new_obs)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Reset environments that have terminated
        for i, done in enumerate(dones):
            if done:
                new_obs[i], _ = self.envs[i].reset()

        self.observations = new_obs
        return np.array(new_obs), np.array(rewards), np.array(dones)

    def close(self):
        for env in self.envs:
            env.close()

    @property
    def state_shape(self):
        return self.envs[0].observation_space.shape

    @property
    def num_actions(self):
        return self.envs[0].action_space.n

    @property
    def action_shape(self):
        if isinstance(self.envs[0].action_space, gym.spaces.discrete.Discrete):
            return (1,)

if __name__ == "__main__":
    vec_env = VectorizedEnvironment("CartPole-v1", 2)

    print(vec_env.state_shape, vec_env.num_actions, vec_env.action_shape)