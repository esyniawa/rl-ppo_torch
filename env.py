import gym
import torch

# TODO: Make this class more general and implement other environments like Atari Env or others
class GymEnvironment:
    def __init__(self,
                 env_name: str,
                 show_render: bool = False):

        if show_render:
            self.env = gym.make(env_name, render_mode="human")
        else:
            self.env = gym.make(env_name)

    def reset(self, device: torch.device = "cpu"):
        """Reset the environment and return the initial state."""
        initial_state, _ = self.env.reset()
        return torch.as_tensor(initial_state, device=device)

    def step(self, action, device: torch.device = "cpu"):
        """
        Take a step in the environment.

        :param action: The action to take
        :return: next_state, reward, done
        """
        next_state, reward, done, _, _ = self.env.step(action)
        return torch.as_tensor(next_state, device=device), reward, done

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    @property
    def num_actions(self):
        """Return the number of possible actions."""
        return self.env.action_space.n

    @property
    def state_shape(self):
        """Return the shape of the state space."""
        return self.env.observation_space.shape

if __name__ == "__main__":
    env = GymEnvironment("CartPole-v1")
    state = env.reset()
    print(state)

    print(env.state_shape, env.num_actions)


