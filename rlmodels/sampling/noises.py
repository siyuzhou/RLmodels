import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed=None, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)

        self.theta = theta
        self.sigma = sigma

        self.random = np.random.RandomState(seed)

        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu.)"""
        self.state = self.mu.copy()

    def sample(self):
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * \
            np.array([self.random.rand() for i in range(len(self.state))])
        self.state += dx
        return self.state
