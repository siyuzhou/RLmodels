import copy
import random

import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)

        random.seed(seed)

        self.theta = theta
        self.sigma = sigma

        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu.)"""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * \
            np.array([np.random.rand() for i in range(len(self.state))])
        self.state += dx
        return self.state
