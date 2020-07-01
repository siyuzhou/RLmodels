import abc
import numpy as np

from .base_sampling import BaseSampling


class Clipping(BaseSampling):
    def __init__(self, low, high):
        self.low = np.array(low)
        self.high = np.array(high)

        if self.low.size != self.high.size:
            raise ValueError("size of 'low' and 'high' must match")

        self.action_size = self.low.size

    @property
    def discrete(self):
        return False

    def __call__(self, actions):
        return np.clip(actions, self.low, self.high)
