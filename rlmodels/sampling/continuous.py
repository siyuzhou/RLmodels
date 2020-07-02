import abc
import numpy as np

from .base_sampling import BaseSampling


class Clipping(BaseSampling):
    def __init__(self, bounds=None):
        if bounds is not None:
            self.low, self.high = np.array(bounds)
            if self.low.size != self.high.size:
                raise ValueError("size of 'low' and 'high' must match")
        else:
            self.low, self.high = -np.inf, np.inf

        self.action_size = self.low.size

    @property
    def discrete(self):
        return False

    def __call__(self, actions):
        return np.clip(actions, self.low, self.high)
