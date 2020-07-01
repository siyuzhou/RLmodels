import abc
import numpy as np


class BaseSamplingMethod(abc.ABC):
    @property
    @abc.abstractmethod
    def discrete(self) -> bool:
        return True

    @abc.abstractmethod
    def __call__(self):
        pass

    def update(self):
        pass


class UniformSampling(BaseSamplingMethod):
    def __init__(self, action_size, seed=None):
        self.action_size = action_size
        self.random = np.random.RandomState(seed)

    @property
    def discrete(self):
        return True

    def __call__(self):
        return self.random.randint(self.action_size)


class GreedySampling(BaseSamplingMethod):
    @property
    def discrete(self):
        return True

    def __call__(self, q_values):
        return np.argmax(q_values)


class EpsilonGreedySampling(BaseSamplingMethod):
    def __init__(self, epsilon_max, epsilon_min, epsilon_decay, seed=None):
        self._epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.random = np.random.RandomState(seed)

    @property
    def discrete(self):
        return True

    @property
    def epsilon(self):
        return self._epsilon

    def update(self):
        self._epsilon = max(self._epsilon * self.epsilon_decay, self.epsilon_min)

    def __call__(self, q_values):
        if self.random.rand() > self.epsilon:
            return np.argmax(q_values)

        return self.random.randint(q_values.shape[-1])
