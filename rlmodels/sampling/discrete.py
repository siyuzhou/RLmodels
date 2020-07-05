import abc
import numpy as np

from .base_sampling import BaseSampling


class UniformSampling(BaseSampling):
    def __init__(self, action_size, seed=None):
        self.action_size = action_size
        self.random = np.random.RandomState(seed)

    @property
    def discrete(self):
        return True

    def __call__(self):
        return self.random.randint(self.action_size)


class GreedySampling(BaseSampling):
    @property
    def discrete(self):
        return True

    def __call__(self, values):
        return np.argmax(values)


class EpsilonGreedySampling(BaseSampling):
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

    def __call__(self, values):
        self.update()
        if self.random.rand() > self.epsilon:
            return np.argmax(values)

        return self.random.randint(values.shape[-1])


class ProbabilitySampling(BaseSampling):
    def __init__(self, seed=None):
        self.random = np.random.RandomState(seed)

    @property
    def discrete(self):
        return True

    def _softmax(self, x, factor, epsilon):
        exps = np.exp(x * factor + epsilon)
        denominator = np.sum(exps, axis=-1, keepdims=True)
        return exps / denominator

    def _sample_index(self, ps):
        # ps is a matrix of probabilities. Each row adds up to 1.
        # Sample a index for each row according to the rows probabilities.
        # numpy.random.choice does not support batch sampling
        assert len(ps.shape) < 3

        # Reshape to accommodate the case where `ps` is 1-d
        ps = ps.reshape(-1, ps.shape[-1])
        n = len(ps)

        p_cumsum = np.cumsum(ps, axis=-1)
        hit = np.random.rand(n, 1)
        idxs = np.argmax(hit < p_cumsum, axis=-1)

        return idxs.squeeze()

    def __call__(self, values, logits=False, factor=1., epsilon=0.):
        if logits:
            # Convert logits to probs
            values = self._softmax(values, factor, epsilon)

        return self._sample_index(values)
