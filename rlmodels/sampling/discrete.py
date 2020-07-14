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
        return self.random.randint(self.action_size), 1/self.action_size


class GreedySampling(BaseSampling):
    @property
    def discrete(self):
        return True

    def __call__(self, values):
        return np.argmax(values), 1.


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
        action_size = values.size
        max_a = np.argmax(values)

        ps = np.oness(action_size) * self.epsilon / action_size
        ps[max_a] += 1 - self.epsilon

        a = self.random.choice(action_size, p=ps)
        p = ps[a]

        self.update()
        return a, p


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

    def _batch_sample_index(self, ps):
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

    def _sample_index(self, ps):
        return self.random.choice(ps.size, p=ps)

    def __call__(self, values, logits=True, factor=1., shift=0.):
        if logits:
            # Convert logits to probs
            values = self._softmax(values, factor, shift)

        a = self._sample_index(values)
        p = values[a]

        return a, p


class EpsilonProbabilitySampling(ProbabilitySampling, EpsilonGreedySampling):
    def __init__(self, epsilon_max, epsilon_min, epsilon_decay, seed=None):
        EpsilonGreedySampling.__init__(self, epsilon_max, epsilon_min, epsilon_decay, seed)
        ProbabilitySampling.__init__(self, seed)

    def __call__(self, values, logits=True, factor=1., shift=0.):
        if logits:
            values = self._softmax(values, factor, shift)

        action_size = values.size
        ps = values * (1 - self.epsilon) + self.epsilon / action_size

        a = self._sample_index(ps)
        p = ps[a]

        self.update()
        return a, p
