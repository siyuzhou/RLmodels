import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlmodels.memories import ReplayBuffer
from rlmodels.kernel import DQN, DoubleDQN, DuelingDQN
from rlmodels.sampling import EpsilonGreedySampling

from .base_agent import BaseAgent
from rlmodels.utils import Config


class DQNAgent(BaseAgent):
    def __init__(self, state_shape, action_size,
                 hidden_units,
                 encoder=None,
                 memory=None,
                 config=None,
                 seed=None):

        if not hidden_units:
            raise ValueError('hidden_layers cannot be empty')

        super().__init__(state_shape, action_size,
                         model=DQN(action_size, hidden_units),
                         encoder=encoder,
                         memory=memory,
                         config=config)

        self.sampling = EpsilonGreedySampling(
            self.config.epsilon_max, self.config.epsilon_min, self.config.epsilon_decay, seed)

    def act(self, state):
        q_values = super().act(state)
        return self.sampling(q_values)


class DoubleDQNAgent(DQNAgent):
    def __init__(self, state_shape, action_size,
                 hidden_units,
                 encoder=None,
                 memory=None,
                 config=None,
                 seed=None):

        if not hidden_units:
            raise ValueError('hidden_layers cannot be empty')

        BaseAgent.__init__(self, state_shape, action_size,
                           model=DoubleDQN(action_size, hidden_units),
                           encoder=encoder,
                           memory=memory,
                           config=config)

        self.config = config
        if self.config is None:
            self.config = Config()

        self.sampling = EpsilonGreedySampling(
            self.config.epsilon_max, self.config.epsilon_min, self.config.epsilon_decay, seed)


class DuelingDQNAgent(DQNAgent):
    def __init__(self, state_shape, action_size,
                 v_units,
                 a_units=None,
                 encoder=None,
                 memory=None,
                 config=None,
                 seed=None):

        if not v_units:
            raise ValueError('v_units cannot be empty')
        if not a_units:
            a_units = v_units

        BaseAgent.__init__(self, state_shape, action_size,
                           model=DuelingDQN(action_size, v_units, a_units),
                           encoder=encoder,
                           memory=memory)

        self.config = config
        if self.config is None:
            self.config = Config()

        self.sampling = EpsilonGreedySampling(
            self.config.epsilon_max, self.config.epsilon_min, self.config.epsilon_decay, seed)
