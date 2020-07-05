import tensorflow as tf
from tensorflow import keras
from ..memories import ReplayBuffer

from rlmodels.memories import ReplayBuffer
from rlmodels.kernel import ActorCritic
from rlmodels.sampling import ProbabilitySampling, EpsilonGreedySampling

from .base_agent import BaseAgent
from rlmodels.utils import Config


class ActorCriticAgent(BaseAgent):
    def __init__(self, state_shape, action_size,
                 actor_units,
                 critic_units=None,
                 encoder=None,
                 memory=None,
                 config=None,
                 seed=None):

        if not actor_units:
            raise ValueError("'actor_units' cannot be empty")
        if not critic_units:
            critic_units = actor_units

        model = ActorCritic(action_size, actor_units, critic_units)

        super().__init__(state_shape, action_size,
                         model,
                         encoder=encoder,
                         memory=memory,
                         config=config,
                         seed=seed)

        self.sampling = EpsilonGreedySampling(
            self.config.epsilon_max, self.config.epsilon_min, self.config.epsilon_decay, seed)

    def act(self, state):
        action_probs = super().act(state)
        return self.sampling(action_probs)
