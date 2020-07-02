import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlmodels.memories import ReplayBuffer
from rlmodels.networks import DeepDeterministicPolicyGradient
from rlmodels.sampling import Clipping, OUNoise

from .base_agent import BaseAgent


class DDPGAgent(BaseAgent):
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

        network = DeepDeterministicPolicyGradient(action_size, actor_units, critic_units)

        super().__init__(state_shape, action_size,
                         network=network,
                         encoder=encoder,
                         memory=memory,
                         config=config)

        self.sampling = Clipping(config.action_low, config.action_high)
        self.noise = OUNoise(action_size, seed)

    def act(self, state):
        with tf.device('/cpu:0'):
            state_tensor = tf.expand_dims(tf.constant(state, dtype=tf.float32), 0)
            action = self.network(self.encoder(state_tensor))
            action = action.numpy().squeeze() + self.noise.sample()

        return self.sampling(action)
