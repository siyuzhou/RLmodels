import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlmodels.memories import ReplayBuffer
from rlmodels.kernel import DeepDeterministicPolicyGradient
from rlmodels.sampling import Clipping, OUNoise

from .base_agent import BaseAgent


class DDPGAgent(BaseAgent):
    def __init__(self, state_shape, action_size,
                 actor_units,
                 critic_units=None,
                 action_bounds=None,
                 encoder=None,
                 memory=None,
                 config=None,
                 seed=None):

        if not actor_units:
            raise ValueError("'actor_units' cannot be empty")
        if not critic_units:
            critic_units = actor_units

        if action_bounds is not None:
            action_bounds = np.array(action_bounds, dtype=np.float32)
            assert action_bounds.shape[0] == 2

        network = DeepDeterministicPolicyGradient(
            action_size, actor_units, critic_units, action_bounds)

        super().__init__(state_shape, action_size,
                         network=network,
                         encoder=encoder,
                         memory=memory,
                         config=config)

        self.sampling = Clipping(action_bounds)
        self.noise = OUNoise(action_size, seed)

    def act(self, state):
        with tf.device('/cpu:0'):
            state_tensor = tf.expand_dims(tf.constant(state, dtype=tf.float32), 0)
            action = self.network(self.encoder(state_tensor)).numpy().squeeze()
            action_noise = action + self.noise.sample()

        clipped_action = self.sampling(action)
        return clipped_action
