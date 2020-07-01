import abc
import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlmodels.networks.encoders import NoEncoder
from rlmodels.memories import ReplayBuffer


class BaseAgent(abc.ABC):
    def __init__(self, state_shape, action_size,
                 network,
                 encoder=None,
                 optimizer=None,
                 memory=None,
                 ):

        self.state_shape = state_shape
        self.action_size = action_size

        self.network = network
        self.encoder = encoder if encoder else NoEncoder(state_shape)
        self.optimizer = optimizer if optimizer else keras.optimizers.Adam()

        self.memory = memory if memory else ReplayBuffer()

    @abc.abstractmethod
    def act(self, state):
        pass

    def step(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

        if len(self.memory) > self.config.batch_size:
            experiences = self._sample(self.config.batch_size)
            self.learn(experiences)

    def _sample(self, n):
        states, actions, rewards, next_states, dones = self.memory.sample(n)

        states = tf.constant(np.vstack(states), tf.float32)
        actions = tf.constant(np.vstack(actions), tf.int32)
        rewards = tf.constant(np.vstack(rewards), tf.float32)
        next_states = tf.constant(np.vstack(next_states), tf.float32)
        dones = tf.constant(np.vstack(dones), tf.float32)

        return states, actions, rewards, next_states, dones

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        with tf.GradientTape() as tape:
            states = self.encoder(states)
            next_states = self.encoder(next_states)

            loss = self.network.loss(states, actions, rewards,
                                     next_states, dones, self.config.gamma)

        trainable_variables = self.encoder.trainable_variables + self.network.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))
