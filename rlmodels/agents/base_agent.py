import os
import abc
import numpy as np
import tensorflow as tf
from tensorflow import keras

from rlmodels.kernel.encoders import NoEncoder
from rlmodels.memories import ReplayBuffer
from rlmodels.utils import Config


class BaseAgent(abc.ABC):
    def __init__(self, state_shape, action_size,
                 network,
                 encoder=None,
                 optimizer=None,
                 memory=None,
                 config=None
                 ):

        self.config = config
        if self.config is None:
            self.config = Config()

        self.state_shape = state_shape
        self.action_size = action_size

        self.network = network
        self.encoder = encoder if encoder else NoEncoder(state_shape)
        self.optimizer = optimizer if optimizer else keras.optimizers.Adam()

        self.memory = memory if memory else ReplayBuffer(self.config.memory_capacity)

        self.action_dtype = tf.int32 if self.network.discrete else tf.float32

    @abc.abstractmethod
    def act(self, state):
        pass

    def step(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

        if len(self.memory) > self.config.batch_size:
            experiences, info = self._sample(self.config.batch_size)
            losses = self.learn(experiences)

            self.memory.update(experiences, info, losses)

    def _sample(self, n):
        (states, actions, rewards, next_states, dones), info = self.memory.sample(n)

        states = tf.constant(np.vstack(states), tf.float32)
        actions = tf.constant(np.vstack(actions), self.action_dtype)
        rewards = tf.constant(np.vstack(rewards), tf.float32)
        next_states = tf.constant(np.vstack(next_states), tf.float32)
        dones = tf.constant(np.vstack(dones), tf.float32)

        return (states, actions, rewards, next_states, dones), info

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        with tf.GradientTape() as tape:
            states = self.encoder(states)
            next_states = self.encoder(next_states)

            losses = self.network.loss(states, actions, rewards,
                                       next_states, dones, self.config.gamma)
            sum_loss = tf.reduce_sum(losses)

        trainable_variables = self.encoder.trainable_variables + self.network.trainable_variables
        grads = tape.gradient(sum_loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        self.network.update(self.config)

        return losses.numpy()  # Return loss as numpy array

    def save_network(self, path):
        os.makedirs(path, exist_ok=True)
        checkpoint = os.path.join(path, 'checkpoint')
        self.network.save_weights(checkpoint)

    def load_network(self, path):
        checkpoint = os.path.join(path, 'checkpoint')
        if os.path.exists(checkpoint):
            self.network.load_weights(checkpoint)
        else:
            print(f'Checkpoint {checkpoint} not found.')
