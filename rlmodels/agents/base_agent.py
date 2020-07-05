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
                 model,
                 encoder=None,
                 optimizer=None,
                 memory=None,
                 config=None,
                 seed=None
                 ):

        self.config = config if config is not None else Config()

        self.state_shape = state_shape
        self.action_size = action_size

        self.model = model
        self.encoder = encoder if encoder is not None else NoEncoder(state_shape)
        self.optimizer = optimizer if optimizer is not None else keras.optimizers.Adam()

        self.memory = memory if memory is not None else ReplayBuffer(self.config.memory_capacity)

        self.action_dtype = tf.int32 if self.model.discrete else tf.float32

    def act(self, state):
        with tf.device('/cpu:0'):
            state_tensor = tf.expand_dims(tf.constant(state, dtype=tf.float32), 0)
            outputs = self.model(self.encoder(state_tensor)).numpy().squeeze(0)

        return outputs

    def step(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done), None)

        if len(self.memory) > self.config.batch_size:
            experiences, info = self._sample(self.config.batch_size)
            losses = self.learn(experiences)

            self.memory.update(info, losses)

    def _sample(self, n):
        experiences, info = self.memory.sample(n)
        states, actions, rewards, next_states, dones = experiences

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

            loss = self.model.loss(states, actions, rewards,
                                   next_states, dones, self.config.gamma)
            # sum_loss = tf.reduce_sum(loss)

        trainable_variables = self.encoder.trainable_variables + self.model.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        self.model.update(self.config)

        return loss.numpy()  # Return loss as numpy array

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        checkpoint = os.path.join(path, 'checkpoint')
        self.model.save_weights(checkpoint)

    def load_model(self, path):
        checkpoint = os.path.join(path, 'checkpoint')
        if os.path.exists(checkpoint):
            self.model.load_weights(checkpoint)
        else:
            print(f'Checkpoint {checkpoint} not found.')
