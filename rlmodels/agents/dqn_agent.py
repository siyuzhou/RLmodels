import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..memories import ReplayBuffer
from ..networks import DQN, DDQN, NoEncoder
from .base_agent import BaseAgent
from ..utils import Config


class DQNAgent(BaseAgent):
    def __init__(self, state_shape, action_size,
                 hidden_units,
                 state_encoder=NoEncoder,
                 state_encoder_params=None,
                 memory=ReplayBuffer,
                 config=None,
                 seed=None):

        if not hidden_units:
            raise ValueError('hidden_layers cannot be empty')

        super().__init__(state_shape, action_size)

        self.config = config
        if self.config is None:
            self.config = Config()

        self.epsilon = self.config.epsilon_max
        self.memory = memory(self.config.memory_capacity, seed)

        if state_encoder_params is None:
            state_encoder_params = {}
        self.state_encoder = state_encoder(state_shape, **state_encoder_params)

        self.network = DQN(action_size, hidden_units)
        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)

        self.random = np.random.RandomState(seed)

    def act(self, state):
        if self.random.rand() > self.epsilon:
            with tf.device('/cpu:0'):
                state_tensor = tf.expand_dims(tf.constant(state, dtype=tf.float32), 0)
                q_values = self.network.output(self.state_encoder(state_tensor))
                q_values = keras.backend.eval(q_values)

            return np.argmax(q_values)

        return self.random.randint(self.action_size)

    def step(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

        if len(self.memory) > self.config.batch_size:
            experiences = self._sample(self.config.batch_size)
            self.learn(experiences)

        self.epsilon = max(
            self.epsilon * self.config.epsilon_decay, self.config.epsilon_min)

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
            states = self.state_encoder(states)
            next_states = self.state_encoder(next_states)

            loss = self.network.loss(states, actions, rewards,
                                     next_states, dones, self.config.gamma)

        trainable_variables = self.state_encoder.trainable_variables + self.network.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))


class DDQNAgent(DQNAgent):
    def __init__(self, state_shape, action_size,
                 hidden_units,
                 state_encoder=NoEncoder,
                 state_encoder_params=None,
                 memory=ReplayBuffer,
                 config=None,
                 seed=None):

        if not hidden_units:
            raise ValueError('hidden_layers cannot be empty')

        BaseAgent.__init__(self, state_shape, action_size)

        self.config = config
        if self.config is None:
            self.config = Config()

        self.epsilon = self.config.epsilon_max
        self.memory = memory(self.config.memory_capacity, seed)

        self.state_encoder = state_encoder(state_shape, state_encoder_params)
        self.network = DDQN(action_size, hidden_units)
        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)

        self.random = np.random.RandomState(seed)

    def learn(self, experiences):
        super().learn(experiences)

        self.network.update(self.config.alpha)
