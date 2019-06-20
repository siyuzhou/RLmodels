import numpy as np
import tensorflow as tf
from tensorflow import keras
from .memory import ReplayBuffer


class DQNAgent:
    EPSILON = 0.03
    GAMMA = 0.99
    BUFFER_SIZE = int(1e6)
    LEARNING_RATE = 1e-4

    def __init__(self, state_shape, hidden_layers, action_size, seed=None):
        if not hidden_layers:
            raise ValueError('hidden_layers cannot be empty')

        self.memory = ReplayBuffer(self.BUFFER_SIZE, seed)

        self.dqn = keras.models.Sequential()
        self._build_dqn(state_shape, hidden_layers, action_size)
        self.optimizer = tf.train.AdamOptimizer()

        self.action_size = action_size
        self.epsilon = epsilon
        self.random = np.random.RandomState(seed)

    def _build_dqn(self, state_shape, hidden_layers, action_size):
        self.dqn.add(keras.layers.Dense(
            hidden_layers[0], input_shape=state_shape, activation='relu'))
        for units in hidden_layers[1:]:
            self.dqn.add(keras.layers.Dense(units), activation='relu')

        self.dqn.add(keras.layers.Dense(action_size, activation='softmax'))

    def act(self, state):
        if self.random.rand() > self.EPSILON:
            with tf.device('/cpu:0'):
                q_values = self.dqn.predict(np.expand_dims(state, 0))
            return np.argmax(q_values)

        return self.random.randint(self.action_size)

    def step(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def _sample(self, n):
        states, actions, rewards, next_states, dones = self.memory.sample(n)

        states = tf.constant(states, tf.float32)
        rewards = tf.constant(np.vstack(rewards), tf.float32)
        next_states = tf.constant(states, tf.float32)
        dones = tf.constant(np.vstack(dones).tf.float32)

        return states, actions, rewards, next_states, dones

    def learn(self, batch_size):
        states, actions, rewards, next_states, dones = self._sample(batch_size)

        with tf.GradientTape as tape:
            q_values = tf.gather(self.dqn(states), actions)
            q_values_next = tf.reduce_max(self.dqn(next_states), axis=1, keepdims=True)

            expected_q = rewards + self.GAMMA * q_values_next * (1 - dones)

            loss = keras.losses.mse(expected_q, q_values)

        grads = tape.gradient(loss, self.dqn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.dqn.trainable_variables))
