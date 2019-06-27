import numpy as np
import tensorflow as tf
from tensorflow import keras
from .memory import ReplayBuffer

tf.enable_eager_execution()


class DQNAgent:
    EPSILON_MIN = 0.01
    EPSILON_MAX = 1
    EPSILON_DECAY = 0.995
    GAMMA = 0.95
    BUFFER_SIZE = int(2000)
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32

    def __init__(self, state_shape, hidden_layers, action_size, seed=None):
        if not hidden_layers:
            raise ValueError('hidden_layers cannot be empty')
        self.epsilon = self.EPSILON_MAX
        self.memory = ReplayBuffer(self.BUFFER_SIZE, seed)

        self.dqn = self._build_dqn(state_shape, hidden_layers, action_size)
        self.optimizer = tf.train.AdamOptimizer()

        self.action_size = action_size
        self.random = np.random.RandomState(seed)

    def _build_dqn(self, state_shape, hidden_layers, action_size):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(
            hidden_layers[0], input_shape=state_shape, activation='relu'))
        for units in hidden_layers[1:]:
            model.add(keras.layers.Dense(units, activation='relu'))
        model.add(keras.layers.Dense(action_size))

        return model

    def act(self, state):
        if self.random.rand() > self.epsilon:
            with tf.device('/cpu:0'):
                q_values = self.dqn.predict(np.expand_dims(state, 0))
            return np.argmax(q_values)

        return self.random.randint(self.action_size)

    def step(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

        if len(self.memory) > self.BATCH_SIZE:
            experiences = self._sample(self.BATCH_SIZE)
            self.learn(experiences)

        self.epsilon = max(self.epsilon * self.EPSILON_DECAY, self.EPSILON_MIN)

    def _sample(self, n):
        states, actions, rewards, next_states, dones = self.memory.sample(n)

        states = tf.constant(np.vstack(states), tf.float32)
        rewards = tf.constant(np.vstack(rewards), tf.float32)
        next_states = tf.constant(np.vstack(next_states), tf.float32)
        dones = tf.constant(np.vstack(dones), tf.float32)

        return states, actions, rewards, next_states, dones

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        q_values_next = tf.reduce_max(self.dqn(next_states), axis=1, keepdims=True)
        expected_q = rewards + self.GAMMA * q_values_next * (1 - dones)

        with tf.GradientTape() as tape:
            q_values = tf.batch_gather(self.dqn(states), np.vstack(actions))
            loss = keras.losses.mse(expected_q, q_values)

        grads = tape.gradient(loss, self.dqn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.dqn.trainable_variables))


class DDQNAgent(DQNAgent):
    def __init__(self, state_shape, hidden_layers, action_size, seed=None):
        super().__init__(state_shape, hidden_layers, action_size, seed)

        self.target_dqn = self._build_dqn(state_shape, hidden_layers, action_size)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        q_values_next = tf.reduce_max(self.target_dqn(next_states), axis=1, keepdims=True)
        expected_q = rewards + self.GAMMA * q_values_next * (1 - dones)

        with tf.GradientTape() as tape:
            q_values = tf.batch_gather(self.dqn(states), np.vstack(actions))
            loss = keras.losses.mse(expected_q, q_values)

        grads = tape.gradient(loss, self.dqn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.dqn.trainable_variables))

        self._update_target(0.1)

    def _update_target(self, alpha):
        new_weights = [(1 - alpha) * target_weights + alpha * local_weights
                       for target_weights, local_weights in zip(self.target_dqn.get_weights(), self.dqn.get_weights())]
        self.target_dqn.set_weights(new_weights)
