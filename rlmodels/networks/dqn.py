import tensorflow as tf
from tensorflow import keras
from .base_network import BaseNetwork


class DQN(BaseNetwork):
    """Deep Q Network"""

    def __init__(self, action_size, hidden_units):
        if not hidden_units:
            raise ValueError("'hidden_units' cannot be empty")

        super().__init__(action_size)

        self.dqn = self._build_dqn(action_size, hidden_units)

    def _build_dqn(self, action_size, hidden_units):
        net = keras.models.Sequential()

        for units in hidden_units:
            net.add(keras.layers.Dense(units, activation='relu'))
        net.add(keras.layers.Dense(action_size))

        return net

    def output(self, states):
        return self.dqn(states)

    def loss(self, states, actions, rewards, next_states, dones, gamma=1):
        q_values = tf.batch_gather(self.dqn(states), actions)  # actions is a column vector.

        q_values_next = tf.reduce_max(self.dqn(next_states), axis=1, keepdims=True)
        expected_q = rewards + gamma * q_values_next * (1 - dones)
        tf.stop_gradient(expected_q)

        loss = keras.losses.mse(expected_q, q_values)

        return loss

    @property
    def trainable_variables(self):
        return self.dqn.trainable_variables

    def update(self):
        pass


class DDQN(DQN):
    """Deep Double Q Network"""

    def __init__(self, action_size, hidden_units):
        if not hidden_units:
            raise ValueError("'hidden_units' cannot be empty")

        super().__init__(action_size, hidden_units)

        self.target_dqn = self._build_dqn(action_size, hidden_units)

    def loss(self, states, actions, rewards, next_states, dones, gamma=1):
        q_values = tf.batch_gather(self.dqn(states), actions)

        q_values_next = tf.reduce_max(self.target_dqn(next_states), axis=1, keepdims=True)
        expected_q = rewards + gamma * q_values_next * (1 - dones)
        tf.stop_gradient(expected_q)

        loss = keras.losses.mse(expected_q, q_values)

        return loss

    def update(self, alpha):
        new_weights = [(1 - alpha) * target_weights + alpha * local_weights
                       for target_weights, local_weights in zip(self.target_dqn.get_weights(), self.dqn.get_weights())]
        self.target_dqn.set_weights(new_weights)
