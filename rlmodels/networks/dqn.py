import tensorflow as tf
from tensorflow import keras
from .base_network import BaseNetwork


class DQN(BaseNetwork):
    def __init__(self, state_size, action_size, hidden_layers):
        if not hidden_layers:
            raise ValueError('hidden_layers cannot be empty')

        super().__init__(state_size, action_size)

        self.dqn = self._build_dqn(self, state_size, action_size, hidden_layers)

    def _build_dqn(self, state_size, action_size, hidden_layers):
        net = keras.models.Sequential()
        net.add(keras.layers.Dense(
            hidden_layers[0], input_shape=state_size, activation='relu'))
        for units in hidden_layers[1:]:
            net.add(keras.layers.Dense(units, activation='relu'))
        net.add(keras.layers.Dense(action_size))

        return net

    def loss(self, states, actions, rewards, next_states, dones, gamma=1):
        q_values = tf.batch_gather(self.dqn(states), actions)

        q_values_next = tf.reduce_max(self.dqn(next_states), axis=1, keepdims=True)
        expected_q = rewards + gamma * q_values_next * (1 - dones)
        tf.stop_gradient(expected_q)

        loss = keras.losses.mse(expected_q, q_values)

        return loss

    def variables(self):
        return self.dqn.trainable_variables()

    def update(self):
        pass
