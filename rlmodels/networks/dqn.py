import tensorflow as tf
from tensorflow import keras
from .base_network import BaseNetwork
from .core import QFunctionDiscrete, QAdvantageFunctionDiscrete


class DQN(BaseNetwork):
    """Deep Q Network"""

    def __init__(self, action_size, units):
        if not units:
            raise ValueError("'units' cannot be empty")

        super().__init__(action_size)

        self.dqn = QFunctionDiscrete(action_size, units)

    @property
    def discrete(self):
        return True

    def call(self, states):
        return self.dqn(states)

    def loss(self, states, actions, rewards, next_states, dones, gamma=1):
        # actions is a column vector.
        q_values = tf.gather(self.dqn(states), actions, batch_dims=-1)

        q_values_next = tf.reduce_max(self.dqn(next_states), axis=1, keepdims=True)
        expected_q = rewards + gamma * q_values_next * (1 - dones)
        expected_q = tf.stop_gradient(expected_q)

        loss = keras.losses.mse(expected_q, q_values)

        return loss

    def update(self, params):
        pass


class DoubleDQN(BaseNetwork):
    def __init__(self, action_size, units):
        if not units:
            raise ValueError("'units' cannot be empty")

        super().__init__(action_size)

        self.dqn = QFunctionDiscrete(action_size, units)
        self.target_dqn = QFunctionDiscrete(action_size, units)

        self.target_dqn.trainable = False

    @property
    def discrete(self):
        return True

    def call(self, states):
        return self.dqn(states)

    def loss(self, states, actions, rewards, next_states, dones, gamma=0.99):
        q_values = tf.gather(self.dqn(states), actions, batch_dims=-1)
        q_values_next = tf.reduce_max(self.target_dqn(next_states), axis=1, keepdims=True)

        expected_q = rewards + gamma * q_values_next * (1 - dones)
        expected_q = tf.stop_gradient(expected_q)

        loss = keras.losses.mse(expected_q, q_values)

        return loss

    def update(self, params):
        alpha = params.alpha

        new_weights = [(1 - alpha) * target_weights + alpha * local_weights
                       for target_weights, local_weights in zip(self.target_dqn.get_weights(), self.dqn.get_weights())]
        self.target_dqn.set_weights(new_weights)


class DuelingDQN(BaseNetwork):
    def __init__(self, action_size, v_units, a_units):
        if not v_units:
            raise ValueError("'v_units' cannot be empty")
        if not a_units:
            raise ValueError("'a_units' cannot be empty")

        super().__init__(action_size)

        self.dqn = QAdvantageFunctionDiscrete(action_size, v_units, a_units)
        self.target_dqn = QAdvantageFunctionDiscrete(action_size, v_units, a_units)

        self.target_dqn.trainable = False

    @property
    def discrete(self):
        return True

    def call(self, states):
        return self.dqn(states)

    def loss(self, states, actions, rewards, next_states, dones, gamma=0.99):
        q_values = tf.gather(self.dqn(states), actions, batch_dims=-1)
        q_values_next = tf.reduce_max(self.target_dqn(states), axis=-1, keepdims=True)

        expected_q = rewards + gamma * q_values_next * (1 - dones)
        expected_q = tf.stop_gradient(expected_q)

        loss = keras.losses.mse(expected_q, q_values)

        return loss

    def update(self, params):
        alpha = params.alpha

        new_weights = [(1 - alpha) * target_weights + alpha * local_weights
                       for target_weights, local_weights in zip(self.target_dqn.get_weights(), self.dqn.get_weights())]
        self.target_dqn.set_weights(new_weights)
