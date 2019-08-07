import tensorflow as tf
from tensorflow import keras
from .base_network import BaseNetwork
from .modules import MLP


class DQN(BaseNetwork):
    """Deep Q Network"""

    def __init__(self, action_size, hidden_units):
        if not hidden_units:
            raise ValueError("'hidden_units' cannot be empty")

        super().__init__(action_size)

        self.dqn = MLP(action_size, hidden_units)

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


class DoubleDQN(BaseNetwork):
    def __init__(self, action_size, hidden_units):
        if not hidden_units:
            raise ValueError("'hidden_units' cannot be empty")

        super().__init__(action_size)

        self.dqn = MLP(action_size, hidden_units)
        self.target_dqn = MLP(action_size, hidden_units)

    def output(self, states):
        return self.dqn(states)

    def loss(self, states, actions, rewards, next_states, dones, gamma=1):
        q_values = tf.batch_gather(self.dqn(states), actions)

        q_values_next = tf.reduce_max(self.target_dqn(next_states), axis=1, keepdims=True)
        expected_q = rewards + gamma * q_values_next * (1 - dones)
        tf.stop_gradient(expected_q)

        loss = keras.losses.mse(expected_q, q_values)

        return loss

    @property
    def trainable_variables(self):
        return self.dqn.trainable_variables

    def update(self, alpha):
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

        self.v_net = MLP(1, v_units)
        self.a_net = MLP(action_size, a_units)

        self.v_net_target = MLP(1, v_units)
        self.a_net_target = MLP(action_size, a_units)

    def _q_values(self, states):
        adv = self.a_net(states)
        return self.v_net(states) + adv - tf.reduce_mean(adv, axis=-1, keepdims=True)

    def _q_values_target(self, states):
        adv = self.a_net_target(states)
        return self.v_net_target(states) + adv - tf.reduce_mean(adv, axis=-1, keepdims=True)

    def output(self, states):
        return self._q_values(states)

    def loss(self, states, actions, rewards, next_states, dones, gamma=1):
        q_values = tf.batch_gather(self._q_values(states), actions)
        q_values_next = tf.reduce_max(self._q_values_target(states), axis=-1, keepdims=True)

        expected_q = rewards + gamma * q_values_next * (1 - dones)
        tf.stop_gradient(expected_q)

        loss = keras.losses.mse(expected_q, q_values)

        return loss

    @property
    def trainable_variables(self):
        return self.v_net.trainable_variables + self.a_net.trainable_variables

    def update(self, alpha):
        new_v_weights = [(1 - alpha) * target_weights + alpha * local_weights
                         for target_weights, local_weights in zip(self.v_net_target.get_weights(), self.v_net.get_weights())]
        new_a_weights = [(1 - alpha) * target_weights + alpha * local_weights
                         for target_weights, local_weights in zip(self.a_net_target.get_weights(), self.a_net.get_weights())]

        self.v_net_target.set_weights(new_v_weights)
        self.a_net_target.set_weights(new_a_weights)
