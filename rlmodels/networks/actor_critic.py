import tensorflow as tf
from tensorflow import keras
from .base_network import BaseNetwork
from .core import DescreteProbablisticPolicy, QFunctionDescrete


class ActorCritic(BaseNetwork):
    """Off-policy version of Actor-Critic model."""

    def __init__(self, action_size, actor_units, critic_units=None):
        if not actor_units:
            raise ValueError("'actor_units' cannot be empty")
        if not critic_units:
            critic_units = actor_units

        super().__init__(action_size)

        self.actor = DescreteProbablisticPolicy(action_size, actor_units, logits=True)
        self.critic = QFunctionDescrete(action_size, critic_units)

    def output(self, states):
        return tf.math.softmax(self.actor(states))

    def trainable_variables(self):
        return self.actor.trainable_variables + self.critic.trainable_variables

    def loss(self, states, actions, rewards, next_states, dones, gamma=0.99, ratio=1.):
        q_a_values = tf.gather(self.critic(states), actions, batch_dims=-1)

        # Loss for actor
        q_a_values_no_grad = tf.stop_gradient(q_a_values)
        log_pi_a = tf.gather(tf.math.log_softmax(self.actor(states)), actions, batch_dims=-1)

        actor_loss = -q_a_values_no_grad * log_pi_a  # q_a_actor_values no gradient

        # Loss for critic
        q_a_values_next = tf.reduce_max(self.critic(next_states), axis=-1, keepdims=True)
        expected_q_a = rewards + gamma * tf.stop_gradient(q_a_values_next) * (1 - dones)

        critic_loss = keras.losses.mse(expected_q_a, q_a_values)

        return actor_loss + ratio * critic_loss
