import tensorflow as tf
from tensorflow import keras
from .base_network import BaseNetwork
from .modules import MLP


class ActorCritic(keras.Model):
    """Off-policy version of Actor-Critic model."""

    def __init__(self, action_size, actor_units, critic_units, ratio=1.):
        if not actor_units:
            raise ValueError("'actor_units' cannot be empty")
        if not critic_units:
            raise ValueError("'critic_units' cannot be empty")

        super().__init__(action_size)

        self.actor = MLP(action_size, actor_units)
        self.critic = MLP(action_size, critic_units)
        self.ratio = ratio

    def output(self, states):
        return tf.math.softmax(self.actor(states))

    def trainable_variables(self):
        return self.actor.trainable_variables + self.critic.trainable_variables

    def loss(self, states, actions, rewards, next_states, dones):
        q_values = self.critic(states)

        q_a_actor = tf.batch_gather(q_values, actions)
        tf.stop_gradient(q_a_actor)
        log_pi_a = tf.batch_gather(tf.math.log_softmax(self.actor(states)), actions)

        actor_loss = -q_values * log_pi_a
