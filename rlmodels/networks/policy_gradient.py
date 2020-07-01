import tensorflow as tf
from tensorflow import keras
from .base_network import BaseNetwork
from .core import DescreteProbablisticPolicy, QFunctionDescrete, ContinuousDeterministicPolicy, QFunction


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

    @property
    def trainable_variables(self):
        return self.actor.trainable_variables + self.critic.trainable_variables

    def loss(self, states, actions, rewards, next_states, dones, gamma=0.99, ratio=1.):
        q_a_values = tf.gather(self.critic(states), actions, batch_dims=-1)

        # Loss for actor
        q_a_values_no_grad = tf.stop_gradient(q_a_values)
        log_pi_a = tf.gather(tf.math.log_softmax(self.actor(states)), actions, batch_dims=-1)

        actor_loss = -q_a_values_no_grad * log_pi_a  # q_a_actor_values no gradient

        # Loss for critic
        # Sample next action off-policy with reduce_max
        q_a_values_next = tf.reduce_max(self.critic(next_states), axis=-1, keepdims=True)
        q_a_targets = rewards + gamma * tf.stop_gradient(q_a_values_next) * (1 - dones)

        critic_loss = keras.losses.mse(q_a_values, q_a_targets)

        return actor_loss + ratio * critic_loss


class DeepDeterministicPolicyGradient(BaseNetwork):
    """Off-policy version of Actor-Critic model."""

    def __init__(self, action_size, actor_units, critic_units=None):
        if not actor_units:
            raise ValueError("'actor_units' cannot be empty")
        if not critic_units:
            critic_units = actor_units

        super().__init__(action_size)

        self.actor = ContinuousDeterministicPolicy(actor_units)
        self.actor_target = ContinuousDeterministicPolicy(actor_units)

        self.critic = QFunction(critic_units)
        self.critic_target = QFunction(critic_units)

    def output(self, states):
        return self.actor(states)

    @property
    def trainable_variables(self):
        return self.actor.trainable_variables + self.critic.trainable_variables

    def loss(self, states, actions, rewards, next_states, dones, gamma=0.99, ratio=1.):
        # Loss for critic
        actions_target_next = self.actor_target(next_states)
        q_values_target_next = self.critic_target(next_states, actions_target_next)
        q_values_target = rewards + gamma * tf.stop_gradient(q_values_target_next) * (1 - dones)

        q_values = self.critic(states, actions)
        critic_loss = keras.losses.mse(q_values, q_values_target)

        # Loss for actor
        actions_pred = self.actor(next_states)
        actor_loss = -tf.reduce_mean(self.critic(states, actions_pred))

        return actor_loss + ratio * critic_loss
