import tensorflow as tf
from tensorflow import keras
from .base_model import BaseModel
from .networks import DiscreteStochasticPolicy, VFunction, ContinuousDeterministicPolicy, QFunction


class ActorCritic(BaseModel):
    """Off-policy version of Actor-Critic model."""

    def __init__(self, action_size, actor_units, critic_units=None):
        if not actor_units:
            raise ValueError("'actor_units' cannot be empty")
        if not critic_units:
            critic_units = actor_units

        super().__init__(action_size)

        self.actor = DiscreteStochasticPolicy(action_size, actor_units, logits=True)
        self.critic = VFunction(critic_units)

    @property
    def discrete(self):
        return True

    def call(self, states):
        return tf.math.softmax(self.actor(states))

    def loss(self, states, actions, rewards, next_states, dones, gamma=0.99, ratio=1.):
        v_values = self.critic(states)

        # Loss for critic
        # Sample next action off-policy with reduce_max
        v_values_next = self.critic(next_states)

        v_values_target = rewards + gamma * tf.stop_gradient(v_values_next) * (1 - dones)

        critic_loss = keras.losses.mse(v_values, v_values_target)

        # Loss for actor
        td_error = tf.stop_gradient(v_values_target - v_values)

        log_pi_a = tf.gather(tf.math.log_softmax(self.actor(states)), actions, batch_dims=-1)

        actor_loss = -tf.squeeze(td_error * log_pi_a, -1)  # q_a_actor_values no gradient

        return actor_loss + ratio * critic_loss

    def update(self, params):
        pass


class DeepDeterministicPolicyGradient(BaseModel):
    """Off-policy version of Actor-Critic model."""

    def __init__(self, action_size, actor_units, critic_units=None, action_bounds=None):
        if not actor_units:
            raise ValueError("'actor_units' cannot be empty")
        if not critic_units:
            critic_units = actor_units

        super().__init__(action_size)

        self.actor = ContinuousDeterministicPolicy(action_size, actor_units, action_bounds)
        self.actor_target = ContinuousDeterministicPolicy(action_size, actor_units, action_bounds)

        self.critic = QFunction(critic_units)
        self.critic_target = QFunction(critic_units)

        self.actor_target.trainable = False
        self.critic_target.trainable = False

    @property
    def discrete(self):
        return False

    def call(self, states):
        return self.actor(states)

    def loss(self, states, actions, rewards, next_states, dones, gamma=0.99, ratio=1.):
        # Loss for critic
        actions_target_next = self.actor_target(next_states)
        q_values_target_next = self.critic_target(next_states, actions_target_next)
        q_values_target = rewards + gamma * tf.stop_gradient(q_values_target_next) * (1 - dones)

        q_values = self.critic(states, actions)
        critic_loss = keras.losses.mse(q_values, q_values_target)

        # Loss for actor
        actions_pred = self.actor(next_states)
        actor_loss = -tf.squeeze(self.critic(states, actions_pred), -1)

        return actor_loss + ratio * critic_loss

    def update(self, params):
        alpha = params.alpha

        new_actor_weights = [(1 - alpha) * target_weights + alpha * local_weights
                             for target_weights, local_weights in zip(self.actor_target.get_weights(), self.actor.get_weights())]
        self.actor_target.set_weights(new_actor_weights)

        new_critic_weights = [(1 - alpha) * target_weights + alpha * local_weights
                              for target_weights, local_weights in zip(self.critic_target.get_weights(), self.critic.get_weights())]
        self.critic_target.set_weights(new_critic_weights)


DDPG = DeepDeterministicPolicyGradient


class TwinDelayedDeepDeterministicPolicyGradient(BaseModel):
    """TD3"""

    def __init__(self, action_size, actor_units, critic_units=None, action_bounds=None):
        if not actor_units:
            raise ValueError("'actor_units' cannot be empty")
        if not critic_units:
            critic_units = actor_units

        super().__init__(action_size)

        self.actor = ContinuousDeterministicPolicy(action_size, actor_units, action_bounds)
        self.actor_target = ContinuousDeterministicPolicy(action_size, actor_units, action_bounds)

        self.critic1 = QFunction(critic_units)
        self.critic_target1 = QFunction(critic_units)
        self.critic2 = QFunction(critic_units)
        self.critic_target2 = QFunction(critic_units)

        self.actor_target.trainable = False
        self.critic_target1.trainable = False
        self.critic_target2.trainable = False

    @property
    def discrete(self):
        return False

    def call(self, states):
        return self.actor(states)

    def loss(self, states, actions, rewards, next_states, dones, gamma=0.99, ratio=1.):
        # Loss for critic
        actions_target_next = self.actor_target(next_states)
        q_values_target_next1 = self.critic1_target(next_states, actions_target_next)
        q_values_target_next2 = self.critic2_target(next_states, actions_target_next)

        q_values_target_next = tf.minimum(q_values_target_next1, q_values_target_next2)

        q_values_target = rewards + gamma * tf.stop_gradient(q_values_target_next) * (1 - dones)

        q_values1 = self.critic1(states, actions)
        critic1_loss = keras.losses.mse(q_values1, q_values_target)

        q_values2 = self.critic2(states, actions)
        critic2_loss = keras.losses.mse(q_values2, q_values_target)

        # Loss for actor
        actions_pred = self.actor(next_states)
        actor_loss = -tf.squeeze(self.critic(states, actions_pred), -1)

        return actor_loss + ratio * (critic1_loss + critic2_loss)

    def update(self, params):
        alpha = params.alpha

        new_actor_weights = [(1 - alpha) * target_weights + alpha * local_weights
                             for target_weights, local_weights in zip(self.actor_target.get_weights(), self.actor.get_weights())]
        self.actor_target.set_weights(new_actor_weights)

        new_critic1_weights = [(1 - alpha) * target_weights + alpha * local_weights
                               for target_weights, local_weights in zip(self.critic_target1.get_weights(), self.critic1.get_weights())]
        self.critic1_target.set_weights(new_critic1_weights)

        new_critic2_weights = [(1 - alpha) * target_weights + alpha * local_weights
                               for target_weights, local_weights in zip(self.critic_target2.get_weights(), self.critic2.get_weights())]
        self.critic2_target.set_weights(new_critic2_weights)


TD3 = TwinDelayedDeepDeterministicPolicyGradient
