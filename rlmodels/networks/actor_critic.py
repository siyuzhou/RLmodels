import tensorflow as tf
from tensorflow import keras
from .base_network import BaseNetwork
from .modules import MLP


class ActorCritic(keras.Model):
    def __init__(self, action_size, actor_units, critic_units):
        if not actor_units:
            raise ValueError("'actor_units' cannot be empty")
        if not critic_units:
            raise ValueError("'critic_units' cannot be empty")

        super().__init__(action_size)

        self.actor = MLP(action_size, actor_units)
        self.critic = MLP(action_size, critic_units)

    def output(self, states):
        return self.actor(states)
