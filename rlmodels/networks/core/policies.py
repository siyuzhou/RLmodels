import tensorflow as tf
from tensorflow import keras
from .modules import MLP


class ContinuousDeterministicPolicy(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()

        self.policy = MLP(1, units)

    def call(self, states):
        return self.policy(states)


class DiscreteProbablisticPolicy(keras.layers.Layer):
    def __init__(self, action_size, units, logits=False):
        super().__init__()

        self.logits = logits
        activation = None
        if not self.logits:
            activation = 'softmax'
        self.policy = MLP(action_size, units, activation=activation)

    def call(self, states):
        return self.policy(states)
