import tensorflow as tf
from tensorflow import keras
from .modules import MLP


class ContinuousDeterministicPolicy(keras.layers.Layer):
    def __init__(self, action_size, units, bounds=None):
        super().__init__()

        if bounds is None:
            self.policy = MLP(action_size, units)
            self.multiplier = tf.ones(action_size)
            self.shift = tf.zeros(action_size)
        else:
            self.policy = MLP(action_size, units, 'tanh')
            bounds = tf.constant(bounds)
            lower, higher = bounds
            self.shift = tf.reduce_mean(bounds, axis=0)
            self.multiplier = (higher - lower) / 2

    def call(self, states):
        return self.policy(states) * self.multiplier + self.shift


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
