import tensorflow as tf
from tensorflow import keras
from .modules import MLP


class VFunction(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()

        self.function = MLP(1, units)

    def call(self, states):
        return self.function(states)


class QFunction(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()

        self.function = MLP(1, units)

    def call(self, states, actions):
        inputs = tf.concat([states, actions], axis=-1)
        return self.function(inputs)


class QFunctionDiscrete(keras.layers.Layer):
    def __init__(self, action_size, units):
        super().__init__()

        self.function = MLP(action_size, units)

    def call(self, states):
        return self.function(states)


class QAdvantageFunction(keras.layers.Layer):
    def __init__(self, action_size, v_units, a_units):
        super().__init__()

        self.v_function = VFunction(v_units)
        self.a_function = QFunction(a_units)

    def call(self, states, actions):
        v_values = self.v_function(states)
        a_values = self.a_function(states, actions)
        return v_values + a_values - tf.reduce_mean(a_values, axis=-1, keepdims=True)


class QAdvantageFunctionDiscrete(keras.layers.Layer):
    def __init__(self, action_size, v_units, a_units):
        super().__init__()

        self.v_function = VFunction(v_units)
        self.a_function = QFunctionDiscrete(action_size, a_units)

    def call(self, states):
        v_values = self.v_function(states)
        a_values = self.a_function(states)
        return v_values + a_values - tf.reduce_mean(a_values, axis=-1, keepdims=True)
