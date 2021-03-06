import tensorflow as tf
from tensorflow import keras


class MLP(keras.layers.Layer):
    def __init__(self, output_size, hidden_units, activation=None):
        super().__init__()

        self.hidden_layers = []
        for units in hidden_units:
            layer = keras.layers.Dense(units, activation='relu')
            self.hidden_layers.append(layer)

        self.out_layer = keras.layers.Dense(output_size, activation=activation)

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)

        return self.out_layer(x)
