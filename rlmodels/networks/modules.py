import tensorflow as tf
from tensorflow import keras

# Works only for tensorflow >= 1.14


class MLP(keras.layers.Layer):
    def __init__(self, output_size, hidden_units, activation=None):
        super().__init__()

        self.hidden_layers = []
        for i, units in enumerate(hidden_units):
            name = f'hidden_{i}'
            layer = keras.layers.Dense(units, activation='relu', name=name)
            self.hidden_layers.append(name)
            setattr(self, name, layer)

        self.out_layer = keras.layers.Dense(output_size, activation=activation)

    def call(self, x):
        for name in self.hidden_layers:
            layer = getattr(self, name)
            x = layer(x)

        return self.out_layer(x)
