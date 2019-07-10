import tensorflow as tf
from tensorflow import keras


class NoEncoder(keras.layers.Layer):
    def __init__(self, input_shape, params):
        super().__init__()
        self.layer = keras.layers.Lambda(lambda x: x, input_shape=input_shape)

    def call(self, x):
        return self.layer(x)


class MLPEncoder(keras.layers.Layer):
    def __init__(self, input_shape, params):
        if not params['hidden_units']:
            raise ValueError('hidden_units cannot be empty')
        hidden_units = params['hidden_units']

        super().__init__()

        name = f'dense_{0}'
        self.hidden_layers = [name]
        setattr(self, name, keras.layers.Dense(hidden_units[0], input_shape=input_shape, activation='relu'))
        for i, units in enumerate(hidden_units[1:]):
            name = f'dense_{i+1}'
            self.hidden_layers.append(name)
            setattr(self, name, keras.layers.Dense(units, activateion='relu'))

    def call(self, x):
        for name in self.hidden_layers:
            layer = getattr(self, name)
            x = layer(x)

        return x


class CNNEncoder(keras.layers.Layer):
    def __init__(self, input_shape, params):
        if not params['filters']:
            raise ValueError('filters cannot be empty')
        filters = params['filters']

        super().__init__()

        conv_name = f'conv_{0}'
        pool_name = f'pool_{0}'
        self.hidden_layers = [conv_name, pool_name]
        setattr(self, conv_name, keras.layers.Conv2D(filters[0], input_shape=input_shape, activation='relu'))
        setattr(self, pool_name, keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        for i, channels in enumerate(filters[1:]):
            conv_name = f'conv_{i+1}'
            pool_name = f'pool_{i+1}'
            self.hidden_layers.extend([conv_name, pool_name])
            setattr(self, conv_name, keras.layers.Conv2D(channels, activation='relu'))
            setattr(self, pool_name, keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        self.flatten = keras.layers.Flatten()

    def call(self, x):
        for name in self.hidden_layers:
            layer = getattr(self, name)
            x = layer(x)

        return self.flatten(x)
