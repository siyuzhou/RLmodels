import abc
from tensorflow import keras


class BaseNetwork(abc.ABC, keras.Model):
    def __init__(self, action_size):
        super().__init__()
        self.action_size = action_size

    @property
    @abc.abstractmethod
    def discrete(self) -> bool:
        pass

    @abc.abstractmethod
    def loss(self, states, actions, rewards, next_states, dones, **kwargs):
        pass

    @abc.abstractmethod
    def update(self, params):
        pass
