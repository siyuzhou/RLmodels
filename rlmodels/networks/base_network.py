import abc


class BaseNetwork(abc.ABC):
    def __init__(self, action_size):
        self.action_size = action_size

    @abc.abstractmethod
    def output(self, states):
        pass

    @abc.abstractmethod
    def loss(self, states, actions, rewards, next_states, dones, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def trainable_variables(self):
        pass

    @abc.abstractmethod
    def update(self, **kwargs):
        pass
