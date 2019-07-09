import abc


class BaseNetwork(abc.ABC):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    @abc.abstractmethod
    def output(self, state):
        pass

    @abc.abstractmethod
    def loss(self, states, actions, rewards, next_states, dones, **kwargs):
        pass

    @abc.abstractmethod
    def variables(self):
        pass

    @abc.abstractmethod
    def update(self):
        pass
