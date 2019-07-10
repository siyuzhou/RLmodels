import abc


class BaseAgent(abc.ABC):
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size

    @abc.abstractmethod
    def act(self, state):
        pass

    @abc.abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass

    @abc.abstractmethod
    def learn(self, experiences):
        pass
