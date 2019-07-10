import abc


class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def act(self, state):
        pass

    @abc.abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass

    @abc.abstractmethod
    def learn(self, experiences):
        pass


class Agent(BaseAgent):
    def __init__(self, config):
        super().__init__()
        self.state_shape = config.state_shape
        self.action_size = config.action_size
