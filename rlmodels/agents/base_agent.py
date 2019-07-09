import abc


class BaseAgent(abc.ABC):
    def __init__(self, state_shape, action_size, config):
        self.state_shape = state_shape
        self.action_size = action_size
        self.config = config

    @abc.abstractmethod
    def act(self, state):
        pass

    @abc.abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass

    @abc.abstractmethod
    def learn(self, states, actions, rewards, next_states, dones):
        pass
