import abc


class BaseMemory(abc.ABC):
    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def add(self, experience, info):
        pass

    @abc.abstractmethod
    def sample(self, n):
        pass

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        pass
