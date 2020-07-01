import abc


class BaseSampling(abc.ABC):
    @property
    @abc.abstractmethod
    def discrete(self) -> bool:
        return True

    @abc.abstractmethod
    def __call__(self):
        pass

    # def update(self):
    #     pass
