class Config:
    def __init__(self,
                 memory_capacity=1,
                 epsilon_min=0.01,
                 epsilon_max=1,
                 epsilon_decay=0.995,
                 gamma=0.95,
                 alpha=0.01,
                 learning_rate=1e-3,
                 batch_size=1,
                 **kwargs
                 ):

        self.memory_capacity = memory_capacity
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_decay = epsilon_decay

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.batch_size = batch_size

        for key, value in kwargs.items():
            self.__setattr__(key, value)

        if self.batch_size > self.memory_capacity:
            raise ValueError("'memory_capacity' must be greater than 'batch_size'")
