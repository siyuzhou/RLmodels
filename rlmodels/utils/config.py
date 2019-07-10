from .. import networks
from .. import memories


class Config:
    def __init__(self,
                 state_shape,
                 action_size,
                 network,
                 network_params,
                 memory='ReplayBuffer',
                 memory_capacity=int(1e5),
                 epsilon_min=0.01,
                 epsilon_max=1,
                 epsilon_decay=0.995,
                 gamma=0.95,
                 learning_rate=1e-3,
                 batch_size=32,
                 state_encoder=None,
                 state_encoder_params=None
                 ):
        pass
