import random
from collections import deque

from .base_memory import BaseMemory


class ReplayBuffer(BaseMemory):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity=1000, seed=None):
        super().__init__()
        self.memory = deque(maxlen=capacity)
        self.random = random.Random(seed)

    def __len__(self):
        return len(self.memory)

    def add(self, experience):
        """Add a new experience to memory."""
        state, action, reward, next_state, done = experience
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, n):
        """Randomly sample a batch of experiences from memory."""
        if len(self.memory) < n:
            experiences = self.memory
        else:
            experiences = self.random.sample(self.memory, k=n)

        states, actions, rewards, next_states, dones = zip(*experiences)

        return (states, actions, rewards, next_states, dones), None

    def update(self, *args, **kwargs):
        pass
