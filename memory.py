import pickle
from collections import deque
import random
import os
from config import REPLAY_MEMORY_SIZE, MEMORY_PATH

class ReplayMemory:
    def __init__(self, capacity=REPLAY_MEMORY_SIZE):
        self.memory = deque(maxlen=capacity)

    def push(self, experience):
        """Saves a transition."""
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def save(self, path=MEMORY_PATH):
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)

    def load(self, path=MEMORY_PATH):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.memory = pickle.load(f)
