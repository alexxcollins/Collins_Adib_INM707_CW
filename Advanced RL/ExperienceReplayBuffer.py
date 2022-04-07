import numpy as np
import random
from collections import namedtuple
from collections import deque

Experience = namedtuple("Experience", ('state', 'action', 'reward', 'next_state', 'done'))


class ExperienceReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._buffer = deque(maxlen=self._buffer_size)

    # getter and setter
    @property
    def buffer_size(self):
        return self._buffer_size

    @property
    def batch_size(self):
        return self._batch_size

    def __len__(self):
        return len(self._buffer)

    def append(self, experience):
        self._buffer.append(experience)

    def sample(self):
        buffer_sample = random.sample(self._buffer, self._batch_size)
        return buffer_sample
