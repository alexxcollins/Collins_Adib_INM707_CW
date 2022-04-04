import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:
    def __init__(self, max_capacity):
        self.max_capacity = max_capacity
        self.memory = deque(maxlen=self.max_capacity)
        self.position = 0

    def push(self, state, action, next_state, reward, done):
        memory_cell = Transition(state, action, next_state, reward, done)
        self.memory.append(memory_cell)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN:

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h1 = F.relu(self.bn1(self.fc1(x.float())))
        h2 = F.relu(self.bn2(self.fc2(h1)))
        h3 = F.relu(self.bn3(self.fc3(h2)))
        output = self.fc4(h3.view(h3.size(0), -1))
        return output
