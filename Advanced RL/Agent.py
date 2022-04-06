import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
from collections import deque

from game import SnakeGameAI, Direction, Point
from model import DQN


class Agent:

    def __init__(self,
                 learning_rate=0.001,
                 gamma=0.9,
                 epsilon=0.9,
                 memory_capacity=100000,
                 ):
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_games = 0

        self.game = SnakeGameAI()
        self.replay_memory = deque(maxlen=memory_capacity)  # popleft()

        self.model = DQN(11, 256, 3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def get_observation(self):
        head = self.game.snake_body[0]
        block_size = self.game.block_size

        point_l = Point(head.x - block_size, head.y)
        point_r = Point(head.x + block_size, head.y)
        point_u = Point(head.x, head.y - block_size)
        point_d = Point(head.x, head.y + block_size)

        dir_l = self.game.direction == Direction.LEFT
        dir_r = self.game.direction == Direction.RIGHT
        dir_u = self.game.direction == Direction.UP
        dir_d = self.game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.game.is_collision(point_r)) or
            (dir_l and self.game.is_collision(point_l)) or
            (dir_u and self.game.is_collision(point_u)) or
            (dir_d and self.game.is_collision(point_d)),

            # Danger right
            (dir_u and self.game.is_collision(point_r)) or
            (dir_d and self.game.is_collision(point_l)) or
            (dir_l and self.game.is_collision(point_u)) or
            (dir_r and self.game.is_collision(point_d)),

            # Danger left
            (dir_d and self.game.is_collision(point_r)) or
            (dir_u and self.game.is_collision(point_l)) or
            (dir_r and self.game.is_collision(point_u)) or
            (dir_l and self.game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.game.rat.x < self.game.snake_head.x,  # food left
            self.game.rat.x > self.game.snake_head.x,  # food right
            self.game.rat.y < self.game.snake_head.y,  # food up
            self.game.rat.y > self.game.snake_head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def get_memory_sample(self, batch_size=1000):
        if len(self.replay_memory) > batch_size:
            mini_sample = random.sample(self.replay_memory, batch_size)  # list of tuples
        else:
            mini_sample = self.replay_memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        return states, actions, rewards, next_states, dones

    def get_e_action(self, state):
        # random moves: tradeoff exploration / exploitation
        action = [0, 0, 0]
        if random.randint(0, 1) < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            action[move] = 1

        return action

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

    def update_policy(self):
        if self.epsilon > 0.5:
            self.epsilon *= 0.9999
        else:
            self.epsilon *= 0.999

    # method to save the model
    def save_model(self, file_name):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, file_name)

    def load_model(self, file_name):
        model_path = os.path.join('./model', file_name)
        if os.path.isfile(model_path):
            print("=> loading checkpoint... ")
            # self.model.load_state_dict(torch.load(model_path))
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
