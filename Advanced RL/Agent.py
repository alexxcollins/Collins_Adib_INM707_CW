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
                 epsilon_decay=[0.9999, 0.999],
                 memory_capacity=100000,
                 batch_size=1000,
                 update_frequency=20,
                 double_dqn=True
                 ):
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.double_dqn = double_dqn
        self.n_games = 0

        self.game = SnakeGameAI()
        self.replay_memory = deque(maxlen=memory_capacity)  # popleft()

        self.policy_net = DQN(11, 256, 3)
        self.target_net = DQN(11, 256, 3)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    # get observation and return np array of size 11
    def get_observation(self) -> np.array:
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

    # method to push to the memory
    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    # method to get random sample from the memory
    def get_memory_sample(self):
        if len(self.replay_memory) > self.batch_size:
            mini_sample = random.sample(self.replay_memory, self.batch_size)  # list of tuples
        else:
            mini_sample = self.replay_memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        return states, actions, rewards, next_states, dones

    def e_greedy_action(self, state) -> list:
        # random moves: tradeoff exploration / exploitation
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            self.policy_net.eval()
            with torch.no_grad():
                state0 = torch.tensor(state, dtype=torch.float)
                prediction = self.policy_net(state0)
            move = torch.argmax(prediction).item()
            action[move] = 1
            self.policy_net.train()

        return action

    def _synchronize_q_networks(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _soft_update_target_q_network_parameters(self) -> None:
        """Soft-update of target q-network parameters with the local q-network parameters."""
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.lr * local_param.data + (1 - self.lr) * target_param.data)

    def update_policy(self):
        if self.epsilon > 0.5:
            self.epsilon *= self.epsilon_decay[0]
        else:
            self.epsilon *= self.epsilon_decay[1]

        if self.epsilon < 0.05:
            self.epsilon = 0.05

    # method to save the model
    def save_model(self, file_name):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save({'state_dict': self.policy_net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, file_name)

    # method to load the weight
    def load_model(self, file_name):
        model_path = os.path.join('./model', file_name)
        if os.path.isfile(model_path):
            print("=> loading checkpoint... ")
            # self.model.load_state_dict(torch.load(model_path))
            checkpoint = torch.load(model_path)
            self.policy_net.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self._synchronize_q_networks()
            print("done !")
        else:
            print("no checkpoint found...")

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
        pred = self.policy_net(state)

        # target = pred.clone()
        if self.double_dqn:
            target = self.policy_net(state)
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    Q_new = reward[idx] + self.gamma * torch.max(self.target_net(next_state[idx]))

                target[idx][torch.argmax(action[idx]).item()] = Q_new

        else:
            target = self.target_net(state)
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    Q_new = reward[idx] + self.gamma * torch.max(self.target_net(next_state[idx]))

                target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.n_games % self.update_frequency == 0:
            self._synchronize_q_networks()