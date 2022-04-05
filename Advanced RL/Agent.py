import torch
import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', "action", 'reward', 'next_state', 'done'))


class Agent:
    def __init__(self,
                 gamma,
                 epsilon,
                 epsilon_decay,
                 epsilon_threshold,
                 memory_capacity,
                 model,
                 optimizer,
                 criterion,
                 ):
        """
        Constructor for the agent class
        :param epsilon (float): Hyperparameter for the DQN, between 0 and 1
        :param epsilon_decay (list): list contains 2 values between 0 and 1
        :param epsilon_threshold (float): threshold for the epsilon decay
        :param memory_capacity (int64): the max capacity for the replay memory
        :param model (Network): The Deep Q Network
        :param optimizer (object): Optimizer from PyTorch
        :param criterion (object): the Loss function from PyTorch
        """
        # this one is to calculate the average rewards
        self._n_games = 0
        self._gamma = gamma
        self._epsilon = epsilon
        self._memory_capacity = memory_capacity
        self._epsilon_threshold = epsilon_threshold
        self._epsilon_decay = epsilon_decay

        # Here we are using deque instead of normal list
        # first, deque is faster than normal list
        # Second, when deque reaches the max, it will start pop from the left and append from the right !
        self._replay_memory = deque(maxlen=self._memory_capacity)

        self.model = model  # our DQN
        self.optimizer = optimizer
        self.criterion = criterion

    # Getter and Setters
    @property
    def n_games(self):
        return self._n_games

    @n_games.setter
    def n_games(self, n_games):
        if n_games > 0:
            self._n_games = n_games

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        if 0 < gamma <= 1:
            self._gamma = gamma

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        if 0 < epsilon <= 1:
            self._epsilon = epsilon

    @property
    def epsilon_threshold(self):
        return self._epsilon_threshold

    @epsilon.setter
    def epsilon(self, epsilon_threshold):
        if 0 < epsilon_threshold < 1:
            self._epsilon_threshold = epsilon_threshold

    @property
    def epsilon_decay(self):
        return self._epsilon_decay

    @epsilon_decay.setter
    def epsilon_decay(self, epsilon_decay):
        if len(epsilon_decay) > 2 and 0 < epsilon_decay[0] < 1 and 0 < epsilon_decay[1] < 1:
            self._epsilon_decay = epsilon_decay

    @property
    def memory_capacity(self):
        return self._memory_capacity

    @memory_capacity.setter
    def memory_capacity(self, memory_capacity):
        if memory_capacity > 0:
            self._memory_capacity = memory_capacity

    @property
    def replay_memory(self):
        return self._replay_memory

    @replay_memory.setter
    def replay_memory(self, replay_memory):
        self._replay_memory = replay_memory

    # method to push to the memory of the agent
    def remember(self, state, action, reward, next_state, done):
        transition = Transition(state, action, reward, next_state, done)
        self.replay_memory.append(transition)

    def get_sample(self, batch_size=1000):
        if len(self._replay_memory) > batch_size:
            memory_sample = random.sample(self._replay_memory, batch_size)
        else:
            memory_sample = self.replay_memory

        states, actions, rewards, next_states, dones = zip(*memory_sample)
        return states, actions, rewards, next_states, dones

    # method to get a greedy action based on random number and epsilon
    def e_greedy_action(self, state):
        random_number = random.uniform(0, 1)
        greedy_action = random_number > self._epsilon
        action = [0, 0, 0]
        if greedy_action:
            index_of_action = random.randint(0, 2)
            action[index_of_action] = 1

        else:
            state = torch.tensor(state, dtype=torch.float)
            predicted_action = self.model(state)
            index_of_action = torch.argmax(predicted_action).item()
            action[index_of_action] = 1

        return action

    # method to train the agent
    def train(self, state, action, reward, next_state, done):
        # convert all inputs to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # get predicted actions
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
        if self.epsilon > self.epsilon_threshold:
            self.epsilon = self.epsilon * self._epsilon_decay[0]
        else:
            self.epsilon = self.epsilon * self._epsilon_decay[1]
