import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import random

from RobotEnvClass import RobotEnv

class SARSA_learning(RobotEnv):

    def __init__(self,
                 dims=(6, 6),
                 rewards={'r_time': -1, 'r_pond': -15, 'r_croissant': 200, 'r_cogs': 200, 'r_work': 15},
                 start=(1, 0),
                 end=(5, 5),
                 positions={'pond': [(2, 4), (4, 3)], 'cogs': [(5, 2)], 'croissant': [(1, 4)]},
                 tubes=[[(0, 0), (3, 5)], [(1, 2), (4, 1)]],
                 walls=[[(0, 2), (0, 3)], [(1, 2), (1, 3)], [(2, 2), (2, 3)], [(1, 5), (2, 5)], [(3, 0), (4, 0)],
                        [(3, 1), (4, 1)], [(5, 2), (5, 3)]],
                 epsilon_decays={'epsilon_threshold': 0.5, 'epsilon_decay1': 0.99999, 'epsilon_decay2': 0.9999},
                 max_steps=1000,
                 max_episodes=1000,
                 random_seed=42
                 ):
        super().__init__(dims,
                         rewards,
                         start,
                         end,
                         positions,
                         tubes,
                         walls,
                         epsilon_decays,
                         max_steps,
                         max_episodes,
                         random_seed
                         )

    def run_episode(self, Q, alpha, gamma, epsilon):
        R_tot = 0
        # print(self._start)
        s = self._start[0] * self._dims[1] + self._start[1]
        goal_state = self._end[0] * self._dims[1] + self._end[1]
        R = self._R.copy()
        action_hist = [s]
        # some lists to keep track of visited cogs and croissant cells
        # to prevent the agent from re-visit them in the same episode to collect resources
        cogs_visited = []
        croissant_visited = []

        cogs_cells = [cog_position[0] * self._dims[1] + cog_position[1] for cog_position in self.positions['cogs']]

        croissant_cells = [croissant_position[0] * self._dims[1] + croissant_position[1] for croissant_position in
                           self.positions['croissant']]

        for i in range(self._max_steps):
            # actions selection
            available, best = self._get_actions(R, Q, s)
            a = self._get_greedy_action(epsilon, available, best)
            action_hist.append(a)

            s_old = s
            s = a
            _available, _best = self._get_actions(R, Q, s)
            _a = self._get_greedy_action(epsilon, _available, _best)
            Q[s_old, a] = Q[s_old, a] + alpha * (R[s_old, a] +
                                                 gamma * Q[s, _a] -
                                                 Q[s_old, a])

            # update total accumulated reward for this episode
            R_tot += R[s_old, a]
            # update total accumulated reward for this episode
            R_tot += R[s_old, a]
            if a in cogs_cells or a in croissant_cells:
                cells = self._get_adjacent_cells(a)
                for cell in cells:
                    R[cell, a] = self._rewards['r_time']
            if s == goal_state:
                break

        return Q, R_tot, action_hist

    '''
    function to run SARSA learning algorithm
    SARSA stands for State-Action-Reward-State-Action
    on-policy
    e-greedy policy
    '''

    def learn(self, alpha, gamma, epsilon):
        Q = self._Q.copy()
        Rtot = np.empty(shape=self.max_episodes)
        a_hist = np.empty(shape=(self.max_episodes), dtype=np.object_)
        Q_hist = np.empty(shape=(self.max_episodes, self._Q.shape[0], self._Q.shape[1]))
        for episode in range(self.max_episodes):
            Q, r, action_hist = self.run_episode(Q, alpha, gamma, epsilon)
            Rtot[episode] = r
            a_hist[episode] = action_hist
            Q_hist[episode, :, :] = Q

            if epsilon > self.epsilon_decays['epsilon_threshold']:
                epsilon *= self.epsilon_decays['epsilon_decay1']
            else:
                epsilon *= self.epsilon_decays['epsilon_decay2']

        return a_hist, Q_hist, Rtot
