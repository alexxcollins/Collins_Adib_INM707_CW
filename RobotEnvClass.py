# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 23:15:26 2022

@author: Khaliladib
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import random


# This is a class for creating the Robot environment

class RobotEnv(ABC):

    # Constructor for initialization, takes as input the parameters of the  env such as start, end, rewards, positions...
    def __init__(self,
                 dims=(6, 6),
                 rewards={'r_time': -1, 'r_pond': -15, 'r_croissant': 200, 'r_cogs': 200, 'r_work': 15},
                 start=(1, 0),
                 end=(5, 5),
                 positions={'pond': [(2, 4), (4, 3)], 'cogs': [(5, 2)], 'croissant': [(1, 4)]},
                 tubes=[[(0, 0), (3, 5)], [(1, 2), (4, 1)]],
                 walls=[[(0, 2), (0, 3)], [(1, 2), (1, 3)], [(2, 2), (2, 3)], [(1, 5), (2, 5)], [(3, 0), (4, 0)],
                        [(3, 1), (4, 1)], [(5, 2), (5, 3)]],
                 max_steps=1000,
                 max_episodes=1000,
                 ):
        self._dims = dims
        self._rewards = rewards
        self._start = start
        self._end = end
        self._tubes = tubes
        self._walls = walls
        self._positions = positions
        self._max_steps = max_steps
        self._max_episodes = max_episodes

        # initialize the grid, R_matrix and Q_matrix
        self._initialize_grid()
        self._initialize_R_matrix()
        self._initialize_Q_matrix()

        self.rng = np.random.default_rng(42)

    # getters and setters

    # getter and setter for dims
    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, dims):
        # When changing the dims we have to re-initialize: grid, R, Q
        self._dims = dims
        self._initialize_grid()
        self._initialize_R_matrix()
        self._initialize_Q_matrix()

    # getter and setter for start
    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, start):
        self._start = start

    # getter and setter for end
    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, end):
        self._end = end

    # getter and setter for tubes
    @property
    def tubes(self):
        return self._tubes

    @tubes.setter
    def tubes(self, tubes):
        self._tubes = tubes

    # getter and setter for max steps
    @property
    def max_steps(self):
        return self._max_steps

    @max_steps.setter
    def max_steps(self, max_steps):
        self._max_steps = max_steps

    # getter and setter for max episodes
    @property
    def max_episodes(self):
        return self._max_episodes

    @max_episodes.setter
    def max_episodes(self, max_episodes):
        self._max_episodes = max_episodes

    # getter and setter for max walls
    @property
    def walls(self):
        return self._walls

    @walls.setter
    def walls(self, walls):
        self._walls = walls

    # some property with only getter, in other words the user can't modify those properties
    @property
    def grid(self):
        return self._grid

    @property
    def R(self):
        return self._R

    @property
    def Q(self):
        return self._Q

    @property
    def rewards(self):
        return self._rewards

    @walls.setter
    def rewards(self, rewards):
        self._rewards = rewards
        self._initialize_grid()
        self._initialize_R_matrix()
        self._initialize_Q_matrix()

    @property
    def positions(self):
        return self._positions

    # function to initialize the grid, the propose of the grid is for visualization
    def _initialize_grid(self):
        self._grid = np.zeros(self.dims)
        for position in self._positions:
            for pos in self._positions[position]:
                self._grid[pos[0], pos[1]] = self._rewards['r_' + position]
                # print(position, ":", pos, ":", self._grid[pos[0], pos[1]])

        self._grid[self._end[0], self._end[1]] = self._rewards['r_work']
        # print(self._grid)

    # initialize the rewards matrix
    def _initialize_R_matrix(self):
        d1 = self.dims[0]
        d2 = self.dims[1]
        self._R = np.empty((d1 * d2, d1 * d2))
        self._R.fill(np.nan)  # Fastest way to initilize R matrix

        # call some methods instead to write all the function here, more cleaner and better for debuging
        self.__fillPossibleActions()
        self.__initializeTunnels()
        self.__initializeCrogs()
        self.__initializePonds()
        self.__initializeCroissants()
        self.__initializeGoalPoint()
        self.__initializeWalls()

    # helper function, used in initialization methods
    def move_to(self, l, cell):
        for i in [-self._dims[0], -1, 1, self._dims[0]]:
            if cell + i < self._dims[0] * self._dims[1]:
                l.append((cell + i, cell))
        return l

    # function to fill all the possible moves
    def __fillPossibleActions(self):
        # All moves where reward is -1 for action. Generate programmatically cos writing by hand is tedious
        ones = []
        for i in range(self._dims[0]):
            for j in range(self._dims[1]):
                cell = i * self._dims[0] + j
                if j != self._dims[0] - 1:
                    ones.append((cell, cell + 1))  # move right unless agent is on right edge
                if cell - self._dims[1] >= 0:
                    ones.append((cell, cell - 6))  # move up if not in top row
                if cell + self._dims[0] < self._dims[0] * self._dims[1]:
                    ones.append((cell, cell + 6))  # move down if cell not in bottom row
                if j != 0:
                    ones.append((cell, cell - 1))  # move left if not on left edge
                ones.append((cell, cell))  # staying still is possible, why not?

        ones = tuple(zip(*ones))
        self._R[ones] = self._rewards['r_time']

        # the purpose of this loop is to remove the option if staying in the same cell
        for i in range(self._dims[0]):
            for j in range(self._dims[1]):
                cell = i * self._dims[0] + j
                self._R[(cell, cell)] = np.nan

    # initialize the goal rewards
    def __initializeGoalPoint(self):
        end_cell = self._end[0] * self._dims[0] + self._end[1]
        ends = self.move_to([], end_cell)
        ends.append([end_cell, end_cell])
        ends = tuple(zip(*ends))
        self._R[ends] = self._rewards['r_work']

    # initialize the Tunnels
    def __initializeTunnels(self):
        tubes_cells = []
        for tubes in self._tubes:
            tubes_cell = []
            for tube in tubes:
                cell_nb = tube[0] * self._dims[0] + tube[1]
                tubes_cell.append(cell_nb)
            # print(tubes_cell)
            tubes_cells.append(tuple(tubes_cell))
        for cell in tubes_cells.copy():
            # print(cell)
            tubes_cells.append((cell[1], cell[0]))

        tubes_cells = tuple(zip(*tubes_cells))
        self._R[tubes_cells] = self._rewards['r_time']

    # initialize the Crogs rewards
    def __initializeCrogs(self):
        cogs = []
        for cog in self._positions['cogs']:
            cogs = self.move_to(cogs, cog[0] * self._dims[0] + cog[1])

        cogs = tuple(zip(*cogs))
        self._R[cogs] = self._rewards['r_cogs']

    # initialize the Ponds rewards
    def __initializePonds(self):
        # don't fall in the pond!
        # print(self._positions['pond'])
        ponds = []
        for pond in self._positions['pond']:
            p = pond[0] * self._dims[0] + pond[1]
            ponds = self.move_to(ponds, p)
            ponds.extend([(p, p)])

        # print(ponds)

        ponds = tuple(zip(*ponds))
        self._R[ponds] = self._rewards['r_pond']

    # initialize the Croissant rewards
    def __initializeCroissants(self):
        croissants = []
        for croissant in self._positions['croissant']:
            c = croissant[0] * self._dims[0] + croissant[1]
            croissants = self.move_to(croissants, c)

        # print(croissants)

        croissants = tuple(zip(*croissants))
        self._R[croissants] = self._rewards['r_croissant']

    # finally, construct the walls
    def __initializeWalls(self):
        for wall in self._walls:
            # print(wall)
            cell0 = wall[0][0] * self._dims[0] + wall[0][1]
            cell1 = wall[1][0] * self._dims[0] + wall[1][1]
            # print(cell0, ":", cell1)
            wall_in_matrix = (cell0, cell1)
            # print(wall_in_matrix)
            self._R[wall_in_matrix] = np.nan

    # display the matrix as pandas dataframe
    def display_matrix(self, matrix, start=None, end=None):
        pd.set_option("display.max_columns", None)
        display(pd.DataFrame(matrix).loc[start:end, start:end])

    # initialize the Q matrix with the same shape as R matrix
    def _initialize_Q_matrix(self):
        self._Q = np.zeros(self._R.shape)

    # function to run over only one episode, takes as input alpha, gamma and epsilon
    @abstractmethod
    def run_episode(self, Q, alpha, gamma, epsilon):
        pass

    @abstractmethod
    def learn(self, alpha, gamma, epsilon):
        pass

    def _get_actions(self, R, Q, s):
        """
        Returns best and all available actions as lists
        """
        available = np.where(~np.isnan(R[s]))[0]
        q_vals = [Q[s, a] for a in available]
        best = available[np.where(q_vals == np.max(q_vals))[0]]
        # change the type from np array to list
        available = available.tolist()
        best = best.tolist()
        return available, best

    def _get_greedy_action(self, epsilon, available, best):
        """
        Given epsilon, and available and best actions,
        Pick an appropriate action.
        """
        if self.rng.uniform() > epsilon:
            a = self.rng.choice(best)
        else:
            a = self.rng.choice(available)
        return a


class Q_Learning(RobotEnv):

    def __init__(self,
                 dims=(6, 6),
                 rewards={'r_time': -1, 'r_pond': -15, 'r_croissant': 200, 'r_cogs': 200, 'r_work': 15},
                 start=(1, 0),
                 end=(5, 5),
                 positions={'pond': [(2, 4), (4, 3)], 'cogs': [(5, 2)], 'croissant': [(1, 4)]},
                 tubes=[[(0, 0), (3, 5)], [(1, 2), (4, 1)]],
                 walls=[[(0, 2), (0, 3)], [(1, 2), (1, 3)], [(2, 2), (2, 3)], [(1, 5), (2, 5)], [(3, 0), (4, 0)],
                        [(3, 1), (4, 1)], [(5, 2), (5, 3)]],
                 max_steps=1000,
                 max_episodes=1000,
                 ):
        super().__init__(dims,
                         rewards,
                         start,
                         end,
                         positions,
                         tubes,
                         walls,
                         max_steps,
                         max_episodes,
                         )

    def run_episode(self, Q, alpha, gamma, epsilon):

        R_tot = 0
        # print(self._start)
        s = self._start[0] * self._dims[0] + self._start[1]
        goal_state = self._end[0] * self._dims[0] + self._end[1]
        # Q = self._Q
        R = self._R
        # print("Starting Point: ", s)
        # print("End Point: ", goal_state)

        # some lists to keep track of visited cogs and croissant cells
        # to prevent the agent from re-visit them in the same episode to collect resources
        cogs_visited = []
        croissant_visited = []

        cogs_cells = [cog_position[0] * self._dims[0] + cog_position[1] for cog_position in self.positions['cogs']]

        croissant_cells = [croissant_position[0] * self._dims[0] + croissant_position[1] for croissant_position in
                           self.positions['croissant']]

        for i in range(self._max_steps):
            # actions selection
            available, best = self._get_actions(R, Q, s)

            # update states:
            # loop to avoid re visit the same crogs and croissant
            move = False
            while not move:
                # chosse an action first
                a = self._get_greedy_action(epsilon, available, best)

                # if the next sell is cogs, and it is the first time we visit them append it to visited and move one
                if a in cogs_cells:
                    if a not in cogs_visited:
                        # print(cogs_visited, a)
                        cogs_visited.append(a)
                        move = True
                    '''
                    else:
                        available.remove(a)
                        best.remove(a)
                        continue
                    '''

                # same thing here
                if a in croissant_cells:
                    if a not in croissant_visited:
                        # print(croissant_visited, a)
                        croissant_visited.append(a)
                        move = True
                    '''
                    else:
                        available.remove(a)
                        best.remove(a)
                        continue
                    '''
                else:
                    move = True

            s_old = s
            s = a

            # update Q:

            Q[s_old, a] = Q[s_old, a] + alpha * (R[s_old, a] +
                                                 gamma * Q[s, :].max() -
                                                 Q[s_old, a])

            # update total accumulated reward for this episode
            R_tot += R[s_old, a]

            if s == goal_state:
                break

        return Q, R_tot

    # function to run Q learning algorithm
    # off-policy
    # greedy policy
    def learn(self, alpha, gamma, epsilon):
        Q = self._Q.copy()
        Rtot = np.array([])
        # Rtot = []
        for episode in range(self.max_episodes):
            Q, r = self.run_episode(Q, alpha, gamma, epsilon)
            # Rtot.append(r)
            Rtot = np.concatenate((Rtot, np.array([r])))

        if epsilon > 0.5:
            epsilon *= 0.99999
        else:
            epsilon *= 0.9999

        return Q, Rtot


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
                 max_steps=1000,
                 max_episodes=1000,
                 ):
        super().__init__(dims,
                         rewards,
                         start,
                         end,
                         positions,
                         tubes,
                         walls,
                         max_steps,
                         max_episodes,
                         )

    def run_episode(self, Q, alpha, gamma, epsilon):
        R_tot = 0
        # print(self._start)
        s = self._start[0] * self._dims[0] + self._start[1]
        goal_state = self._end[0] * self._dims[0] + self._end[1]
        R = self._R

        # some lists to keep track of visited cogs and croissant cells
        # to prevent the agent from re-visit them in the same episode to collect resources
        cogs_visited = []
        croissant_visited = []

        cogs_cells = [cog_position[0] * self._dims[0] + cog_position[1] for cog_position in self.positions['cogs']]

        croissant_cells = [croissant_position[0] * self._dims[0] + croissant_position[1] for croissant_position in
                           self.positions['croissant']]

        for i in range(self._max_steps):
            # actions selection
            available, best = self._get_actions(R, Q, s)

            # update states:
            # loop to avoid re visit the same crogs and croissant
            move = False
            while not move:
                # chosse an action first
                a = self._get_greedy_action(epsilon, available, best)

                # if the next sell is cogs, and it is the first time we visit them append it to visited and move one
                if a in cogs_cells:
                    if a not in cogs_visited:
                        # print(cogs_visited, a)
                        cogs_visited.append(a)
                        move = True
                    '''
                    else:
                        available.remove(a)
                        best.remove(a)
                        continue
                    '''

                # same thing here
                if a in croissant_cells:
                    if a not in croissant_visited:
                        # print(croissant_visited, a)
                        croissant_visited.append(a)
                        move = True
                    '''
                    else:
                        available.remove(a)
                        best.remove(a)
                        continue
                    '''
                else:
                    move = True

            s_old = s
            s = a
            _available, _best = self._get_actions(R, Q, s)
            _a = self._get_greedy_action(epsilon, _available, _best)
            Q[s_old, a] = Q[s_old, a] + alpha * (R[s_old, a] +
                                                 gamma * Q[s, _a] -
                                                 Q[s_old, a])

            # update total accumulated reward for this episode
            R_tot += R[s_old, a]
            if s == goal_state:
                break

        return Q, R_tot

    '''
    function to run SARSA learning algorithm
    SARSA stands for State-Action-Reward-State-Action
    on-policy
    e-greedy policy
    '''

    def learn(self, alpha, gamma, epsilon):
        Q = self._Q.copy()
        Rtot = np.array([])
        # Rtot = []
        for episode in range(self.max_episodes):
            Q, r = self.run_episode(Q, alpha, gamma, epsilon)
            # Rtot.append(r)
            Rtot = np.concatenate((Rtot, np.array([r])))

        if epsilon > 0.5:
            epsilon *= 0.99999
        else:
            epsilon *= 0.9999

        return Q, Rtot

    
class Q_Learning_Randomness(RobotEnv):

    def __init__(self,
                 dims=(6, 6),
                 rewards={'r_time': -1, 'r_pond': -15, 'r_croissant': 200, 'r_cogs': 200, 'r_work': 15},
                 start=(1, 0),
                 end=(5, 5),
                 positions={'pond': [(2, 4), (4, 3)], 'cogs': [(5, 2)], 'croissant': [(1, 4)]},
                 tubes=[[(0, 0), (3, 5)], [(1, 2), (4, 1)]],
                 walls=[[(0, 2), (0, 3)], [(1, 2), (1, 3)], [(2, 2), (2, 3)], [(1, 5), (2, 5)], [(3, 0), (4, 0)],
                        [(3, 1), (4, 1)], [(5, 2), (5, 3)]],
                 max_steps=1000,
                 max_episodes=1000,
                 ):
        super().__init__(dims,
                         rewards,
                         start,
                         end,
                         positions,
                         tubes,
                         walls,
                         max_steps,
                         max_episodes,
                         )

    def run_episode(self, Q, alpha, gamma, epsilon):

        R_tot = 0
        # print(self._start)
        s = self._start[0] * self._dims[0] + self._start[1]
        goal_state = self._end[0] * self._dims[0] + self._end[1]
        # Q = self._Q
        R = self._R
        # print("Starting Point: ", s)
        # print("End Point: ", goal_state)

        # some lists to keep track of visited cogs and croissant cells
        # to prevent the agent from re-visit them in the same episode to collect resources
        cogs_visited = []
        croissant_visited = []

        cogs_cells = [cog_position[0] * self._dims[0] + cog_position[1] for cog_position in self.positions['cogs']]

        croissant_cells = [croissant_position[0] * self._dims[0] + croissant_position[1] for croissant_position in
                           self.positions['croissant']]

        for i in range(self._max_steps):
            # actions selection
            available, best = self._get_actions(R, Q, s)

            # update states:
            # loop to avoid re visit the same crogs and croissant
            move = False
            while not move:
                # chosse an action first
                a = self._get_greedy_action(epsilon, available, best)

                # if the next sell is cogs, and it is the first time we visit them append it to visited and move one
                if a in cogs_cells:
                    if a not in cogs_visited:
                        # print(cogs_visited, a)
                        cogs_visited.append(a)
                        move = True
                    '''
                    else:
                        available.remove(a)
                        best.remove(a)
                        continue
                    '''

                # same thing here
                if a in croissant_cells:
                    if a not in croissant_visited:
                        # print(croissant_visited, a)
                        croissant_visited.append(a)
                        move = True
                    '''
                    else:
                        available.remove(a)
                        best.remove(a)
                        continue
                    '''
                else:
                    move = True

            s_old = s
            s = a

            # First get the available actions from rhe current state
            _available, _best = self._get_actions(R, Q, s)
            # calculate the probability distribution accroding to the Q-values
            sum_list = sum(_available)
            propa = [round((value/sum_list) * 100) for value in _available]
            # Choise the next state non-deterministically
            non_deterministic_action = random.choices(_available, weights=propa)
            # update Q:
            Q[s_old, a] = Q[s_old, a] + alpha * (R[s_old, a] +
                                                 gamma * non_deterministic_action[0] -
                                                 Q[s_old, a])

            # update total accumulated reward for this episode
            R_tot += R[s_old, a]

            if s == goal_state:
                break
            
        return Q, R_tot

    # function to run Q learning algorithm
    # off-policy
    # greedy policy
    def learn(self, alpha, gamma, epsilon):
        Q = self._Q.copy()
        Rtot = np.array([])
        # Rtot = []
        for episode in range(self.max_episodes):
            Q, r = self.run_episode(Q, alpha, gamma, epsilon)
            # Rtot.append(r)
            Rtot = np.concatenate((Rtot, np.array([r])))

        if epsilon > 0.5:
            epsilon *= 0.99999
        else:
            epsilon *= 0.9999

        return Q, Rtot