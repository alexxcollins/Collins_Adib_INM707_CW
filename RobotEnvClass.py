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
                 epsilon_decays={'epsilon_threshold': 0.5, 'epsilon_decay1': 0.99999, 'epsilon_decay2': 0.9999},
                 max_steps=1000,
                 max_episodes=1000,
                 random_seed=42
                 ):
        self._dims = dims  # first number is height, second is width
        self._rewards = rewards
        self._start = start
        self._end = end
        self._tubes = tubes
        self._walls = walls
        self._epsilon_decays = epsilon_decays
        self._positions = positions
        self._max_steps = max_steps
        self._max_episodes = max_episodes

        # initialize the grid, R_matrix and Q_matrix
        self._initialize_grid()
        self._initialize_R_matrix()
        self._initialize_Q_matrix()

        self.rng = np.random.default_rng(random_seed)

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
        self._initialize_grid()
        self._initialize_R_matrix()
        self._initialize_Q_matrix()
        self.__initializeTunnels()

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

    # getter and setter for epsilon_decays
    @property
    def epsilon_decays(self):
        return self._epsilon_decays

    @epsilon_decays.setter
    def epsilon_decays(self, epsilon_decays):
        self._epsilon_decays = epsilon_decays

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

    # function to print a grid visualisation for the task   
    def visualise_world(self):
        W = self._dims[1] * 2 + 1
        H = self._dims[0] * 2 + 1
        # create empty grid
        rows = [['   '] * W for r in range(H)]

        # add dots for spaces where agent can move
        for i in range(W):
            for j in range(H):
                if i % 2 == 1 and j % 2 == 1:
                    rows[j][i] = '.  '

        # border round the grid
        rows[0] = ['X  ' for c in rows[0]]
        rows[H - 1] = ['X  ' for c in rows[H - 1]]
        for i in range(1, H - 1):
            rows[i][0] = 'X  '
            rows[i][W - 1] = 'X  '

        # insert walls
        for w in self._walls:
            rows[(w[0][0] + w[1][0]) + 1][w[0][1] + w[1][1] + 1] = 'X  '

        # insert tubes
        for n, t in enumerate(self._tubes):
            rows[t[0][0] * 2 + 1][t[0][1] * 2 + 1] = 'T{} '.format(n)
            rows[t[1][0] * 2 + 1][t[1][1] * 2 + 1] = 'T{} '.format(n)

        # insert start, end
        for label, p in [('S  ', self._start), ('E  ', self._end)]:
            rows[p[0] * 2 + 1][p[1] * 2 + 1] = label

        # insert features
        symbol = {'pond': 'P  ', 'cogs': 'G  ', 'croissant': 'C  '}
        for key in self._positions:
            for p in self._positions[key]:
                rows[p[0] * 2 + 1][p[1] * 2 + 1] = symbol[key]

        return self._create_viz_string(rows)

    # function to join strings in rows and add key
    def _create_viz_string(self, rows):
        s = '\n' + ''.join([''.join(r) + '\n' for r in rows])
        s += '\n'
        s += 'key:\n'
        s += 'S  = start location for agent\n'
        s += 'E  = end location for agent\n'
        s += '.  = empty cell\n'
        s += 'X  = boundary or wall beetween cells\n'
        s += 'Tn = nth Tube start or end. Agent can travel between the two Tn in one time step\n'
        s += 'P  = pond: falling in is cold and wet\n'
        s += 'G  = cog: agent is rewarded for collecting\n'
        s += 'C  = croissant: agent is rewarded for collecting\n'

        return s

    # initialize the rewards matrix
    def _initialize_R_matrix(self):
        d1 = self.dims[0]
        d2 = self.dims[1]
        self._R = np.empty((d1 * d2, d1 * d2))
        self._R.fill(np.nan)  # Fastest way to initilize R matrix

        # call some methods instead to write all the function here, cleaner and better for debuging
        self.__fillPossibleActions()
        self.__initializeTunnels()
        self.__initializeCogs()
        self.__initializePonds()
        self.__initializeCroissants()
        self.__initializeGoalPoint()
        self.__initializeWalls()

    # helper function, used in initialization methods
    def move_to(self, l, feature):
        """ creates a list of tuples with cell agent is moving to and cell agent is moving from"""
        cell = feature[0] * self._dims[1] + feature[1]

        if feature[0] > 0:  # cell not on top edge
            l.append((cell - self._dims[1], cell))
        if feature[0] < self._dims[0] - 1:  # cell not on bottom edge
            l.append((cell + self._dims[1], cell))
        if feature[1] > 0:  # cell not left hand edge
            l.append((cell - 1, cell))
        if feature[1] < self._dims[1] - 1:  # cell not on right hand edge
            l.append((cell + 1, cell))

        return l

    # function to fill all the possible moves
    def __fillPossibleActions(self):
        # All moves where reward is .self_rewards['r_time'] for action.
        ones = []
        for i in range(self._dims[0]):  # iterating over rows
            for j in range(self._dims[1]):  # iterating across columns
                cell = i * self._dims[1] + j
                if j != self._dims[1] - 1:
                    ones.append((cell + 1, cell))  # move right unless agent is on right edge
                if i != self._dims[0] - 1:
                    ones.append((cell + self._dims[1], cell))  # move up if not in top row
                if i != 0:
                    ones.append((cell - self._dims[1], cell))  # move down not in bottom row
                if j != 0:
                    ones.append((cell - 1, cell))  # move left if not on left edge
                ones.append((cell, cell))  # staying still is possible, why not?

        ones = tuple(zip(*ones))
        self._R[ones] = self._rewards['r_time']
        for i in range(self._dims[0]):
            for j in range(self._dims[1]):
                cell = i * self._dims[0] + j
                self._R[(cell, cell)] = self._rewards['r_time']
                #self._R[(cell, cell)] = np.nan
                #self._R[(cell, cell)] = 0

    # initialize the goal rewards
    def __initializeGoalPoint(self):
        end_cell = self._end[0] * self._dims[1] + self._end[1]
        ends = self.move_to([], self._end)
        ends.append([end_cell, end_cell])
        ends = tuple(zip(*ends))
        self._R[ends] = self._rewards['r_work']

    # initialize the Tunnels
    def __initializeTunnels(self):
        tubes_cells = []
        for tubes in self._tubes:
            tubes_cell = []
            for tube in tubes:
                cell_nb = tube[0] * self._dims[1] + tube[1]
                tubes_cell.append(cell_nb)
            # print(tubes_cell)
            tubes_cells.append(tuple(tubes_cell))
        for cell in tubes_cells.copy():
            # print(cell)
            tubes_cells.append((cell[0], cell[1]))

        tubes_cells = tuple(zip(*tubes_cells))
        self._R[tubes_cells] = self._rewards['r_time']

    # initialize the Cogs rewards
    def __initializeCogs(self):
        cogs = []
        for cog in self._positions['cogs']:
            cogs = self.move_to(cogs, cog)

        cogs = tuple(zip(*cogs))
        self._R[cogs] = self._rewards['r_cogs']

    # initialize the Ponds rewards
    def __initializePonds(self):
        # don't fall in the pond!
        # print(self._positions['pond'])
        ponds = []
        for pond in self._positions['pond']:
            p = pond[0] * self._dims[1] + pond[1]
            ponds = self.move_to(ponds, pond)
            ponds.extend([(p, p)])

        ponds = tuple(zip(*ponds))
        self._R[ponds] = self._rewards['r_pond']

    # initialize the Croissant rewards
    def __initializeCroissants(self):
        croissants = []
        for croissant in self._positions['croissant']:
            croissants = self.move_to(croissants, croissant)

        croissants = tuple(zip(*croissants))
        self._R[croissants] = self._rewards['r_croissant']

    # finally, construct the walls
    def __initializeWalls(self):
        for wall in self._walls:
            cell0 = wall[0][0] * self._dims[1] + wall[0][1]
            cell1 = wall[1][0] * self._dims[1] + wall[1][1]
            wall_in_matrix = ((cell0, cell1), (cell1, cell0))
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

    # method to get adjacent cells
    def _get_adjacent_cells(self, cell):
        cells = np.where([~np.isnan(self.R[cell,])])[1]
        return cells

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