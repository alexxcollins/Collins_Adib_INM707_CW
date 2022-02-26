from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class WorkJourney(ABC):
    """Class to run the grid Q-learning.
    Better code would be to write base class and then child class to run specific grid configurations
    """
    
    def __init__(self, reward_dict, max_steps, max_episodes,
                 start_state, goal_state):
        """
        reward_dict: **kwargs dice for the create reward matrix function.
                     again - is there a better way to code this than within init here?
        max_steps: maximum number of steps to run in one episode. 
                   Should be big enough never to be hit.
        max_episodes: maximum number of episodes to run for each
                      Q-learning cycle.
        """
        self.RM = create_robot_work_RM(**reward_dict)
        # need to change below. Could be a function for random start state.
        
        self.start = start_state 
        self.goal = goal_state # should be beter way to code this
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        
    def display_matrix(self, M, start_idx=None, end_idx=None):
        """Display a formatted pandas DataFrame.
        
        Intended use if for Q or R matrix.
        
        start_idx and end_idx are options inputs to restrict slice
        the dataframe output to df.loc[start_idx:end_idx, start_idx:end_idx]
        
        """
        pd.set_option("display.max_columns", None)
        display(pd.DataFrame(Ri).loc[start_idx:end_idx, start_idx:end_idx])
    

def create_robot_work_RM(r_time = -1, r_pond = -15,
                  r_croissant = 200, r_cogs = 200,
                  r_work = 15):
    """create the reward matrix.
    
    This function returns a (2,2) np.array with 4 different reward matrices:
    
    R = [[Ri, Rc],
         [Rg, Rb]]
         
    So the initial reward matrix is accessed via R[0,0]. When the croissant has been
    collected, the correct reward matrix is Rc = R[0,1].
    
    This functions is specifically relevant to walking to the robot factory 6x6 grid
    outlined in the Jupyter Notebook.
    """
    # R-initial
    Ri = np.ones(shape=(36,36)) * np.nan
    
    # All moves where reward is -1 for action. Generate programmatically cos writing by hand is tedious
    ones = []
    for i in range(6):
        for j in range(6):
            cell = i*6 + j
            if j != 5:
                ones.append((cell, cell+1)) # move right unless agent is on right edge
            if cell - 6 >= 0:
                ones.append((cell, cell-6)) # move up if not in top row
            if cell + 6 < 36:
                ones.append((cell, cell+6)) # move down if cell not in bottom row
            if j != 0:
                ones.append((cell, cell-1)) # move left if not on left edge
            ones.append((cell, cell)) # staying still is possible, why not?
    # add tube lines
    ones.extend([(0,23), (23,0), (8,25), (25,8)])
    ones = tuple(zip(*ones))
    Ri[ones] = r_time

    # now dissallow moves across walls. Just dissallow moves one way across a wall...
    nans = [(2,3), (8,9), (14,15), (18, 24), (19,25), (32,33), (11,17)]
    # ... and now dissallow moves the other way:
    nans.extend([(t[1], t[0]) for t in nans])
    Ri[tuple(zip(*nans))] = np.nan

    def move_to(l, cell):
        for i in [-6, -1, 1, 6]:
            l.append((cell + i, cell))
        return l

    # don't fall in the pond!
    ponds = move_to(move_to([], 16), 27)
    # staying in the pond is also pretty unpleasant. Brrrr!
    ponds.extend([(16,16), (27,27)])
    ponds = tuple(zip(*ponds))
    Ri[ponds] = r_pond

    # nice to eat a croissant before work
    crois = move_to([], 10)
    crois = tuple(zip(*crois))
    Ri[crois] = r_croissant

    # finally we write something by hand!
    Ri[(26, 31), (32, 32)] = r_cogs

    Ri[(29, 34, 35), (35, 35, 35)] = r_work

    ##########
    # Now we have our reward matrix, we make three new reward matrices
    # which don't have rewards for visiting cogs or croissants if they
    # have already been visited
    ##########

    # visited croissant but not cog
    Rc = Ri.copy()
    Rc[crois] = r_time

    # visited cog but not croissant
    Rg = Ri.copy()
    Rg[(26, 31), (32, 32)] = r_time

    # visited cog and croissant
    Rb = Rg.copy()
    Rb[crois] = r_time
    
    return np.array([[Ri, Rc], [Rg, Rb]])