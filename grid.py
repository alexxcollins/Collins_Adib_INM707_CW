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
        self.goal_state = goal_state # should be beter way to code this
        self.croissant = 10 # should be beter way to code this
        self.cogs = 32 # should be beter way to code this
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.rng = np.random.default_rng(42)
        
    def run_episode(self, Q, alpha, gamma, epsilon):
        """Run one episode of the game.
        
        To do: verbose mode where we print off calculations etc as we go
        """
        s = self.start
        # croissant and cogs haven't been visited yet:
        idx_c = 0
        idx_g = 0
        
        Rtot = 0 # keep a track of total reward earnt during episode
        # keep track of actions and rewards in a numpy:
        ar_pairs = np.ones((self.max_steps,2)) * np.nan
        # array. Do in numpy array for easy broadcasting and array arithmetic.
        
        for t in range(self.max_steps):
            R = self._get_R(self.RM, idx_g, idx_c)
            available, best = self._get_actions(R, Q, s)
            
            # update states:
            a = self._get_greedy_action(epsilon, available, best)
            s_old = s
            s = a
            
            # update Q:
            Q[s_old, a] = Q[s_old, a] + alpha * (R[s_old, a] +
                                                gamma * Q[s, :].max() -
                                                Q[s_old, a])
            
            # update total accumulated reward for this episode
            Rtot += R[s_old, a]
            
            # update action list
            ar_pairs[t, :] = np.array([[a, R[s_old, a]]])
            
            # update memory if croissant or cogs have been visited
            idx_g, idx_c = self._update_memory(s, idx_g, idx_c)
            
            if s == self.goal_state:
                break
            
        return Rtot, Q, ar_pairs
    
    def Q_learning(self, alpha, gamma,
                   eps_dict={'epsilon':0.9}):
        """Run Q-learning algorithm for given hyper-parameters
        
        Pass values for alpha and gamma.
        Epsilon and epsilon decay parameters are passed as a dictionary
        for greater flexibility should the child class define different
        decay functions for epsilon.
        """
        Q = self.create_Q()
        # this creates arracy of shape (self.max_episodes,)
        epsilon = self.epsilon_decay(**eps_dict)
        
        # create np arrays to store Q-matrix and total rewards for 
        # each episode
        Qs = np.zeros(shape = Q[np.newaxis].shape)
        Rtot = np.array([])
        # set up array to store action reward pairs for all step
        # in each episode
        ar_pairs = np.ones((self.max_episodes,
                            self.max_steps,
                            2)) * np.nan
        
        for episode in range(self.max_episodes):
            r, Q, ar = self.run_episode(Q, alpha, gamma, epsilon[episode])
            Rtot = np.concatenate((Rtot, np.array([r])))
            Qs = np.concatenate((Qs, Q[np.newaxis]), axis=0)
            ar_pairs[episode] = ar[np.newaxis]
            
            # (there is currently no early stopping defined)
            # if self._early_stop():
            #     break
        
        return Rtot, Qs, ar_pairs
        
    def display_matrix(self, M, start_idx=None, end_idx=None):
        """Display a formatted pandas DataFrame.
        
        Intended use if for Q or R matrix.
        
        start_idx and end_idx are options inputs to restrict slice
        the dataframe output to df.loc[start_idx:end_idx, start_idx:end_idx]
        
        """
        pd.set_option("display.max_columns", None)
        display(pd.DataFrame(M).loc[start_idx:end_idx, start_idx:end_idx])
        
    def _get_actions(self, R, Q, s):
        """Returns best and all available actions as lists
        """
        available = np.where(~np.isnan(R[s]))[0]
        q_vals = [Q[s,a] for a in available]
        best = available[np.where(q_vals == np.max(q_vals))[0]]
        return available, best
    
    def _get_greedy_action(self, epsilon, available, best):
        """Given epsilon, and available and best actions,
        Pick an appropriate action.
        """
        if self.rng.uniform() > epsilon:
            a = self.rng.choice(best)
        else:
            a = self.rng.choice(available)
        return a
    
    def epsilon_decay(self, epsilon, epsilon_decay_1,
                      epsilon_decay_2,
                      epsilon_decay_threshold,
                      episodes=None):
        """array of epsilon value. 
        
        Epsilon value can be accessed via episode index during training
        The full array can be used to plot epsilon to visualise decay
        """
        if episodes is None: episodes = self.max_episodes
        
        # Calculate cumulative effect of each decay factor
        d1 = np.ones(episodes) * epsilon_decay_1
        d2 = np.ones(episodes) * epsilon_decay_2
        d1c = d1.cumprod()
        d2c = d2.cumprod()

        # calculate epsilon array, using d1c and d2c after threshold breached
        eps = d1c * epsilon
        thresh_idx = np.argmax(eps < epsilon_decay_threshold)
        # after threshold has been reached, epsilon decays be decay_2
        eps[thresh_idx:] = eps[thresh_idx] * d2c[thresh_idx:] / d2c[thresh_idx]
        
        return eps
    
    def _update_memory(self, s, idx_g=None, idx_c=None):
        """if croissant or cogs have been visited, set appropirate idx to 1
        """
        if s == self.croissant: idx_c = 1
        if s == self.cogs: idx_g =1
        return idx_g, idx_c
    
    def _get_R(self, RM, idx_g, idx_c):
        """Return correct reward matrix depending on history
        of the agent
        """
        return self.RM[idx_g, idx_c]
    
    def create_Q(self):
        return np.zeros(self.RM[0,0].shape)
    
    def _early_stop():
        return False

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