import numpy as np
import matplotlib.pyplot as plt


def find_path(Q):
    state = 6
    path = [state]
    # path = []
    end_state = False

    while not end_state:
        old_state = state
        state = np.where(Q[old_state,] == Q[old_state,].max())[0][0]
        if state not in path:
            path.append(state)
            if state == 35:
                end_state = True

        elif state == old_state:
            print("The Agent chose to stay in his position")
            end_state = True

        else:
            print("The Agent stucked into a loop")
            end_state = True

    return path

def calculate_rewards(R_matrix, path):
    r = 0
    steps = 0
    for idx in range(len(path)-1):
        r += R_matrix[path[idx], path[idx+1]]
        steps += 1
    return r,steps

def path_in_grids(path, dims):
    return [(cell//dims[1], cell%dims[0]) for cell in path]

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def average_R_display(Rtot, rolling_periods=None):
    """display average of total rewards over episodes
    
    rolling_periods: None or int
        if None, display the cumulative average.
        if int, display the moving average over n episodes
            this is more helpful for seeing convergence
            as earlier very large/small R-values are discarded.
    """
    if rolling_periods:
        Rtot_avg = moving_average(Rtot, n=rolling_periods)
        x = np.arange(rolling_periods - 1, len(Rtot))
    else:
        Rtot_avg = (Rtot.cumsum() / np.arange(1, len(Rtot) + 1))[20:]
        x = np.arange(20, len(Rtot))
        
    plt.plot(x, Rtot_avg)
    plt.title("Average Rewards per episode")
    plt.show()

def steps_plot(a_hist, rolling_periods=None):
    """
    rolling_periods: None or int
        if None, display the cumulative average.
        if int, display the moving average over n episodes
            this is more helpful for seeing convergence
            as earlier very large/small R-values are discarded.
    """
    y = [len(h) for h in a_hist]
    x = np.arange(len(a_hist))

    if rolling_periods:
        y_avg = moving_average(y, n=rolling_periods)
        x = np.arange(rolling_periods - 1, len(y))
    else:
        y_avg = (np.cumsum(y) / np.arange(1, len(y) + 1))[20:]
        x = np.arange(20, len(y))
    plt.plot(x, y_avg)
    plt.title("Average steps per episode")
    plt.show()


def plot_Q_changing(Q_hist, absolute=True, rolling_periods=None):
    
    if absolute:
        difference_Qs = np.absolute(Q_hist[1:,:,:] - Q_hist[:-1, :, :])
    else:
        difference_Qs = Q_hist[1:,:,:] - Q_hist[:-1, :, :]

    Q_plot = np.sum(difference_Qs, axis=(1,2))
    if rolling_periods:
        x = np.arange(rolling_periods - 1, len(Q_plot))
        Q_plot = moving_average(Q_plot, n=rolling_periods)
    else:
        x = np.arange(len(Q_plot))

    plt.plot(x, Q_plot)
    plt.title("Changing in Q matrixs")
    plt.show()