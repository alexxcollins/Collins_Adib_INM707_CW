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


def average_R_display(Rtot):
  Rtot_avg = Rtot.cumsum() / np.arange(1, len(Rtot) + 1)
  plt.plot(np.arange(20, len(Rtot_avg)), Rtot_avg[20:])
  plt.title("Average Rewards per episode")
  plt.show()


def steps_plot(a_hist):
    y = []
    x = np.arange(len(a_hist))
    for his in a_hist:
        y.append(len(his))

    y_avg = np.cumsum(y) / np.arange(1, len(y) + 1)
    plt.plot(np.arange(20, len(y_avg)), y_avg[20:])
    plt.title("Average steps per episode")
    plt.show()


def plot_Q_changing(Q_hist, absolute=True):
    Q_plot = []
    for idx in range(1, len(Q_hist)):
        if absolute:
            difference_Qs = np.absolute(Q_hist[idx] - Q_hist[idx - 1])
        else:
            difference_Qs = Q_hist[idx] - Q_hist[idx - 1]

        sum = np.sum(difference_Qs)

        Q_plot.append(sum)

    plt.plot(np.arange(len(Q_plot)), Q_plot)
    plt.title("Changing in Q matrixs")
    plt.show()