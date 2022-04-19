import matplotlib.pyplot as plt
from IPython import display
from Agent import Agent
import numpy as np

plt.ion()

def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    display.clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)

def plot(scores, mean_scores, title=None, clear_output=True):
    if clear_output:
        display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    if not title:
        plt.title('Training...')
    else:
        plt.title(title)
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], '{:.2f}'.format(scores[-1]))
    # plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], '{:.2f}'.format(mean_scores[-1]))  
    plt.show(block=False)
    plt.pause(.1)
    
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    
def ax_plot(ax, scores, mean_scores, window=100, title=None):
    ax.set_xlabel('Number of Games')
    ax.set_ylabel('Score')
    if title:
        ax.set_title(title)

    y = np.concatenate((scores[:, np.newaxis], mean_scores[:, np.newaxis]),
                       axis=1)
    
    if window:
        roll_mean = moving_average(scores, n=window)
        roll_mean = np.pad(roll_mean,
                           (len(scores) - len(roll_mean), 0),
                           'constant', constant_values=np.nan)
        y = np.concatenate((y, roll_mean[:, np.newaxis]), axis=1)
    
    ax.plot(y)
    ax.text(len(scores)-1, scores[-1], '{:.2f}'.format(scores[-1]), fontsize=8)
    ax.text(len(mean_scores)-1, mean_scores[-1], '{:.2f}'.format(mean_scores[-1]))
    
def training_loop(game, model_name, load_model=False, 
                  get_observation = 'relative_snake',
                  greedy=True, 
                  double_dqn=False, 
                  dueling_dqn=False,
                  num_episodes=1000,
                  plot_update_at_end=False):
    
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    epsilons = []
    agent = Agent(double_dqn=double_dqn,
                  dueling_dqn=dueling_dqn,
                  game=game, greedy=greedy)
    if load_model:
        agent.load_model(model_name)
        print('loaded {}'.format(model_name))

    episode = 0
    while episode < num_episodes:
        state = agent.get_observation()
        action = agent.choose_action(state)
        reward, done, score = agent.game.play_step(action)
        new_state = agent.get_observation()
        # remember
        agent.remember(state, action, reward, new_state, done)

        if done:
            episode += 1
            # train long memory, plot result
            agent.game.reset()
            agent.update_policy()

            states, actions, rewards, new_states, dones = agent.get_memory_sample()
            agent.learn(states, actions, rewards, new_states, dones)

            # Alex: I don't think we should do the below: while running greedy policy
            # there is randomness to when the snake gets a high score anyway.
            # if score > record:
            #     record = score
            #     if greedy:
            #         # don't save the model if evaluating it
            #         agent.save_model(model_name)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.number_episodes
            plot_mean_scores.append(mean_score)
            #print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Mean Score: ', mean_score)
            if plot_update_at_end and not episode == num_episodes:
                update_progress(episode/num_episodes)
                print('{}; episode: {}'.format(model_name.split('.')[0], episode))
            else:
                plot(plot_scores, plot_mean_scores)
                print(episode)
                
            
            
    return agent, np.array(plot_scores), np.array(plot_mean_scores)