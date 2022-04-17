import matplotlib.pyplot as plt
from IPython import display
from Agent import Agent
import numpy as np

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
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
    
def training_loop(game, model_name, load_model=False, 
                  get_observation = 'relative_snake',
                  greedy=True, 
                  double_dqn=False, 
                  dueling_dqn=False,
                  num_episodes=1000):
    
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

            states, actions, rewards, new_states, dones = agent.get_memory_sample()
            agent.learn(states, actions, rewards, new_states, dones)

            if score > record:
                record = score
                if greedy:
                    # don't save the model if evaluating it
                    agent.save_model(model_name)



            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.number_episodes
            plot_mean_scores.append(mean_score)
            #print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Mean Score: ', mean_score)
            plot(plot_scores,plot_mean_scores)
            
    return agent, np.array(plot_scores), np.array(plot_mean_scores)