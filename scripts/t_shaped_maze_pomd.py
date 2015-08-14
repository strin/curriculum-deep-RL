
# coding: utf-8

# In[28]:

import scriptinit
import numpy as np
import matplotlib
import pylab as plt
from IPython import display
from t_maze import *
from agent import RecurrentReinforceAgent


# In[29]:

# set-up a simple maze with small N
length = 3
maze = TMaze(length, noise=False)
maze_task = TMazeTask(maze, gamma=0.98)


# In[30]:

# initialize the agent
rr_agent = RecurrentReinforceAgent(maze_task, hidden_dim=36, num_samples=10)


# In[31]:

# set-up the basic task
def experiment(agent, task, MAX_EPISODES):
    '''
        Number of runs until first success
    '''
    
    def run():
        task.reset()
        curr_observation = task.get_start_state()  # the initial observation (left or right)
        steps = 0
        while True:
            steps += 1
            action = agent.get_action(curr_observation)
            next_observation, reward = task.perform_action(action)
            if next_observation is None:
                agent.end_episode(reward)
                return steps, reward
            else:
                agent.learn(next_observation, reward)
                curr_observation = next_observation
                
    def percent_successful(trials=10):
        num_success = 0.
        for _ in xrange(trials):
            steps, reward = run()
            if(reward > 0):
                num_success += 1
        return num_success / float(trials)
    
    step = 50
    total_steps = 0.
    episodes = []
    success_percentages = []
    for episode in xrange(0, MAX_EPISODES, step):
        per_succ = percent_successful()
        episodes.append(episode)
        success_percentages.append(per_succ)

        plt.plot(episodes, success_percentages, 'b')
        plt.xlabel('Iterations')
        plt.ylabel('Number of Steps')
        plt.title('Number of Steps (Avg. over 20 episodes) to Goal Completion versus Time')
        display.display(plt.gcf())
        display.clear_output(wait=True)

        if per_succ > .90:
            break
        for _ in xrange(step):
            steps, reward = run()
            total_steps += steps

    return total_steps


# In[33]:

# run the experiment!
experiment(rr_agent, maze_task, MAX_EPISODES=50000)


# In[ ]:




# In[ ]:



