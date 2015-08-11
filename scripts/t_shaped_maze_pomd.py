
# coding: utf-8

# In[ ]:

import scriptinit
import numpy as np
import matplotlib
import pylab as plt
from t_maze import *
from agent import RecurrentReinforceAgent


# In[ ]:

# set-up a simple maze with small N
length = 5
maze = TMaze(length, noise=False)
maze_task = TMazeTask(maze, gamma=0.98)


# In[ ]:

# initialize the agent
rr_agent = RecurrentReinforceAgent(maze_task, hidden_dim=128, num_samples=10)


# In[ ]:

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
                
    def percent_successful(trials=25):
        num_success = 0.
        for _ in xrange(trials):
            steps, reward = run()
            if(reward > 0):
                num_success += 1
        return num_success / float(trials)
    
    step = 50
    total_steps = 0.
    for episode in xrange(0, MAX_EPISODES, step):
        for _ in xrange(step):
            steps, reward = run()
            total_steps += steps
        per_succ = percent_successful()
        
        print 'Percent successful: ', per_succ
    
        if per_succ > .80:
            break
    
    return total_steps


# In[ ]:

# run the experiment!
experiment(rr_agent, maze_task, MAX_EPISODES=50000)

