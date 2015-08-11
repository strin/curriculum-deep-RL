
# coding: utf-8

# In[ ]:

import scriptinit
import sys
import numpy as np
import matplotlib
import pylab as plt
from hypercube import *
from agent import RecurrentReinforceAgent


# In[ ]:

# set-up a simple 2-D cube
dimensions = (3, 3)
walls = np.zeros(dimensions)
maze = HyperCubeMaze(dimensions=dimensions, action_stoch=0., grid=walls)
task = HyperCubeMazeTask(maze, wall_penalty=-0.1, time_penalty=0., reward=4., gamma=0.9)


# In[ ]:

# set a simple 2-corner goal
goal_vec = np.zeros((4, 1))
goal_vec[0] = 1.
goal_vec[1] = 1.
task.set_goals(goal_vec)


# In[ ]:

# initialize the agent
rr_agent = RecurrentReinforceAgent(task, hidden_dim=128, num_samples=10)


# In[ ]:

# set-up the experiment
def experiment(agent, task, NUM_EPISODES):
    def trial():
        task.reset()
        current_state = task.get_current_state()
        steps = 0
        while True:
            steps += 1
            action = agent.get_action(current_state)
            next_state, reward = task.perform_action(action)
            if reward > 0:
                print next_state
                print 'hit!'
                sys.stdout.flush()
                sys
            if task.is_terminal():
                agent.end_episode(reward)
                return steps
            else:
                agent.learn(next_state, reward)
                current_state = next_state
    
    print task.goals
    for episode in xrange(NUM_EPISODES):
        num_steps = trial()
        print 'Number of steps: ', num_steps


# In[ ]:

# run the experiment
print experiment(rr_agent, task, NUM_EPISODES=500)


# In[ ]:




# In[ ]:




# In[ ]:



