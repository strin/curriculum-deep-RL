
# coding: utf-8

# In[ ]:

import scriptinit
import numpy as np
import util
from hypercube import *
from experiment import *
from agent import RecurrentReinforceAgent, DQN


# In[ ]:

# general parameters
hyperparams = {
    # task specific
    'dimensions' : (5, 5),
    'action_stochasticity' : 0.0,
    'wall_penalty': -0.1,
    'time_penalty' : -0.1,
    'reward' : 4,
    'gamma' : 0.9,

#     # model specific (RecurrentReinforce Agent)
#     'hidden_dimension' : 128,
#     'num_trajectory_samples' : 10,
#     'truncate_gradient' : -1,  # number of steps to run BPTT (-1 means use the whole sequence)
#     'max_trajectory_length' : float('inf'),  # the length of longest (s, a, r) sequence used
    
    # model specific (DQN)
    'hidden_dimension' : 128,
    'lr' : 0.05,
    'epsilon': 0.15,
    
    # experiment specific
    'max_episodes' : 500,
    'report_wait' : 50,
    'save_wait' : 100,
    'fully_observed': True
}

# load into namespace and log to metadata
for var, val in hyperparams.iteritems():
    exec("{0} = hyperparams['{0}']".format(var))
    util.metadata(var, val)


# In[ ]:

# set up the task
world = np.zeros(dimensions)
maze = HyperCubeMaze(dimensions=dimensions, action_stoch=0., grid=world)
task = HyperCubeMazeTask(maze, wall_penalty=-0.1, time_penalty=-0.1, reward=4., gamma=0.9, fully_observed=True)


# for debugging, let's just use a simple 2-corner goal (assumes the world is 2D!)
goal_vec = np.random.randint(0, 2, size=(2 ** len(dimensions), 1))
while np.sum(goal_vec) == 0.:
    goal_vec = np.random.randint(0, 2, size=(2 ** len(dimensions), 1))

task.set_goals(goal_vec)

# initialize the agent
# agent = RecurrentReinforceAgent(task, hidden_dim=hidden_dimension,
#                                 num_samples=num_trajectory_samples,
#                                 truncate_gradient=truncate_gradient,
#                                 max_trajectory_length=max_trajectory_length)

agent = DQN(task, hidden_dim=hidden_dimension, lr=lr, epsilon=epsilon)


# In[ ]:

# set up the experiment environment
controllers = [BasicController(report_wait=report_wait, save_wait=save_wait, max_episodes=max_episodes)]
observers = [HyperCubeObserver(report_wait=report_wait), AverageRewardObserver(report_wait=report_wait)]
experiment = Experiment(agent, task, controllers=controllers, observers=observers)


# In[ ]:

# launch experiment
experiment.run_experiments()


# In[ ]:



