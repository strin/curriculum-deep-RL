
# coding: utf-8

# In[ ]:

import scriptinit
import numpy as np
import util
from t_maze import *
from experiment import *
from agent import RecurrentReinforceAgent


# In[ ]:

# general parameters
hyperparams = {
    # task specific
    'maze_length' : 10,
    'noisy_observations': False,
    'gamma': 0.98,

    # model specific
    'hidden_dimension' : 128,
    'num_trajectory_samples' : 10,
    'truncate_gradient' : -1,  # number of steps to run BPTT (-1 means use the whole sequence)
    'max_trajectory_length' : float('inf'),  # the length of longest (s, a, r) sequence used
    
    
    # experiment specific
    'max_episodes' : 5000,
    'report_wait' : 50,
    'save_wait' : 100,
    'experiment_samples' : 20
}


# load into namespace and log to metadata
for var, val in hyperparams.iteritems():
    exec("{0} = hyperparams['{0}']".format(var))
    util.metadata(var, val)


# In[ ]:

# set-up the task
maze = TMaze(length = maze_length, noise=noisy_observations)
task = TMazeTask(maze, gamma=gamma)

# initialize the agent
rr_agent = RecurrentReinforceAgent(task, hidden_dim=hidden_dimension,
                                   num_samples=num_trajectory_samples,
                                   truncate_gradient=truncate_gradient,
                                   max_trajectory_length=max_trajectory_length)


# In[ ]:

# prepare the experiment
controllers = [BasicController(report_wait=report_wait, save_wait=save_wait, max_episodes=max_episodes)]
observers = [TMazeObserver(num_samples=experiment_samples, report_wait=report_wait)]
experiment = Experiment(rr_agent, task, controllers=controllers, observers=observers)


# In[ ]:

experiment.run_experiments()

