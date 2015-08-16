
# coding: utf-8

# In[ ]:

import scriptinit
import numpy as np
import util
from hypercube import *
from experiment import *
from agent import RecurrentReinforceAgent


# In[ ]:

# general parameters
hyperparams = {
    # task specific
    'dimensions' : (5, 5),
    'action_stochasticity' : 0.,
    'wall_penalty': -0.1,
    'time_penalty' : -0.1,
    'reward' : 4,
    'gamma' : 0.9,

    # model specific
    'hidden_dimension' : 128,
    'num_trajectory_samples' : 10,
    'truncate_gradient' : -1,  # number of steps to run BPTT (-1 means use the whole sequence)
    'max_trajectory_length' : float('inf'),  # the length of longest (s, a, r) sequence used
    
    
    # experiment specific
    'max_episodes' : 5000,
    'report_wait' : 50,
    'save_wait' : 100,
}

# load into namespace and log to metadata
for var, val in hyperparams.iteritems():
    exec("{0} = hyperparams['{0}']".format(var))
    util.metadata(var, val)


# In[ ]:

# set up the task
dimensions = (5, 5)
world = np.zeros(dimensions)
maze = HyperCubeMaze(dimensions=dimensions, action_stoch=0., grid=world)
task = HyperCubeMazeTask(maze, wall_penalty=-0.1, time_penalty=-0.1, reward=4., gamma=0.9)


# for debugging, let's just use a simple 2-corner goal (assumes the world is 2D!)
goal_vec = np.zeros((4, 1))
goal_vec[0] = 1.
goal_vec[3] = 1.
task.set_goals(goal_vec)

# initialize the agent
rr_agent = RecurrentReinforceAgent(task, hidden_dim=hidden_dimension,
                                   num_samples=num_trajectory_samples,
                                   truncate_gradient=truncate_gradient,
                                   max_trajectory_length=max_trajectory_length)


# In[ ]:

# set up the experiment environment
controllers = [BasicController(report_wait=report_wait, save_wait=save_wait, max_episodes=max_episodes)]
observers = [HyperCubeObserver(report_wait=report_wait)]
experiment = Experiment(rr_agent, task, controllers=controllers, observers=observers)


# In[ ]:

# launch experiment
experiment.run_experiments()


# In[ ]:



