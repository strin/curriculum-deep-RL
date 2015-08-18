
# coding: utf-8

# In[ ]:

import scriptinit
import numpy as np
import argparse
import util
from t_maze import *
from experiment import *
from agent import RecurrentReinforceAgent


# In[ ]:

# step up argument parsing
parser = argparse.ArgumentParser()

# task arguments
parser.add_argument('-ml', '--maze_length', type=int, required=True)
parser.add_argument('-no', '--noisy_observations', type=int, required=True) 
parser.add_argument('-g', '--gamma', type=float, required=True)

# model arguments
parser.add_argument('-hd', '--hidden_dimension', type=int, required=True)
parser.add_argument('-ns', '--num_trajectory_samples', type=int, required=True)
parser.add_argument('-tg', '--truncate_gradient', type=int, default=-1)  # number of steps to run BPTT (-1 means use the whole sequence)
parser.add_argument('-mt', '--max_trajectory_length', type=float, default=float('inf'))  # the length of longest (s, a, r) sequence used

# experiment specific arguments
parser.add_argument('-me', '--max_episodes', type=int, required=True)
parser.add_argument('-rw', '--report_wait', type=int, required=True)
parser.add_argument('-sw', '--save_wait', type=int, required=True)
parser.add_argument('-ex', '--experiment_samples', type=int, required=True)


# In[ ]:

# load arguments into namespace and log to metadata
if util.in_ipython():
    args = parser.parse_args(['-ml', '3', '-no', '0', '-g', '0.98', '-hd', '128', '-ns', '10', '-me','5000',
                              '-rw', '100', '-sw', '1000', '-ex', '50', '-mt', 'inf'])
else:
    args = parser.parse_args()

hyperparams = vars(args)

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


# In[ ]:



