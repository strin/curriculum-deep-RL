
# coding: utf-8

# In[ ]:

import scriptinit
import numpy as np
import argparse
import random
import util
from hypercube import *
from experiment import *
from diagnostics import VisualizeTrajectoryController
from agent import DQN


# In[ ]:

# set random seeds
# When running final experiments, also set the numpy seed so the run is reproducible.
SEED=999


# In[ ]:

# step up argument parsing
parser = argparse.ArgumentParser()

def tup(s):
    try:
        s = s[1:-1]  # strip off the ( ) 
        return tuple(map(int, s.split(',')))
    except:
        raise argparse.ArgumentTypeError("Must give a tuple!")

# task arguments
parser.add_argument('-d', '--dimensions', type=tup, required=True)  # expects a (x, y, z, ...) tuple
parser.add_argument('-as', '--action_stochasticity', type=float, required=True)
parser.add_argument('-wp', '--wall_penalty', type=float, required=True)
parser.add_argument('-tp', '--time_penalty', type=float, required=True)
parser.add_argument('-r', '--reward', type=float, required=True)
parser.add_argument('-g', '--gamma', type=float, required=True)
parser.add_argument('-ms', '--maximum_steps', type=int)


# model arguments
parser.add_argument('-hd', '--hidden_dimension', type=int, required=True)
parser.add_argument('-lr', '--lr', type=float, required=True)
parser.add_argument('-eps', '--epsilon', type=float, required=True)

# curriculum argument
parser.add_argument('-up', '--update_every', type=float, required=True)


# experiment arguments
parser.add_argument('-me', '--max_episodes', type=int, required=True)
parser.add_argument('-rw', '--report_wait', type=int, required=True)
parser.add_argument('-sw', '--save_wait', type=int, required=True)
parser.add_argument('-vw', '--visualize_wait', type=int)
parser.add_argument('-fo', '--fully_observed', type=int, required=True) 
parser.add_argument('-ss', '--state_samples', type=int, required=True)
parser.add_argument('-es', '--eval_samples', type=int, required=True)


# In[ ]:

if util.in_ipython():
    args = parser.parse_args(['-d','(5, 5)', '-as', '0.', '-wp', '-0.1', '-tp', '-0.1', '-r', '4', '-g', '0.9',
                              '-hd', '128', '-lr', '0.05', '-eps', '0.15', '-me', '1000', '-rw', '2',
                              '-sw', '5', '-fo', '1', '-ss', '25', '-es', '5', '-ms', '500', '-up', '1'])
else:
    args = parser.parse_args()

hyperparams = vars(args)

# load into namespace and log to metadata
for var, val in hyperparams.iteritems():
    exec("{0} = hyperparams['{0}']".format(var))
    util.metadata(var, val)


# In[ ]:

# set up the dataset
goal_length = 2 ** len(dimensions)
goals = ["".join(seq) for seq in itertools.product("01", repeat=goal_length)]  # all bit strings of length # corners in world

# remove the all zeros strings and convert to numpy 1D arrays
goals = [np.asarray(list(g)).astype(int) for g in goals if int(g) > 0]

# randomly shuffle the data (ensure the same train/test split between runs...)
random.seed(SEED)
random.shuffle(goals)

# get goal training/test splits
split = int(0.8 * len(goals))
train, test = goals[:split], goals[split:]


# In[ ]:

# initialize the hypercube maze and the task
world = np.zeros(dimensions)  # no walls for now...
maze = HyperCubeMaze(dimensions=dimensions, action_stoch=action_stochasticity, grid=world)

task = HyperCubeMazeTask(hypercubemaze=maze, initial_goal=train[0],
                         wall_penalty=wall_penalty, time_penalty=time_penalty,
                         reward=reward, gamma=gamma, fully_observed=fully_observed,
                         maximum_steps=maximum_steps)


# In[ ]:

# compile the agent
agent = DQN(task, hidden_dim=hidden_dimension, lr=lr, epsilon=epsilon)


# In[ ]:

# set up the experiment environment
controllers = [BasicController(report_wait=report_wait, save_wait=save_wait, max_episodes=max_episodes)]

# fixed curriculum for now (uniformly sample a goal on each episode)... this is something to experiment with!
controllers.append(HyperCubeCurriculum(goals=train, update_every=update_every, curr_type='uniform'))

if len(dimensions) == 2 and visualize_wait is not None and visualize_wait > 0:
    controllers.append(VisualizeTrajectoryController(visualize_wait=visualize_wait, dir_name='trajectories'))

observers = [AverageRewardObserver(report_wait=report_wait), AverageQValueObserver(state_samples=state_samples, report_wait=report_wait)]


goal_dsets = {'train' : train, 'test': test}
observers.append(HyperCubeObserver(goal_dsets=goal_dsets, eval_samples=eval_samples, report_wait=report_wait))

experiment = Experiment(agent, task, controllers=controllers, observers=observers)


# In[ ]:

# launch experiment
experiment.run_experiments()


# In[ ]:




# In[ ]:



