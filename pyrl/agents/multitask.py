import random
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
import cPickle as pickle
# from theano.printing import pydotprint

from pyrl.tasks.task import Environment
from pyrl.agents.agent import DQN
import pyrl.layers
import pyrl.optimizers

class QfuncMT(object):
    def __init__(self, tasks):
        self.tasks = tasks

    def __call__(self, states, actions, task):
        raise NotImplementedError()

class HordeDQN(QfuncMT):
    '''
    Horde use a combination of agents to address complex knowledge
    composition.
    '''
    def __init__(self, tasks, arch, epsilon=0.05):
        self.tasks = tasks
        self.dqn_by_task = {task: DQN(arch, epsilon=epsilon) for task in tasks}

    def __call__(self, states, actions, task):
        return self.dqn_by_task[task](states, actions)

