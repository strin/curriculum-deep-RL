# basic components of an agent.
import random
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
import cPickle as pickle
from theano.printing import pydotprint

from pyrl.tasks.task import Environment
import pyrl.layers
import pyrl.optimizers

class Policy(object):
    def get_action(self, state):
        raise NotImplementedError()

class Vfunc(object):
    '''
    value function.
    '''
    def __call__(self, state):
        raise NotImplementedError()

class TabularVfunc(object):
    '''
    A tabular value function
    '''
    def __init__(self, num_states):
        # Tabular representation of state-value function initialized uniformly
        self.num_states = num_states
        self.V = [1. for s in xrange(self.num_states)]

    def __call__(self, state):
        assert state >= 0 and state <= self.num_states
        return self.V[state]

    def update(self, state, val):
        self.V[state] = val

class Qfunc(object):
    '''
    Q state-action value function.
    '''
    def __init__(self, env):
        self.env = env

    def __call__(self, state, action):
        raise NotImplementedError()

    def _get_greedy_action(self, state):
        action_values = []
        for action in self.env.get_allowed_actions():
            value = self.__call__(state, action)
            action_values.append((action, value))
        action_values = sorted(action_values, key=lambda ac: ac[1], reverse=True)
        return action_values[0][0]

class DQN(Qfunc):
    '''
    A deep Q function that uses theano.
    (TODO:) DQN takes states and actions inputs.
    '''
    def __init__(self, env, arch_func):
        Qfunc.__init__(self, env)
        self.arch_func = arch_func

    def visualize_net(self):
        pydotprint(self.action_values, outfile='__pydotprint%d__.png' % id(self), format='png')
        return '__pydotprint%d__.png' % id(self)

    def _initialize_net(self):
        '''
        Initialize the deep Q neural network.
        '''
        # construct computation graph for forward pass
        self.states = T.matrix('states')
        self.action_values, model = self.arch_func(self.states)
        self.params = sum([layer.params for layer in model], [])

        self.fprop = theano.function(inputs=[self.states],
                                     outputs=self.action_values,
                                     name='fprop')

    def apply(self, states):
        '''
        apply the DQN to input states.
        '''
        return self.fprop(states)

    def __call__(self, states, actions):
        '''
        states: any matrix / tensor that fits the arch_func, expect the first dimension
            be data points.
        actions: a 1-d iteratable of actions.
        '''
        resp = self.apply(states)
        values = np.zeros(len(actions))
        for (ni, action) in enumerate(actions):
            values[ni] = resp[ni, action]
        return values









