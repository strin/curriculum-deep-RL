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
import pyrl.prob as prob

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
    def __call__(self, state, action):
        raise NotImplementedError()

    def _get_greedy_action(self, state):
        action_values = []
        for action in self.env.get_allowed_actions():
            value = self.__call__(state, action)
            action_values.append((action, value))
        action_values = sorted(action_values, key=lambda ac: ac[1], reverse=True)
        return action_values[0][0]

    def get_action(self, state):
        return self._get_greedy_action(state)

    def get_action_distribution(self, state, **kwargs):
        '''
        return a dict of action -> probability.
        '''
        # deterministic action.
        action = self.get_action(state)
        return {action: 1.}

    def is_tabular(self):
        '''
        if True, then the Qfunc takes state_id as the state input.
        else, Qfunc takes state_vector
        '''
        raise NotImplementedError()


class DQN(Qfunc):
    '''
    A deep Q function that uses theano.
    '''
    def __init__(self, task, arch_func):
        '''
        epsilon: probability for taking a greedy action.
        '''
        self.arch_func = arch_func
        self.task = task
        self._initialize_net()

    def is_tabular(self):
        return False

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
        self.params = sum([layer.params for layer in model.values()], [])

        self.fprop = theano.function(inputs=[self.states],
                                     outputs=self.action_values,
                                     name='fprop')

    def apply(self, states, actions):
        '''
        states: any matrix / tensor that fits the arch_func, expect the first dimension
            be data points.
        actions: a 1-d iteratable of actions.
        '''
        resp = self.fprop(states)
        values = np.zeros(len(actions))
        for (ni, action) in enumerate(actions):
            values[ni] = resp[ni, action]
        return values

    def _get_eps_greedy_action_distribution(self, state_vector, epsilon):
        # transpose since the DQN expects row vectors
        state_vector = state_vector.reshape(1, -1)

        # uniform distribution.
        probs = [epsilon / self.task.get_num_actions()] * self.task.get_num_actions()

        # increase probability at greedy action..
        action = np.argmax(self.fprop(state_vector))
        probs[action] += 1-epsilon
        return probs

    def _get_eps_greedy_action(self, state_vector, epsilon):
        # transpose since the DQN expects row vectors
        state_vector = state_vector.reshape(1, -1)

        # epsilon greedy w.r.t the current policy
        if(random.random() < epsilon):
            action = np.random.randint(0, self.task.get_num_actions())
        else:
            # a^* = argmax_{a} Q(s, a)
            action = np.argmax(self.fprop(state_vector))
        return action

    def _get_softmax_action_distribution(self, state_vector, temperature):
        state_vector = state_vector.reshape(1, -1)
        qvals = self.fprop(state_vector).reshape(-1)
        qvals = qvals / temperature
        return np.exp(prob.normalize_log(qvals))

    def _get_softmax_action(self, state_vector, temperature):
        probs = self._get_softmax_action_distribution(state_vector, temperature)
        return npr.choice(range(self.task.get_num_actions()), 1, replace=True, p=probs)[0]

    def get_action(self, state_vector, **kwargs):
        if 'method' in kwargs:
            method = kwargs['method']
            if method == 'eps-greedy':
                return self._get_eps_greedy_action(state_vector, kwargs['epsilon'])
            elif method == 'softmax':
                return self._get_softmax_action(state_vector, kwargs['temperature'])
        else:
            return self._get_eps_greedy_action(state_vector, epsilon=0.05)

    def get_action_distribution(self, state_vector, **kwargs):
        if 'method' in kwargs:
            method = kwargs['method']
            if method == 'eps-greedy':
                log_probs = self._get_eps_greedy_action_distribution(state_vector, kwargs['epsilon'])
            elif method == 'softmax':
                log_probs = self._get_softmax_action_distribution(state_vector, kwargs['temperature'])
        else: # default, 0.05-greedy policy.
            log_probs = self._get_eps_greedy_action_distribution(state_vector, epsilon=0.05)
        return {action: log_probs[action] for action in range(self.task.get_num_actions())}


    def __call__(self, state_vector, action):
        actions = [action]
        return self.apply(state_vector.reshape(1, -1), actions)

def compute_Qfunc_logprob(qfunc, task, softmax_t = 1.):
    '''
        the Qfuncs are normalized to a softmax destribution.
    '''
    table = np.zeros((task.get_num_states(), task.get_num_actions()))
    states = task.get_valid_states()
    for state in states:
        for action in range(task.get_num_actions()):
            table[state, action] = qfunc(state, action) / softmax_t
        table[state, :] = prob.normalize_log(table[state, :])
    return table

def compute_Qfunc_V(qfunc, task):
    table = np.zeros(task.get_num_states())
    states = task.get_valid_states()
    for state in states:
        vals = []
        for action in range(task.get_num_actions()):
            if not qfunc.is_tabular():
                vals.append(qfunc(task.wrap_stateid(state), action))
            else:
                vals.append(qfunc(state, action))
        table[state] = max(vals)
    return table

def eval_policy_reward(policy, task, num_episodes = 100):
    task.reset()
    while task.is_terminal():
        task.reset()

    curr_state = task.get_current_state()

    total_reward = 0.
    num_steps = 1.

    for ei in range(num_episodes):
        # TODO: Hack!
        while True:
            if num_steps >= 200:
                break

            action = policy.get_action(curr_state)

            curr_state, reward = task.perform_action(action)
            total_reward += reward * task.gamma ** (num_steps)

            num_steps += 1

    return total_reward / num_episodes

