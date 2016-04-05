# basic components of an agent.
import random
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
import cPickle as pickle
# from theano.printing import pydotprint

import pyrl.layers
import pyrl.optimizers
import pyrl.prob as prob

class Policy(object):
    def get_action(self, state, valid_actions=None):
        raise NotImplementedError()


class RandomPolicy(object):
    def __init__(self, num_actions):
        self.num_actions = num_actions


    def get_action(self, state, valid_actions=None):
        if not valid_actions:
            valid_actions = range(self.num_actions)
        action = prob.choice(valid_actions, 1)[0]
        return action


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

    def copy(self):
        import dill as pickle
        return pickle.loads(pickle.dumps(self))

    def is_tabular(self):
        '''
        if True, then the Qfunc takes state_id as the state input.
        else, Qfunc takes state_vector
        '''
        raise NotImplementedError()


class QfuncTabular(Qfunc):
    '''
    a tabular representation of Q funciton.
    '''
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.table = np.zeros((num_states, num_actions))

    def is_tabular(self):
        return True

    def __call__(self, state, action):
        '''
        given state (int), and action (int), output the Q value.
        '''
        assert(type(state) == int)
        assert(type(action) == int)
        return self.table[state, action]

    def _get_eps_greedy_action_distribution(self, state, epsilon):
        raise NotImplementedError()

    def _get_eps_greedy_action(self, state, epsilon, valid_actions):
        if(random.random() < epsilon):
            action = npr.choice(valid_actions, 1)[0]
        else:
            # a^* = argmax_{a} Q(s, a)
            vals = [self.table[state, a] for a in valid_actions]
            max_poses = np.argwhere(vals == np.amax(vals)).reshape(-1)
            action_i = npr.choice(max_poses, 1)
            action = valid_actions[action_i]
        return action

    def _get_softmax_action_distribution(self, state, temperature, valid_actions=None):
        if valid_actions == None:
            valid_actions = range(self.num_actions)
        qvals = self.table[state, valid_actions]
        qvals = qvals / temperature
        p = np.exp(prob.normalize_log(qvals))
        return p

    def _get_softmax_action(self, state, temperature, valid_actions):
        probs = self._get_softmax_action_distribution(state, temperature, valid_actions)
        return npr.choice(valid_actions, 1, replace=True, p=probs)[0]

    def get_action(self, state, **kwargs):
        if 'valid_actions' in kwargs:
            valid_actions = kwargs['valid_actions']
        else:
            valid_actions = range(self.num_actions) # do not have a valid actions constraints.
        if 'method' in kwargs:
            method = kwargs['method']
            if method == 'eps-greedy':
                return self._get_eps_greedy_action(state, kwargs['epsilon'], valid_actions=valid_actions)
            elif method == 'softmax':
                return self._get_softmax_action(state, kwargs['temperature'], valid_actions=valid_actions)
        else:
            return self._get_eps_greedy_action(state, epsilon=0.05, valid_actions=valid_actions)

    def get_action_distribution(self, state, **kwargs):
        if 'method' in kwargs:
            method = kwargs['method']
            if method == 'eps-greedy':
                log_probs = self._get_eps_greedy_action_distribution(state, kwargs['epsilon'])
            elif method == 'softmax':
                log_probs = self._get_softmax_action_distribution(state, kwargs['temperature'])
        else: # default, 0.05-greedy policy.
            log_probs = self._get_eps_greedy_action_distribution(state, epsilon=0.05)
        return {action: log_probs[action] for action in range(self.num_actions)}

    def av(self, state):
        return self.table[state, :]

class DQN(Qfunc):
    '''
    A deep Q function that uses theano.
    TODO: dependence on task is only on task.num_actions.
    '''
    def __init__(self, num_actions, arch_func, state_type=T.matrix):
        '''
        epsilon: probability for taking a greedy action.
        '''
        self.arch_func = arch_func
        self.num_actions = num_actions
        self.state_type = state_type
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
        self.states = self.state_type('states')
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
        probs = [epsilon / self.num_actions] * self.num_actions

        # increase probability at greedy action..
        action = np.argmax(self.fprop(state_vector))
        probs[action] += 1-epsilon
        return probs

    def _get_eps_greedy_action(self, state, epsilon, valid_actions):
        # transpose since the DQN expects row vectors
        state = state.reshape(1, *state.shape)

        # epsilon greedy w.r.t the current policy
        if(random.random() < epsilon):
            action = npr.choice(valid_actions, 1)[0]
        else:
            # a^* = argmax_{a} Q(s, a)
            resp = self.fprop(state)[0]
            action_i = np.argmax([resp[a] for a in valid_actions])
            action = valid_actions[action_i]
        return action

    def _get_softmax_action_distribution(self, state, temperature, valid_actions=None):
        if valid_actions == None:
            valid_actions = range(self.num_actions)
        state = state.reshape(1, *state.shape)
        qvals = self.fprop(state).reshape(-1)[valid_actions]
        qvals = qvals / temperature
        p = np.exp(prob.normalize_log(qvals))
        return p

    def _get_softmax_action(self, state_vector, temperature, valid_actions):
        probs = self._get_softmax_action_distribution(state_vector, temperature, valid_actions)
        return npr.choice(valid_actions, 1, replace=True, p=probs)[0]

    def get_action(self, state, **kwargs):
        if 'valid_actions' in kwargs:
            valid_actions = kwargs['valid_actions']
        else:
            valid_actions = range(self.num_actions) # do not have a valid actions constraints.
        if 'method' in kwargs:
            method = kwargs['method']
            if method == 'eps-greedy':
                return self._get_eps_greedy_action(state, kwargs['epsilon'], valid_actions=valid_actions)
            elif method == 'softmax':
                return self._get_softmax_action(state, kwargs['temperature'], valid_actions=valid_actions)
        else:
            return self._get_eps_greedy_action(state, epsilon=0.05, valid_actions=valid_actions)

    def get_action_distribution(self, state_vector, **kwargs):
        if 'method' in kwargs:
            method = kwargs['method']
            if method == 'eps-greedy':
                log_probs = self._get_eps_greedy_action_distribution(state_vector, kwargs['epsilon'])
            elif method == 'softmax':
                log_probs = self._get_softmax_action_distribution(state_vector, kwargs['temperature'])
        else: # default, 0.05-greedy policy.
            log_probs = self._get_eps_greedy_action_distribution(state_vector, epsilon=0.05)
        return {action: log_probs[action] for action in range(self.num_actions)}

    def __call__(self, state_vector, action):
        actions = [action]
        return self.apply(state_vector.reshape(1, -1), actions)

    def av(self, state):
        return self.fprop(np.array([state]))[0]

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

