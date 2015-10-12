import random
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
from IPython.display import Image
from theano.printing import pydotprint
import layers
import optimizers
import cPickle as pickle
from util import unit_vec
from agent import OnlineAgent
import arch

DEFAULT_PARAMS = dict(l2_reg=0.0, lr=1e-3, epsilon=0.05,
                memory_size=250, minibatch_size=64)

class DQN(OnlineAgent):
    ''' Q-learning with a neural network function approximator

        TODO:   - reward clipping
                - epsilon decay
    '''
    def __init__(self, task, arch_func, l2_reg=0.0, lr=1e-3, epsilon=0.05,
                 memory_size=250, minibatch_size=64):
        '''
        arch_func: (states) -> action_values, model
            is a function that specifies the archiecture of the nerual net
        '''
        self.task = task
        self.state_dim = task.get_state_dimension()
        self.num_actions = task.get_num_actions()
        self.gamma = task.gamma
        self.l2_reg = l2_reg
        self.lr = lr
        self.epsilon = epsilon
        self.memory_size = memory_size  # number of experiences to store
        self.minibatch_size = minibatch_size
        self.arch_func = arch_func

        self.model = self._initialize_net()

        # for now, keep experience as a list of tuples
        self.experience = []
        self.exp_idx = 0

        # used for streaming updates
        self.last_state = None
        self.last_action = None

    def __call__(self, state_id):
        return self.fprop(np.transpose(self.task.wrap_stateid(state_id)))

    def visualize_net(self):
        pydotprint(self.action_values, outfile='__pydotprint%d__.png' % id(self), format='png')
        return '__pydotprint%d__.png' % id(self)

    def _initialize_net(self):
        '''
            Attaches fprop and bprop functions to the class
            Simple 2 layer net with l2-loss for now.

        '''
        # construct computation graph for forward pass
        states = T.matrix('states')
        action_values, model = self.arch_func(states)

        self.action_values = action_values
        self.fprop = theano.function(inputs=[states], outputs=action_values,
                                     name='fprop')

        # build computation graph for backward pass (using the variables
        # introduced previously for forward)
        targets = T.vector('target')
        last_actions = T.lvector('action')
        mse = layers.MSE(action_values[T.arange(action_values.shape[0]),
                         last_actions], targets)

        # regularization
        params = sum([layer.params for layer in model.values()], [])

        l2_penalty = 0.
        for param in params:
            l2_penalty += (param ** 2).sum()

        cost = mse + self.l2_reg * l2_penalty

        updates = optimizers.Adam(cost, params, alpha=self.lr)

        # takes a single gradient step
        td_errors = T.sqrt(mse)
        self.bprop = theano.function(inputs=[states, last_actions, targets],
                                     outputs=td_errors, updates=updates)

        return model

    def end_episode(self, reward):
        if self.last_state is not None:
            self._add_to_experience(self.last_state, self.last_action, None,
                                    reward)
        self.last_state = None
        self.last_action = None

    def get_action(self, state):
        state = self.task.wrap_stateid(state)

        # transpose since the DQN expects row vectors
        state = state.reshape(1, -1)

        # epsilon greedy w.r.t the current policy
        if(random.random() < self.epsilon):
            action = np.random.randint(0, self.num_actions)
        else:
            # a^* = argmax_{a} Q(s, a)
            action = np.argmax(self.fprop(state))

        self.last_state = state
        self.last_action = action

        return action

    def _add_to_experience(self, s, a, ns, r):
        # TODO: improve experience replay mechanism by making it harder to
        # evict experiences with high td_error, for example
        if len(self.experience) < self.memory_size:
            self.experience.append((s, a, ns, r))
        else:
            self.experience[self.exp_idx] = (s, a, ns, r)
            self.exp_idx += 1
            if self.exp_idx >= self.memory_size:
                self.exp_idx = 0

    def _update_net(self):
        '''
            sample from the memory dataset and perform gradient descent on
            (target - Q(s, a))^2
        '''

        # don't update the network until sufficient experience has been
        # accumulated
        if len(self.experience) < self.memory_size:
            return

        states = np.zeros((self.minibatch_size, self.state_dim,))
        next_states = np.zeros((self.minibatch_size, self.state_dim))
        actions = np.zeros(self.minibatch_size, dtype=int)
        rewards = np.zeros(self.minibatch_size)

        # sample and process minibatch
        samples = random.sample(self.experience, self.minibatch_size)
        terminals = []
        for idx, sample in enumerate(samples):
            state, action, next_state, reward = sample
            states[idx, :] = state.reshape(-1)
            actions[idx] = action
            rewards[idx] = reward

            if next_state is not None:
                next_states[idx, :] = next_state.reshape(-1)
            else:
                terminals.append(idx)

        # compute target reward + \gamma max_{a'} Q(ns, a')
        next_qvals = np.max(self.fprop(next_states), axis=1)

        # Ensure target = reward when NEXT_STATE is terminal
        next_qvals[terminals] = 0.

        targets = rewards + self.gamma * next_qvals

        self.bprop(states, actions, targets.flatten())

    def learn(self, next_state, reward):
        next_state = self.task.wrap_stateid(next_state)
        self._add_to_experience(self.last_state, self.last_action,
                                next_state, reward)
        self._update_net()

    def save_params(self, path):
        assert path is not None
        print 'Saving params to ', path
        params = {}
        for name, layer in self.model.iteritems():
            params[name] = layer.get_params()
        pickle.dump(params, file(path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def load_params(self, path):
        assert path is not None
        print 'Restoring params from ', path
        params = pickle.load(file(path, 'r'))
        for name, layer in self.model.iteritems():
            layer.set_params(params[name])

def update_default_params(new_params):
    args = dict(DEFAULT_PARAMS)
    args.update(new_params)
    return args

def DQN_2Layer(task, hidden_dim, **kwargs):
    args = update_default_params(kwargs)
    return DQN(task,
               lambda states:
                    arch.two_layer(states,
                                   input_dim=task.get_state_dimension(),
                                   hidden_dim=hidden_dim,
                                   outputdim=task.get_num_actions()),
               **args)
