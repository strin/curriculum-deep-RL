# algorithms from UVFA: universal function value approximation.
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T

from pyrl.algorithms.valueiter import ValueIterationSolver
from pyrl.agents.agent import Qfunc
import pyrl.optimizers as optimizers
import pyrl.layers as layers
import pyrl.prob as prob
from pyrl.utils import Timer
from pyrl.tasks.task import Task

from pyrl.layers import SoftMax

def factorize_value_matrix(valmat, rank_n = 3, num_iter = 10000):
    from optspace import optspace
    # convert to sparse matrix.
    smat = []
    for i in range(valmat.shape[0]):
        for j in range(valmat.shape[1]):
            smat.append((i, j, valmat[i, j]))

    (X, S, Y) = optspace(smat, rank_n = rank_n,
        num_iter = num_iter,
        tol = 1e-4,
        verbosity = 0,
        outfile = ""
    )

    [X, S, Y] = map(np.matrix, [X, S, Y])

    mse = np.sqrt(np.sum(np.power(X * S * Y.T - valmat, 2)) / X.shape[0] / Y.shape[0])
    return (X, S, Y, mse)


class TwoStreamQfunc(object):
    '''
    univeral value Q functions.
    '''
    def __init__(self, env, arch_state, arch_goal, S):
        self.env = env
        self.arch_state = arch_state
        self.arch_goal = arch_goal
        self.S = S

    def _initialize_net(self):
        '''
        Initialize the deep Q neural network.
        '''
        # construct computation graph for forward pass
        self.states = T.matrix('states')
        self.feat_states, model = self.arch_states(self.states)
        self.params = []

        self.params.extend(sum([layer.params for layer in model], []))
        self.fprop_states = theano.function(inputs=[self.states],
                                     outputs=self.feat_states,
                                     name='fprop_states')

        self.goals = T.matrix('goals')
        self.feat_goals, model = self.arch_goal(self.states)
        self.params.extend(sum([layer.params for layer in model], []))
        self.fprop_goals = theano.function(inputs=[self.goals],
                                     outputs=self.feat_goals,
                                     name='fprop_goals')

    def __call__(self, states, actions, goals):
        feat_states = self.fprop_states(states)
        feat_goals = self.fprop(goals)
        assert feat_states.shape[0] == feat_goals.shape[0]
        vals = np.zeros(feat_states.shape[0])
        for ni in range(feat_states.shape[0]):
            vals[ni] = np.dot(
                np.dot(feat_states[ni, :].reshape(1, -1),
                       self.S),
                feat_goals[ni, :].reshape(-1, 1)
            )
        return vals


class Horde(object):
    '''
    horde architecture.
    '''
    def __init__(self, dqn_by_goal, gamma=0.95, l2_reg=0.0, lr=1e-3,
                 experiences=[], minibatch_size=64):
        self.dqn_by_goal = dqn_by_goal
        self.l2_reg = l2_reg
        self.lr = lr
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.experiences = experiences
        self.bprop_by_goal = {}
        self._compile_bp()


    def _compile_bp(self):
        '''
        compile backpropagation foreach of the dqns.
        '''
        self.bprop_by_goal = {}
        for (goal, dqn) in self.dqn_by_goal.items():
            states = dqn.states
            action_values = dqn.action_values
            params = dqn.params
            targets = T.vector('target')
            last_actions = T.lvector('action')

            # loss function.
            mse = layers.MSE(action_values[T.arange(action_values.shape[0]),
                                last_actions], targets)
            # l2 penalty.
            l2_penalty = 0.
            for param in params:
                l2_penalty += (param ** 2).sum()

            cost = mse + self.l2_reg * l2_penalty

            # back propagation.
            updates = optimizers.Adam(cost, params, alpha=self.lr)

            td_errors = T.sqrt(mse)
            self.bprop_by_goal[goal] = theano.function(inputs=[states, last_actions, targets],
                                        outputs=td_errors, updates=updates)


    def learn(self, num_iter=10, print_lag=50):
        for it in range(num_iter):
            for (goal, dqn) in self.dqn_by_goal.items():
                bprop = self.bprop_by_goal[goal]
                samples = prob.choice(self.experiences,
                                      self.minibatch_size, replace=True) # draw with replacement.

                # sample a minibatch.
                states = [None] * self.minibatch_size
                next_states = [None] * self.minibatch_size
                actions = np.zeros(self.minibatch_size, dtype=int)
                rewards = np.zeros(self.minibatch_size)
                nvas = []
                terminals = []

                for idx, sample in enumerate(samples):
                    state, action, next_state, reward, meta = sample
                    nva = meta['curr_valid_actions']

                    states[idx] = np.array(state['raw_state'])
                    states[idx][1, goal[0], goal[1]] = 1.
                    actions[idx] = action
                    reward = next_state['pos'][goal[0], goal[1]] # TODO: hack for gridworld.
                    rewards[idx] = reward
                    nvas.append(nva)

                    next_states[idx] = np.array(next_state['raw_state'])
                    next_states[idx][1, goal[0], goal[1]] = 1.
                    if reward > 0.:
                        terminals.append(idx)

                states = np.array(states)
                next_states = np.array(next_states)

                # learn through backpropagation.
                next_qvals = dqn.fprop(next_states)
                next_vs = np.zeros(self.minibatch_size)
                for idx in range(self.minibatch_size):
                    if idx not in terminals:
                        next_vs[idx] = np.max(next_qvals[idx, nvas[idx]])


                targets = rewards + self.gamma * next_vs


                error = bprop(states, actions, targets.flatten())

            if print_lag and print_lag > 0 and it % print_lag == 0:
                print 'iter = ', it, 'error = ', error


class AntiHorde(object):
    def __init__(self, dqn, goals, gamma=0.95, l2_reg=0.0, lr=1e-3,
                 experiences=[], minibatch_size=64):
        self.dqn = dqn
        self.goals = goals
        self.l2_reg = l2_reg
        self.lr = lr
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.experiences = experiences
        self.bprop_by_goal = {}
        self._compile_bp()

    def _compile_bp(self):
        dqn = self.dqn
        states = dqn.states
        action_values = dqn.action_values
        params = dqn.params
        targets = T.vector('target')
        last_actions = T.lvector('action')

        # loss function.
        mse = layers.MSE(action_values[T.arange(action_values.shape[0]),
                            last_actions], targets)
        # l2 penalty.
        l2_penalty = 0.
        for param in params:
            l2_penalty += (param ** 2).sum()

        cost = mse + self.l2_reg * l2_penalty

        # back propagation.
        updates = optimizers.Adam(cost, params, alpha=self.lr)

        td_errors = T.sqrt(mse)
        self.bprop = theano.function(inputs=[states, last_actions, targets],
                            outputs=td_errors, updates=updates)


    def learn(self, num_iter=10, print_lag = 50):
        for it in range(num_iter):
            for goal in self.goals:
                dqn = self.dqn
                bprop = self.bprop
                samples = prob.choice(self.experiences,
                                        self.minibatch_size, replace=True) # draw with replacement.

                # sample a minibatch.
                states = [None] * self.minibatch_size
                next_states = [None] * self.minibatch_size
                actions = np.zeros(self.minibatch_size, dtype=int)
                rewards = np.zeros(self.minibatch_size)
                nvas = []
                terminals = []

                for idx, sample in enumerate(samples):
                    state, action, next_state, reward, meta = sample
                    nva = meta['curr_valid_actions']

                    states[idx] = np.array(state['raw_state'])
                    states[idx][1, goal[0], goal[1]] = 1.
                    actions[idx] = action
                    reward = next_state['pos'][goal[0], goal[1]] # TODO: hack for gridworld.
                    rewards[idx] = reward
                    nvas.append(nva)

                    next_states[idx] = np.array(next_state['raw_state'])
                    next_states[idx][1, goal[0], goal[1]] = 1.
                    if reward > 0.:
                        terminals.append(idx)

                states = np.array(states)
                next_states = np.array(next_states)

                # learn through backpropagation.
                next_qvals = dqn.fprop(next_states)
                next_vs = np.zeros(self.minibatch_size)
                for idx in range(self.minibatch_size):
                    if idx not in terminals:
                        next_vs[idx] = np.max(next_qvals[idx, nvas[idx]])


                targets = rewards + self.gamma * next_vs


                error = bprop(states, actions, targets.flatten())

            if print_lag and print_lag > 0 and it % print_lag == 0:
                print 'iter = ', it, 'error = ', error


class PolicyDistill(object):
    def __init__(self, dqn_mt, dqn_by_goal, experiences, lr=1e-3, l2_reg=0., minibatch_size=128, loss='KL'):
        '''
        '''
        self.l2_reg = l2_reg
        self.minibatch_size = minibatch_size
        self.lr = lr
        self.loss = loss
        assert(self.loss in ['KL', 'l2', 'l1', 'l1-action', 'l1-exp'])

        # experience is a dict: task -> experience buffer.
        # an experience buffer is a list of tuples (s, q, va)
        # s = state, q = list of normalized qvals, va = corresponding valid actions.
        self.experiences = experiences
        self.dqn_mt = dqn_mt
        self.dqn_by_goal = dqn_by_goal
        self.goals = self.dqn_by_goal.keys()


        self._compile_bp()


    def _compile_bp(self):
        states = self.dqn_mt.states
        params = self.dqn_mt.params

        targets = T.matrix('target')
        is_valid = T.matrix('is_valid')
        last_actions = T.lvector('action')

        # compute softmax for action_values
        # numerical stability in mind.
        action_values = self.dqn_mt.action_values

        # loss function: KL-divergence.
        if self.loss == 'KL':
            action_values -= (1 - is_valid) * 1e10
            action_values_softmax = SoftMax(action_values)
            action_values_softmax += (1 - is_valid)
            loss = T.sum(targets * is_valid * (T.log(targets) - T.log(action_values_softmax)))
        elif self.loss == 'l2':
            loss = T.sum((targets - action_values) ** 2 * is_valid)
        elif self.loss == 'l1' or self.loss == 'l1-action':
            loss = T.sum(abs(targets.flatten() - action_values[T.arange(action_values.shape[0]),
                                last_actions])  * is_valid.flatten())
        elif self.loss == 'l1-exp':
            temperature = 1.
            qt = targets / temperature
            qt_max = T.max(action_values, axis=1).dimshuffle(0, 'x')
            qt_sub = qt - qt_max
            b = T.log(T.exp(qt_sub).sum(axis=1)).dimshuffle(0, 'x') + qt_max
            qs = action_values / temperature
            loss = T.sum(abs(T.exp(qs - b) - T.exp(qt - b)))

        # l2 penalty.
        l2_penalty = 0.
        for param in params:
            l2_penalty += (param ** 2).sum()

        cost = loss + self.l2_reg * l2_penalty

        updates = optimizers.Adam(cost, params, alpha=self.lr)
        self.bprop = theano.function(inputs=[states, last_actions, targets, is_valid],
                                     outputs=loss / states.shape[0], updates=updates,
                                     on_unused_input='warn')


    def learn(self, num_iter=100, temperature=1., print_lag=None):
        for it in range(num_iter):
            dqn = self.dqn_mt
            bprop = self.bprop
            samples = prob.choice(self.experiences,
                                    self.minibatch_size, replace=True) # draw with replacement.

            # sample a minibatch.
            is_valids = []
            targets = []
            states = []
            actions = np.zeros(self.minibatch_size, dtype=int)

            for idx, sample in enumerate(samples):
                # randomly choose a goal.
                goal = prob.choice(self.goals, 1)[0]
                dqn = self.dqn_by_goal[goal]

                state, last_action, next_state, reward, meta = sample
                valid_actions = meta['last_valid_actions']
                num_actions = meta['num_actions']
                raw_state = np.array(state['raw_state'])
                raw_state[1, goal[0], goal[1]] = 1.

                states.append(raw_state)

                is_valid = [1. for action in range(num_actions) if action in set(valid_actions)]

                if self.loss == 'KL':
                    target = dqn._get_softmax_action_distribution(raw_state, temperature=temperature, valid_actions=valid_actions)
                elif self.loss == 'l2' or self.loss == 'l1' or self.loss == 'l1-exp':
                    target = dqn.av(raw_state)
                elif self.loss == 'l1-action':
                    target = [dqn.av(raw_state)[last_action]]
                    is_valid  = [is_valid[last_action]]

                is_valids.append(is_valid)
                targets.append(target)
                actions[idx] = last_action

            states = np.array(states)
            targets = np.array(targets)
            is_valids = np.array(is_valids)

            score = self.bprop(states, actions, targets, is_valids)

            if print_lag and print_lag > 0 and it % print_lag == 0:
                print 'iter = ', it, 'score = ', score



class HordeShared(object):
    '''
    horde architecture.
    '''
    def __init__(self, dqn_by_goal, gamma=0.95, l2_reg=0.0, lr=1e-3,
                 experiences=[], minibatch_size=64):
        self.dqn_by_goal = dqn_by_goal
        self.l2_reg = l2_reg
        self.lr = lr
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.experiences = experiences
        self.bprop_by_goal = {}
        self._compile_bp()


    def _compile_bp(self):
        '''
        compile backpropagation foreach of the dqns.
        '''
        self.bprop_by_goal = {}
        for (goal, dqn) in self.dqn_by_goal.items():
            states = dqn.states
            action_values = dqn.action_values
            params = dqn.params
            targets = T.vector('target')
            shared_values = T.vector('shared_values')
            last_actions = T.lvector('action')

            # loss function.
            mse = layers.MSE(action_values[T.arange(action_values.shape[0]),
                                last_actions], targets) \
                    + T.mean(abs(action_values[T.arange(action_values.shape[0]),
                                    last_actions] - shared_values))
            # l2 penalty.
            l2_penalty = 0.
            for param in params:
                l2_penalty += (param ** 2).sum()

            cost = mse + self.l2_reg * l2_penalty

            # back propagation.
            updates = optimizers.Adam(cost, params, alpha=self.lr)

            td_errors = T.sqrt(mse)
            self.bprop_by_goal[goal] = theano.function(inputs=[states, last_actions, targets, shared_values],
                                        outputs=td_errors, updates=updates)


    def learn(self, num_iter=10, print_lag=50, dqn_mt=None):
        for it in range(num_iter):
            for (goal, dqn) in self.dqn_by_goal.items():
                bprop = self.bprop_by_goal[goal]
                samples = prob.choice(self.experiences,
                                      self.minibatch_size, replace=True) # draw with replacement.

                # sample a minibatch.
                states = [None] * self.minibatch_size
                next_states = [None] * self.minibatch_size
                actions = np.zeros(self.minibatch_size, dtype=int)
                rewards = np.zeros(self.minibatch_size)
                nvas = []
                terminals = []

                for idx, sample in enumerate(samples):
                    state, action, next_state, reward, meta = sample
                    nva = meta['curr_valid_actions']

                    states[idx] = np.array(state['raw_state'])
                    states[idx][1, goal[0], goal[1]] = 1.
                    actions[idx] = action
                    reward = next_state['pos'][goal[0], goal[1]] # TODO: hack for gridworld.
                    rewards[idx] = reward
                    nvas.append(nva)

                    next_states[idx] = np.array(next_state['raw_state'])
                    next_states[idx][1, goal[0], goal[1]] = 1.
                    if reward > 0.:
                        terminals.append(idx)

                states = np.array(states)
                next_states = np.array(next_states)

                # learn through backpropagation.
                shared_values = dqn_mt.fprop(next_states)[range(len(actions)), actions]
                next_qvals = dqn.fprop(next_states)
                next_vs = np.zeros(self.minibatch_size)
                for idx in range(self.minibatch_size):
                    if idx not in terminals:
                        next_vs[idx] = np.max(next_qvals[idx, nvas[idx]])


                targets = rewards + self.gamma * next_vs


                error = bprop(states, actions, targets.flatten(), shared_values)

            if print_lag and print_lag > 0 and it % print_lag == 0:
                print 'iter = ', it, 'error = ', error

