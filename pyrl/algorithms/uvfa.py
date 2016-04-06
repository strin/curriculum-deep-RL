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


    def learn(self, num_iter=10):
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

                terminals = []
                for idx, sample in enumerate(samples):
                    state, action, next_state, reward, meta = sample
                    nva = meta['curr_valid_actions']

                    states[idx] = state
                    actions[idx] = action
                    rewards[idx] = reward
                    nvas.append(nva)

                    if next_state is not None:
                        next_states[idx] = next_state
                    else:
                        next_states[idx] = state
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







