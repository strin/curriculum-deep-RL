# algorithms from UVFA: universal function value approximation.
from multiprocessing import Pool
import numpy as np
import theano
import theano.tensor as T

from pyrl.algorithms.valueiter import ValueIterationSolver
from pyrl.agents.agent import Qfunc

def compute_tabular_value(task):
    solver = ValueIterationSolver(task)
    solver.learn()
    return solver.vfunc.V

def eval_tabular_value(task, func):
    V = np.zeros(task.get_num_states())
    for state in range(task.get_num_states()):
        V[state] = func(state)
    return V

def compute_tabular_values(tasks, num_cores = 8):
    ''' take a list of tabular tasks, and return states x tasks value matrix.
    '''
    pool = Pool(num_cores)
    vals = pool.map(compute_tabular_value, tasks)
    return np.transpose(np.array(vals))

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





