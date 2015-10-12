# algorithms from UVFA: universal function value approximation.
from multiprocessing import Pool
import numpy as np

from pyrl.algorithms.valueiter import ValueIterationSolver

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

