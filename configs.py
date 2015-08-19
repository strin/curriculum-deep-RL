'''
    USAGE: Each config specifies a set of parameters to be passed in from the
           command line as follows.

           "'string': val" will become "--string val"

           To specify hyperparameters for multiple experiments simultaneously,
           specify them in a list [a, b, c].

           In this case,

                "'s1': [a, b], 's2': [b, c]"

           will become four separate experiments consisting of all possible
           permutations of the hyperparameters.

            "--s1 a --s2 b, --s1 a, --s2 c, --s1 b --s2 b, --s1 b, --s2 c"
'''

import numpy as np


def random_reals_range(low, high, num_samples):
    '''
        Helper function to randomly sample from an interval over the reals
    '''
    return np.random.rand(low=low, high=high, size=(1, num_samples)).tolist()[0]


def random_int_range(low, high, num_samples):
    '''
        Helper function to randomly sample from an interval integers
    '''
    return np.random.randint(low=low, high=high, size=(1, num_samples)).tolist()[0]


t_shaped_maze = {
    # task specific
    'maze_length': random_int_range(3, 20, 5),
    'noisy_observations': 0,  # 0 for noiseless, non-zero for noise
    'gamma': 0.98,

    # model specific
    'hidden_dimension': 128,
    'num_trajectory_samples': 10,
    'truncate_gradient': -1,  # number of steps to run BPTT (-1 means use the whole sequence)
    'max_trajectory_length': float('inf'),  # the length of longest (s, a, r) sequence used

    # experiment specific
    'max_episodes': 70000,
    'report_wait': 100,
    'save_wait': 1000,
    'experiment_samples': 50
}

hypercube = {
    # task specific
    'dimensions': (10, 10, 10),
    'action_stochasticity': 0.,
    'wall_penalty': -0.1,
    'time_penalty': -0.1,
    'reward': 4,
    'gamma': 0.9,

    # model specific (DQN)
    'hidden_dimension': 128,
    'lr': 0.05,
    'epsilon': 0.15,

    # experiment specific
    'max_episodes': 500,
    'report_wait': 50,
    'save_wait': 100,
    'fully_observed': 1,  # 0 if partially observed, non-zero otherwise
    'task_samples': 25
}
