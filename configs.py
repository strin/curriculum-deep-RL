# general parameters
t_shaped_maze = {
    # task specific
    'maze_length': 5,
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
