import numpy as np
import numpy.random as npr
import time
from os import path
from pyrl.tasks.task import Task
from pyrl.algorithms.valueiter import compute_tabular_value
from pyrl.utils import mkdir_if_not_exist
from matplotlib import pyplot

def record_game(policy, task, output_path, max_step=9999, **policy_args):
    '''
    record the play of policy on task, and save each frame to *output_path*.

    Requirements:
        task should have visualize(self, fname=None) implemented.
    '''
    step = 0
    mkdir_if_not_exist(output_path)
    task.reset()

    while step < max_step:
        task.visualize(path.join(output_path, '%d' % step))
        curr_state = task.curr_state
        action = policy.get_action(curr_state, valid_action=task.valid_actions, **policy_args)
        task.step(action)
        step += 1

    task.reset()


def record_game_multi(policy, task, output_path, num_times=5, max_step=9999, **policy_args):
    for ni in range(num_times):
        record_game(policy, task, path.join(output_path, '%d' % ni), max_step, **policy_args)


def replay_game(output_path):
    '''
    replay the frames of game play saved in *output_path*.
    '''
    step = 0
    get_path = lambda step: path.join(output_path, '%d' % step)
    while path.exists(get_path(step)):
        image = pyplot.imread(get_path(step))
        pyplot.imshow(image)
        time.sleep(0.1)


