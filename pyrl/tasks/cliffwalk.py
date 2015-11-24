from pyrl.tasks.task import Task

import random
import numpy as np
import matplotlib.pyplot as plt

class CliffWalk(Task):
    actions = [0, 1]

    def __init__(self, start_pos, size):
        self.state_1d = np.zeros(size)
        self.curr_pos = 0
        self.start_pos = start_pos
        self.size = size
        self.dead = False
        self.reset()

    def reset(self):
        self.state_1d[self.curr_pos] = 0.
        self.curr_pos = self.start_pos
        self.state_1d[self.curr_pos] = 1.
        self.dead = False

    @property
    def curr_state(self):
        return np.array(self.state_1d)

    @property
    def num_states(self):
        return self.size

    @property
    def state_shape(self):
        return self.state_1d.shape

    @property
    def num_actions(self):
        return len(self.actions)

    @property
    def valid_actions(self):
        return self.actions

    def step(self, action):
        if action == 0:
            self.dead = True
            return 0.
        self.state_1d[self.curr_pos] = 0.
        self.curr_pos += 1
        self.state_1d[self.curr_pos] = 1.
        if self.curr_pos == self.size - 1:
            return 1.
        return 0.

    def is_end(self):
        return self.dead or self.curr_pos == self.size - 1

    def __repr__(self):
        return '(' + str(int(self.curr_pos)) + '/' + str(self.size) + ')'


