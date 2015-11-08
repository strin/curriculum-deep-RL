from pyrl.tasks.task import Task

import random
import numpy as np
import matplotlib.pyplot as plt

class GridWorld(Task):
    ''' RL variant of gridworld where the dynamics and reward function are not
        fully observed

        Currently, this task is episodic. If an agent reaches one of the
        reward states, it receives the reward and is reset to a new starting
        location.
    '''
    N = (0, 1)
    E = (1, 0)
    W = (-1, 0)
    S = (0, -1)

    actions = [N, E, W, S]

    def __init__(self, grid, action_stoch, goal, rewards, wall_penalty, gamma):
        ''' grid is a 2D numpy array with 0 indicates where the agent
            can pass and 1 indicates an impermeable wall.

            action_stoch is the percentage of time
            environment makes a random transition
            '''
        self.grid = grid
        self.action_stoch = action_stoch
        self.goal = goal
        self.free_pos = self._free_pos()
        self.curr_pos = random.choice(self.free_pos)
        self.hit_wall = False

        self.wall_penalty = wall_penalty
        self.rewards = rewards
        self.env = grid
        self.gamma = gamma

        # state representation.
        (h, w) = self.grid.shape
        self.state_2d = np.zeros((2 * h, w))
        self.state_2d[self.curr_pos] = 1.
        for pos in self.goal:
            self.state_2d[h + pos[0], pos[1]] = 1.

        # start the game fresh.
        self.reset()

    def _free_pos(self):
        pos = []
        self.state_id = {}
        self.state_pos = {}
        state_num = 0
        w, h = self.grid.shape
        for i in xrange(w):
            for j in xrange(h):
                if self.grid[i][j] == 0. and (i, j) not in self.goal:
                    pos.append((i, j))
                    self.state_id[(i, j)] = state_num
                    self.state_pos[state_num] = (i, j)
                    state_num += 1
        return pos

    def reset(self):
        self.hit_wall = False
        self.state_2d[self.curr_pos] = 0.
        self.curr_pos = random.choice(self.free_pos)
        self.state_2d[self.curr_pos] = 1.

    @property
    def curr_state(self):
        '''
        state is a 1x2HxW tensor.
        '''
        return np.array([self.state_2d])

    @property
    def num_states(self):
        w, h = self.grid.shape
        return w * h

    @property
    def state_shape(self):
        return self.state_2d.shape

    @property
    def shape(self):
        return self.grid.shape

    @property
    def num_actions(self):
        return len(self.actions)

    @property
    def valid_actions(self):
        return range(len(self.actions))

    def _move(self, state, act):
        return (state[0] + act[0], state[1] + act[1])

    def _out_of_bounds(self, state):
        return state[0] < 0 or state[0] >= self.grid.shape[0] or state[1] < 0 \
            or state[1] >= self.grid.shape[1] or self.grid[state[0], state[1]]

    def step(self, action):
        # update state_2d matrix
        self.state_2d[self.curr_pos] = 0.

        # compute new coordinate.
        if random.random() < self.action_stoch:
            tmp = self._move(self.curr_pos, random.choice(self.actions))
        else:
            tmp = self._move(self.curr_pos, self.actions[action])
        if self._out_of_bounds(tmp):
            self.hit_wall = True
        else:
            self.hit_wall = False
            self.curr_pos = tmp

        # update state_2d matrix.
        self.state_2d[self.curr_pos] = 1.

        # return reward.
        if self.curr_pos not in self.rewards:
            return 0.
        else:
            return float(self.rewards[self.curr_pos])

    def is_end(self):
        return self.curr_pos in self.goal

    def visualize(self):
        plt.imshow(self.grid)
        plt.axis('off')

class GridWorldFixedStart(GridWorld):
    def __init__(self, start_pos, grid, action_stoch, goal, rewards, wall_penalty, gamma):
        self.start_pos = start_pos
        GridWorld.__init__(self, grid, action_stoch, goal, rewards, wall_penalty, gamma)
        assert(start_pos in self.free_pos)

    def reset(self):
        self.hit_wall = False
        self.state_2d[self.curr_pos] = 0.
        self.curr_pos = self.start_pos
        self.state_2d[self.curr_pos] = 1.

