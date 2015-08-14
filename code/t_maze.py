import random
import environment
import numpy as np
import matplotlib


class TMaze(environment.Environment):

    left = np.asarray([[0], [1], [1]])
    right = np.asarray([[1], [1], [0]])
    corridor = np.asarray([[1], [0], [1]])
    junction = np.asarray([[0], [1], [0]])

    actions = ['N', 'E', 'W', 'S']

    def __init__(self, length, noise=False):
        self.length = length
        self.noise = noise
        self.reset()

    def get_state_dimension(self):
        return 3

    def get_num_actions(self):
        return 4

    def get_allowed_actions(self, state):
        if state is None:
            return []
        else:
            return range(len(self.actions))

    def get_current_state(self):
        if self.current_state in ['left', 'right']:
            return None

        if self.current_state < self.length:
            corridor = self.corridor
            if self.noise:
                corridor[0] = np.random.rand()
                corridor[2] = np.random.rand()
            return corridor

        return self.junction

    def get_start_state(self):
        if self.goal is 'left':
            return self.left
        else:
            return self.right

    def perform_action(self, action):
        act = self.actions[action]

        next_state = 0
        if self.current_state < self.length:
            if act in ['E', 'W']:
                next_state = self.current_state
            elif act == 'N':
                next_state = self.current_state + 1
            else:
                next_state = self.current_state - 1

        if self.current_state == self.length:
            if act == 'N':
                next_state = self.current_state
            elif act == 'S':
                next_state = self.current_state - 1
            elif act == 'E':
                next_state = 'right'
            else:
                next_state = 'left'

        if next_state >= 0:
            self.current_state = next_state

        return self.get_current_state()  # returns the correct state representation

    def reset(self):
        self.current_state = 0
        self.goal = random.choice(['left', 'right'])


class TMazeTask(environment.Task):

    def __init__(self, tmaze, gamma=0.98):
        self.env = tmaze
        self.gamma = gamma

    def get_start_state(self):
        return self.env.get_start_state()

    def reset(self):
        self.env.reset()

    def perform_action(self, action):
        start = self.env.current_state
        next_state = self.env.perform_action(action)

        reward = 0.
        if next_state is None:  # reached a terminal state
            if self.env.current_state == self.env.goal:
                reward = 4  # agent took the correct action!
            else:
                reward = -0.1

        elif start == self.env.current_state:
            reward = -0.1  # agent stood still

        return (next_state, reward)

    def visualize(self):
        '''
            Visualize the current game board.
        '''
        self.cmap = matplotlib.colors.ListedColormap(['black', 'grey', 'blue', 'green', 'red'])
        self.color_norm = matplotlib.colors.BoundaryNorm(range(6), 5)

        # construct the game board
        world = np.zeros((3, self.env.length + 1))
        world[0, :-1] = 0
        world[2, :-1] = 0

        world[1, :-1] = 1
        world[:, -1] = 1

         # show the position of the agent
        curr_pos = self.env.current_state
        if curr_pos == 'left':
            if self.env.goal == 'left':
                world[0, -1] = 4
            else:
                world[0, -1] = 2
                world[2, -1] = 3
        elif curr_pos == 'right':
            if self.env.goal == 'right':
                world[2, -1] = 4
            else:
                world[2, -1] = 2
                world[0, -1] = 3
        else:
            world[1, curr_pos] = 2
            if self.env.goal == 'left':
                world[0, -1] = 3
            else:
                world[2, -1] = 3

        return world
