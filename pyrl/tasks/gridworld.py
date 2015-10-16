import random
import numpy as np
import matplotlib.pyplot as plt

from pyrl.tasks.task import Environment, Task

class Grid(Environment):
    ''' Basic gridworld environment'''
    N = (0, 1)
    E = (1, 0)
    W = (-1, 0)
    S = (0, -1)

    actions = [N, E, W, S]

    curr_state = None
    hit_wall = False

    def __init__(self, grid, action_stoch):
        ''' grid is a 2D numpy array with 0 indicates where the agent
            can pass and 1 indicates an impermeable wall.

            action_stoch is the percentage of time
            environment makes a random transition
            '''
        self.grid = grid
        self.action_stoch = action_stoch
        self.free_pos = self._free_pos()
        self.num_states = grid.shape[0] * grid.shape[1]
        self.curr_state = random.choice(self.free_pos)
        self.hit_wall = False

    def _free_pos(self):
        pos = []
        self.state_id = {}
        self.state_pos = {}
        w, h = self.grid.shape
        for i in xrange(w):
            for j in xrange(h):
                if self.grid[i][j] == 0.:
                    state_num = i * h + j
                    pos.append((i, j))
                    self.state_id[(i, j)] = state_num
                    self.state_pos[state_num] = (i, j)
        return pos

    @property
    def shape(self):
        return self.grid.shape

    def support_tabular(self):
        return True

    def get_valid_states(self):
        ''' get all states without dummy ones '''
        return self.state_pos.keys()

    def get_num_states(self):
        ''' can have dummy states '''
        return self.num_states

    def get_state_dimension(self):
        return self.num_states

    def get_current_state(self):
        return self.state_id[self.curr_state]

    def get_current_state_vector(self):
        '''
        return a 0-1 vector representation of the states.
        '''
        state_vector = np.zeros(self.get_state_dimension)
        state_vector[self.get_current_state()] = 1.
        return state_vector

    def get_num_actions(self):
        return len(self.actions)

    def get_allowed_actions(self, state):
        return range(len(self.actions))

    def _move(self, state, act):
        return (state[0] + act[0], state[1] + act[1])

    def _out_of_bounds(self, state):
        return state[0] < 0 or state[0] >= self.grid.shape[0] or state[1] < 0 \
            or state[1] >= self.grid.shape[1] or self.grid[state[0], state[1]]

    def perform_action(self, action):
        if random.random() < self.action_stoch:
            tmp = self._move(self.curr_state, random.choice(self.actions))
        else:
            tmp = self._move(self.curr_state, self.actions[action])
        if self._out_of_bounds(tmp):
            self.hit_wall = True
        else:
            self.hit_wall = False
            self.curr_state = tmp

        return self.state_id[self.curr_state]

    def reset(self):
        self.hit_wall = False
        self.curr_state = random.choice(self.free_pos)

    def _next_state(self, state, action):
        ns = self._move(state, action)
        if self._out_of_bounds(ns):
            return self.state_id[state]

        return self.state_id[ns]

    def next_state_distribution(self, state, action):
        next = {}
        next[self._next_state(state, self.actions[action])] = 1. - self.action_stoch
        for act in self.actions:
            ns = self._next_state(state, act)
            if ns in next:
                next[ns] += self.action_stoch / float(len(self.actions))
            else:
                next[ns] = self.action_stoch / float(len(self.actions))

        return next.items()

    def visualize(self):
        plt.imshow(self.grid)
        plt.axis('off')

# TODO: FIX THE STATE REPRESENTATION SO THE CONVERSION TO TABULAR HAPPENS HERE!
class GridWorld(Task):
    ''' RL variant of gridworld where the dynamics and reward function are not
        fully observed

        Currently, this task is episodic. If an agent reaches one of the
        reward states, it receives the reward and is reset to a new starting
        location.
    '''
    def __init__(self, grid, gamma, rewards, wall_penalty, tabular=True):
        ''' Assumes grid has already been initialized. Rewards is a map of
            (y, x)-coordinates and the reward for reaching that point'''
        Task.__init__(self, grid, gamma)
        self.wall_penalty = wall_penalty
        self.rewards = rewards
        self.tabular = tabular
        self.reset()

    def perform_action(self, action):
        curr_state = self.env.get_current_state()
        next_state = self.env.perform_action(action)
        reward = self.get_reward(curr_state, action, next_state)

        if not self.tabular:
            agent_state = np.zeros_like(self.env.grid)
            agent_state[self.env.state_pos[next_state]] = 1.0
            next_state = agent_state.ravel().reshape(-1, 1)

        return (next_state, reward)

    def reset(self):
        while(self.env.state_pos[self.env.get_current_state()] in self.rewards):
            self.env.reset()

    def get_allowed_actions(self, state):
        if (self.env.state_pos[state] in self.rewards):
            return []
        else:
            return self.env.get_allowed_actions(state)

    def get_reward(self, state, action, next_state):
        # returns the reward based on the (s, a, s') triple
        if (state == next_state):
            return self.wall_penalty

        if (self.env.state_pos[next_state] in self.rewards):
            return self.rewards[self.env.state_pos[next_state]]

        return 0.  # no reward

class GridWorldUltimate(Task):
    '''
        the most complete variant of GridWorld, which includes:
            agent, goals, grid, demons.
        the state representation is concatenation of all four.
    '''
    def __init__(self, grid_env, goal, demons, gamma, rewards, wall_penalty):
        ''' Assumes grid has already been initialized. Rewards is a map of
            (y, x)-coordinates and the reward for reaching that point

            Parameters
            =========
            grid (numpy.ndarray) - the 2D world representation
            goal (dict {(y,x)->1, ...}) - which cells of the grid the agent tries to reach (beans in pacman).
            demons (dict {(y,x)->1, ...} - which cells of the grid demons currenty are at.

            (TODO: demons breaks abstraction of env)
        '''
        Task.__init__(self, grid_env, gamma)
        self.wall_penalty = wall_penalty
        self.goal = goal
        self.demons = demons
        self.rewards = rewards
        # state representation.
        self.goal_2d = np.zeros_like(self.env.grid)
        for pos in self.goal:
            self.goal_2d[pos] = 1.
        self.goal_resized = self.goal_2d.reshape(-1, 1)
        self.demons_2d = np.zeros_like(self.env.grid)
        for pos in self.demons:
            self.demons_2d[pos] = 1.
        self.demons_resized = self.demons_2d.reshape(-1, 1)
        self.grid_resized = self.env.grid.reshape(-1, 1)
        self.reset()

    def get_valid_states(self):
        return [key for key in self.env.get_valid_states() if self.env.state_pos[key] not in self.rewards]

    def get_num_states(self):
        return self.env.get_num_states()

    def get_state_dimension(self):
        return self.env.grid.reshape(-1).shape[0] * 2 # grid, agent, demon, goal.

    def get_current_state(self):
        return self.env.get_current_state()

    def get_current_state_vector(self):
        return self.wrap_stateid(self.get_current_state)

    @staticmethod
    def create_from_state(state, H, W, action_stoch=0.2, wall_penalty=0., gamma=0.9):
        '''
        given state vector representation, create a GridWorldUltimate object.
        '''
        world = state[H * W : 2 * H * W].reshape(H, W)
        grid = Grid(world, action_stoch=action_stoch)
        def extract_pos_map(mat):
            non_zeros = zip(*map(list, mat.nonzero()))
            return {pos: 1 for pos in non_zeros}
        goal = extract_pos_map(state[2 * H * W : 3 * H * W].reshape(H, W))
        rewards = dict(goal)
        demons = extract_pos_map(state[3 * H * W : 4 * H * W].reshape(H, W))
        return GridWorldUltimate(grid, goal, demons, rewards, wall_penalty=wall_penalty, gamma=gamma)

    def wrap_stateid(self, stateid):
        '''
        given a state_id, return a column vector representation.
        '''
        agent_state = np.zeros_like(self.env.grid)
        agent_state[self.env.state_pos[stateid]] = 1.0
        return self.wrap_state(agent_state)

    def wrap_state(self, state):
        '''
        generate a state column vector that represents:
        [agent, grid, goal, demon]
        '''
        state_resized = state.reshape(-1, 1)
        return np.concatenate((state_resized,
            self.goal_resized), axis=0)

    def get_state_vector(self, state):
        '''
        get state embedding (without goals, demons, etc.).
        '''
        state_mat = np.zeros_like(self.env.grid)
        state_mat[self.env.state_pos[state]] = 1.0
        state_resized = state_mat.reshape(-1)
        return state_resized

    def get_goal_vector(self):
        return self.goal_resized.reshape(-1)

    def perform_action(self, action):
        curr_state = self.env.get_current_state()
        next_state = self.env.perform_action(action)
        reward = self.get_reward(curr_state, action, next_state)

        return (next_state, reward)

    def reset(self):
        while(self.env.state_pos[self.env.get_current_state()] in self.goal):
            self.env.reset()

    def get_allowed_actions(self, state):
        if (self.env.state_pos[state] in self.rewards):
            return []
        else:
            return self.env.get_allowed_actions(state)

    def get_num_actions(self):
        return self.env.get_num_actions()

    def get_reward(self, state, action, next_state):
        # returns the reward based on the (s, a, s') triple
        if (state == next_state):
            return self.wall_penalty

        if (self.env.state_pos[next_state] in self.rewards):
            return self.rewards[self.env.state_pos[next_state]]

        return 0.  # no reward

    def next_state_distribution(self, state, action):
        return self.env.next_state_distribution(self.env.state_pos[state], action)

    def show_V(v, H, W):
        plt.imshow(v.reshape(H, W), interpolation='none')
        plt.axis('off')
        print v

def generate_gridworlds(world, action_stoch=0.2, wall_penalty=0., gamma=0.9):
    grid = Grid(world, action_stoch=action_stoch)
    gridworlds = []
    for pos in grid.free_pos:
        goal = {pos: 1.}
        rewards = dict(goal)
        demons = {}
        gridworlds.append(
            GridWorldUltimate(grid, goal, demons, rewards=rewards, wall_penalty=wall_penalty, gamma=gamma)
        )
    return gridworlds

