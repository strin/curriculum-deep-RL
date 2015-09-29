import random
import environment
import numpy as np


class Grid(environment.Environment):
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
        self.num_states = len(self.free_pos)
        self.curr_state = random.choice(self.free_pos)
        self.hit_wall = False

    def _free_pos(self):
        pos = []
        self.state_id = {}
        self.state_pos = {}
        state_num = 0
        w, h = self.grid.shape
        for i in xrange(w):
            for j in xrange(h):
                if self.grid[i][j] == 0.:
                    pos.append((i, j))
                    self.state_id[(i, j)] = state_num
                    self.state_pos[state_num] = (i, j)
                    state_num += 1
        return pos

    def get_num_states(self):
        return self.num_states

    def get_state_dimension(self):
        return 1

    def get_num_actions(self):
        return len(self.actions)

    def get_allowed_actions(self, state):
        return range(len(self.actions))

    def get_current_state(self):
        return self.state_id[self.curr_state]

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


class GridWorldMDP(environment.MDP):
    ''' Classic gridworld cast as a Markov Decision Process. In particular,
        exports the reward function and the state-transition function'''

    def __init__(self, grid, rewards, wall_penalty, gamma):
        ''' Assumes grid has already been initialized. Rewards is a map of
            (x, y)-coordinates and the reward for reaching that point'''
        self.wall_penalty = wall_penalty
        self.rewards = rewards
        self.env = grid
        self.gamma = gamma

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

        return 0

    def next_state_distribution(self, state, action):
        return self.env.next_state_distribution(self.env.state_pos[state], action)


# TODO: FIX THE STATE REPRESENTATION SO THE CONVERSION TO TABULAR HAPPENS HERE!
class GridWorld(environment.Task):
    ''' RL variant of gridworld where the dynamics and reward function are not
        fully observed

        Currently, this task is episodic. If an agent reaches one of the
        reward states, it receives the reward and is reset to a new starting
        location.
        '''
    def __init__(self, grid, rewards, wall_penalty, gamma, tabular=True):
        ''' Assumes grid has already been initialized. Rewards is a map of
            (y, x)-coordinates and the reward for reaching that point'''
        self.wall_penalty = wall_penalty
        self.rewards = rewards
        self.env = grid
        self.gamma = gamma
        self.tabular = tabular
        self.reset()

    def get_state_dimension(self):
        if self.tabular:
            return self.env.get_state_dimension()

        return self.env.grid.shape[0] * self.env.grid.shape[1]

    def get_current_state(self):
        if self.tabular:
            return self.env.get_current_state()

        agent_state = np.zeros_like(self.env.grid)
        agent_state[self.env.state_pos[self.env.get_current_state()]] = 1.0
        return agent_state.ravel().reshape(-1, 1)

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


class GridWorldWithGoals(environment.Task):
    '''A variant of the GridWorld task.

    each state consists of the state in the GridWorld task as well as a goal
    vector.
        '''
    def __init__(self, grid, goal, rewards, wall_penalty, gamma):
        ''' Assumes grid has already been initialized. Rewards is a map of
            (y, x)-coordinates and the reward for reaching that point

            goal is a 1-dimensional numpy vector.
        '''
        self.wall_penalty = wall_penalty
        self.env = grid
        self.gamma = gamma
        self.reset(goal=goal,
                   rewards=rewards)

    def get_state_dimension(self):
        return self.env.grid.reshape(-1).shape[0] + self.goal.reshape(-1).shape[0]

    def wrap_state_with_goal(self, state_resized):
        goal_resized = self.goal.ravel().reshape(-1, 1)
        return np.concatenate((state_resized, goal_resized), axis=0)

    def get_current_state(self):
        agent_state = np.zeros_like(self.env.grid)
        agent_state[self.env.state_pos[self.env.get_current_state()]] = 1.0
        state_resized = agent_state.ravel().reshape(-1, 1)
        return self.wrap_state_with_goal(state_resized)

    def perform_action(self, action):
        curr_state = self.env.get_current_state()
        next_state = self.env.perform_action(action)
        reward = self.get_reward(curr_state, action, next_state)

        agent_state = np.zeros_like(self.env.grid)
        agent_state[self.env.state_pos[next_state]] = 1.0
        next_state = agent_state.ravel().reshape(-1, 1)
        next_state_with_goal = self.wrap_state_with_goal(next_state)
        return (next_state_with_goal, reward)

    def reset(self, goal, rewards):
        self.goal = goal
        self.rewards = rewards
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
