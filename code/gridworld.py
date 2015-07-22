import random
import environment


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
        state_num = 0
        w, h = self.grid.shape
        for i in xrange(w):
            for j in xrange(h):
                if self.grid[i][j] == 0.:
                    pos.append((i, j))
                    self.state_id[(i, j)] = state_num
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

    def get_starting_state(self):
        return self.state_id[self.curr_state]

    def _move(self, act):
        return (self.curr_state[0] + act[0], self.curr_state[1] + act[1])

    def perform_action(self, action):
        if random.random() < self.action_stochasticity:
            tmp = self._move(random.choice(self.actions))
        else:
            tmp = self._move(self.actions[action])
        if tmp[0] < 0 or tmp[0] >= self.grid.shape[0] or tmp[1] < 0 \
                or tmp[1] >= self.grid.shape[1]:
            self.hit_wall = True
        elif self.grid[tmp[0], tmp[1]]:
            self.hit_wall = True
        else:
            self.hit_wall = False
            self.curr_state = tmp

        return self.state_id[self.curr_state]

    def reset(self):
        self.hit_wall = False
        self.curr_state = random.choice(self.free_pos)


class GridWorldMDP(environment.MDP):
    ''' Classic gridworld cast as a Markov Decision Process. In particular,
        exports the reward function and the state-transition function'''
    # TODO: IMPLEMENT ME


class GridWorld(environment.Task):
    ''' RL variant of gridworld where the dynamics and reward function are not
        fully observed'''
    # TODO: IMPLEMENT ME
