import environment
import numpy as np
import itertools
import matplotlib
import random
import util
from experiment import Observer, Controller


class HyperCubeMaze(environment.Environment):

    def __init__(self, dimensions=(5, 5, 5), action_stoch=0., grid=None):
        self.dimensions = dimensions
        self.action_stoch = action_stoch
        if grid is None:  # didn't specify a world map, so create one without walls
            grid = np.zeros(dimensions)
        self.grid = grid
        self.reset()

    def get_state_dimension(self):
        # just the state required to represent position in the
        # maze independent the goal vectors
        return len(self.dimensions)

    def get_num_actions(self):
        return len(self.dimensions) * 2

    def get_allowed_actions(self, state):
        '''
            No notion of terminal actions at the environment level.
        '''
        return range(self.get_num_actions())

    def get_current_state(self):
        return self.curr_state

    def perform_action(self, action):
        # actions {2*dim, 2*dim + 1} correspond to +1 and -1 along dim,
        # respectively
        if np.random.rand() < self.action_stoch:
            action = np.random.randint(0, self.get_num_actions())

        dim = action / 2
        act = 1
        if action % 2 == 1:
            act = -1

        next_state = list(self.curr_state)
        next_state[dim] += act
        next_state = tuple(next_state)
        if next_state[dim] >= 0 and next_state[dim] < self.dimensions[dim] \
                and self.grid[next_state] == 0.:
            self.curr_state = tuple(next_state)

        return self.curr_state

    def reset(self):
        # choose the starting position along each coordinate randomly
        # note that each direction can be a different length
        state = []
        while True:
            for dim in xrange(len(self.dimensions)):
                state.append(np.random.randint(0, self.dimensions[dim]))

            state = tuple(state)
            if self.grid[state] == 0.:  # don't start out on top of a wall
                break

        self.curr_state = state


class HyperCubeMazeTask(environment.Task):

    def __init__(self, hypercubemaze, initial_goal, wall_penalty=-0.1, time_penalty=0.,
                 reward=4., gamma=0.9, fully_observed=False, maximum_steps=None):
        self.env = hypercubemaze
        self.gamma = gamma
        self.fully_observed = fully_observed
        self.maximum_steps = maximum_steps

        self.num_steps = 0

        # keeps track of goal. Set via SET_GOALS
        self.set_goals(initial_goal)

        self.wall_penalty = wall_penalty  # reward for hitting a wall
        self.time_penalty = time_penalty  # reward earned on steps without action
        self.reward = reward  # reward earned for reaching a goal (assume all goals are the same)

    def set_goals(self, goal_vec):
        goal_vec = goal_vec.reshape(-1, 1)  # always a column vector
        assert(2 ** len(self.env.dimensions) == len(goal_vec))
        self.goal_vec = goal_vec
        self.goals = self._get_goals()
        self.remaining_vec = np.copy(goal_vec)  # goals we've visited

    def _get_goals(self):
        maximums = [max_dim - 1 for max_dim in self.env.dimensions]
        possible_goals = list(itertools.product(*zip([0] * len(self.env.dimensions),
                                                maximums)))
        goals = {}
        for idx in xrange(len(self.goal_vec)):
            if self.goal_vec[idx]:
                goals[possible_goals[idx]] = idx

        return goals

    def get_state_dimension(self):
        if self.goal_vec is None:
            print 'Must set a goal before initializing task'
            assert False
        if not self.fully_observed:
            return self.env.get_state_dimension() + len(self.goal_vec)
        else:
            return self.env.get_state_dimension() + 2 * len(self.goal_vec)

    def _get_state_vector(self, state):
        location = np.asarray(state).reshape(-1, 1)
        if not self.fully_observed:
            return np.concatenate([location, self.goal_vec])
        else:
            return np.concatenate([location, self.goal_vec, self.remaining_vec])

    def get_start_state(self):
        return self._get_state_vector(self.env.get_start_state())

    def get_current_state(self):
        return self._get_state_vector(self.env.get_current_state())

    def reset(self):
        self.env.reset()
        self.num_steps = 0
        self.remaining_vec = np.copy(self.goal_vec)

    def perform_action(self, action):
        self.num_steps += 1
        curr_state = self.env.get_current_state()
        next_state = self.env.perform_action(action)
        reward = self.get_reward(curr_state, action, next_state)
        return (self._get_state_vector(next_state), reward)

    def is_terminal(self):
        if self.maximum_steps is not None and self.num_steps > self.maximum_steps:
            return True

        return np.sum(self.remaining_vec) == 0.

    def get_reward(self, state, action, next_state):
        if state == next_state:
            return self.wall_penalty

        if next_state in self.goals and self.remaining_vec[self.goals[next_state]] == 1:
            self.remaining_vec[self.goals[next_state]] = 0.
            return self.reward

        return self.time_penalty

    def visualize(self):
        '''
            Visualize the current game board.
        '''
        pass
        if len(self.env.grid.shape) > 2:
            raise NotImplementedError()

        self.cmap = matplotlib.colors.ListedColormap(['grey', 'black', 'blue', 'green', 'yellow', 'red'])
        self.color_norm = matplotlib.colors.BoundaryNorm(range(7), 6)

        world = np.copy(self.env.grid)  # assume walls are 1

        # show the location of the agent
        location = self.env.get_current_state()

        marked = False
        for goal, idx in self.goals.iteritems():
            if self.remaining_vec[idx] == 1:
                if location == goal:
                    world[goal] = 3
                    marked = True
                else:
                    world[goal] = 4  # not yet visited
            else:
                if location == goal:
                    world[goal] = 3
                    marked = True
                else:
                    world[goal] = 5

        if not marked:
            world[location] = 2

        return world


class HyperCubeObserver(Observer):
    '''
        Assumes the task is the hypercube task and report average number
        of steps to completion.
    '''
    def __init__(self, goal_dsets, eval_samples, report_wait=10):
        self.report_wait = report_wait
        self.dsets = goal_dsets
        self.eval_samples = eval_samples

    def trial(self, agent, task, goal):
        task.reset()
        task.set_goals(goal)
        agent.reset()
        curr_obs = task.get_start_state()
        steps = 1
        while True:
            act = agent.get_action(curr_obs)
            next_obs, reward = task.perform_action(act)
            if task.is_terminal():
                return steps, reward
            steps += 1
            curr_obs = next_obs

    def avg_steps(self, experiment, dset):
        sample = util.sample_if_large(dset, self.eval_samples)
        steps = [self.get_steps(experiment, goal) for goal in sample]
        return np.mean(steps)

    def get_steps(self, experiment, goal):
        episode_steps = []
        for _ in xrange(10):
            steps, reward = self.trial(experiment.agent, experiment.task, goal)
            episode_steps.append(steps)
        return np.mean(episode_steps)

    def observe(self, experiment):
        if experiment.num_episodes % self.report_wait == 0:

            metrics = {}
            for name, dset in self.dsets.iteritems():
                metrics[('avg_steps', name)] = self.avg_steps(experiment, dset)

            return metrics

        return None


class HyperCubeCurriculum(Controller):
    '''
        Two types of curriculum for now. The uniform and the bernoulli one.
    '''
    def __init__(self, goals, update_every=1, curr_type='uniform'):
        self.goals = goals
        self.update_every = update_every
        self.curr_type = curr_type

    def control(self, experiment):
        if experiment.num_episodes % self.update_every == 0:
            experiment.task.reset()

            goal = None
            if self.curr_type == 'uniform':
                goal = self.goals[random.randint(0, len(self.goals)-1)]
            else:
                raise NotImplementedError()

            experiment.task.set_goals(goal)
