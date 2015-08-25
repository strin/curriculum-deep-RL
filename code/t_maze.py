import random
import environment
import numpy as np
import matplotlib
from experiment import Observer


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


class TMazeObserver(Observer):
    '''
        Assumes the task is the t-shaped maze task and reports percent
        success and average numver of steps on NUM_SAMPLES trials.
    '''
    def __init__(self, num_samples=10, report_wait=10, test_task=None):
        self.report_wait = report_wait  # number of episodes to average reward over
        self.num_samples = num_samples
        self.test_task = test_task

    def trial(self, agent, task):
        task.reset()
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

    def get_statistics(self, agent, task):
            step_history = []
            success_history = []

            for _ in xrange(self.num_samples):

                steps, final_reward = self.trial(agent, task)
                step_history.append(steps)
                success_history.append(final_reward > 0)

            percent_success = np.mean(success_history)
            avg_steps = np.mean(step_history)

            return percent_success, avg_steps

    def observe(self, experiment):
        if experiment.num_episodes % self.report_wait == 0:

            agent = experiment.agent
            tasks = {'train': experiment.task}

            if self.test_task is not None:
                tasks['test'] = self.test_task

            metrics = {}
            for name, task in tasks.iteritems():
                percent_success, avg_steps = self.get_statistics(agent, task)

                metrics[('percent_success', name)] = percent_success
                metrics[('avg_steps', name)] = avg_steps

            return metrics

        return None
