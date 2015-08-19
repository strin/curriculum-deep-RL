import abc
import numpy as np
from collections import defaultdict
import cPickle as pickle
from datetime import datetime
import time
import util


class Experiment(object):
    def __init__(self, agent, task, controllers=None, observers=None):
        '''
            Designed to be similar to OnlineMaximizer, but specifically targeted
            for reinforcement learning experiments
        '''

        # CORE INPUTS
        # ------------
        self.agent = agent
        self.task = task

        self.halt = False

        # TRACKING
        # ----- ----- ----- ----- -----
        if observers is None:
            observers = []
        self.observers = observers
        self.history = defaultdict(lambda: (list(), list()))

        # CONTROLLERS
        # ----- ----- ----- ----- -----
        if controllers is None:
            controllers = [BasicController()]
        self.controllers = controllers

    def track(self):
        report = []
        for observer in self.observers:
            metrics = observer.observe(self)
            if metrics is None:
                continue
            for name, val in metrics.iteritems():
                timestamps, values = self.history[name]
                timestamps.append(self.num_episodes)
                values.append(val)

                util.metadata(name, val)
                report.append((name, val))

        if len(report) > 0:
            print ', '.join(['{}: {:.3f}'.format('.'.join(name), val) for name, val in report])
            with open('history.cpkl', 'w') as f:
                pickle.dump(dict(self.history), f)

    def run_experiments(self):
        '''
            Things to explore later:
                1) Adding multithreading capabilities
                2) More tracking capabilities?
        '''
        print 'Running experiments!'

        # no. of steps
        self.total_steps = 0
        self.num_episodes = 0
        while True:
            self.task.reset()

            self.episode_steps = 0
            self.episode_reward = 0.

            current_state = self.task.get_start_state()
            while True:
                action = self.agent.get_action(current_state)
                next_state, reward = self.task.perform_action(action)
                self.episode_reward += reward

                self.total_steps += 1
                self.episode_steps += 1

                if self.task.is_terminal():
                    self.agent.end_episode(reward)
                    break
                else:
                    self.agent.learn(next_state, reward)
                    current_state = next_state

            # these controllers will modify self.halt
            for controller in self.controllers:
                controller.control(self)

            self.track()

            self.num_episodes += 1

            if self.halt:
                return


class Controller(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def control(self, maximizer):
        return


class Observer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def observe(self, maximizer):
        return


class BasicController(Controller):
    '''
        TODO: For now, everything assumes that tasks are episodic.
    '''

    def __init__(self, report_wait=30, save_wait=30, max_episodes=10000):
        self.report_wait = report_wait
        self.save_wait = save_wait
        self.max_episodes = max_episodes

    def control(self, experiment):
        if experiment.num_episodes >= self.max_episodes:
            print 'Halted after reaching max episodes.'
            experiment.halt = True

        if experiment.num_episodes % self.report_wait == 0:
            print 'total steps: {}, episodes: {:.2f}'.format(experiment.total_steps,
                                                             experiment.num_episodes)
            util.metadata('total steps', experiment.total_steps)
            util.metadata('episodes', experiment.num_episodes)

            # report last seen
            time_rep = datetime.now().strftime('%H:%M:%S %m/%d')
            util.metadata('last_seen', time_rep)

            # report memory used
            util.metadata('gb_used', util.gb_used())

        if experiment.num_episodes % self.save_wait == 0 and experiment.num_episodes != 0:
            print 'saving params...'
            experiment.agent.save_params('params.cpkl')


class AverageRewardObserver(Observer):
    def __init__(self, report_wait=10):
        self.report_wait = report_wait  # number of episodes to average reward over
        self.reward_history = []

    def observe(self, experiment):
        self.reward_history.append(experiment.episode_reward)

        if experiment.num_episodes % self.report_wait == 0:
            average_reward = np.mean(self.reward_history)
            self.reward_history = []
            return {('average_reward', 'average_reward'): average_reward}

        return None


class AverageQValueObserver(Observer):
    def __init__(self, task_samples=100, report_wait=30):
        self.task_samples = task_samples  # number of states to randomly sample
        self.report_wait = report_wait
        self.states = []

    def observe(self, experiment):
        if experiment.num_episodes == 0:
            # Randomly sample TASK_SAMPLES states to evaluate the performance
            # of the algorithm

            # observer is only defined for agents performing Q-learning
            if not hasattr(experiment.agent, 'get_qvals'):
                raise NotImplementedError()

            num_actions = experiment.task.get_num_actions()
            experiment.task.reset()
            self.states.append(experiment.task.get_start_state())
            while True:
                rand_action = np.random.randint(num_actions)
                next_state, reward = experiment.task.perform_action(rand_action)
                if experiment.task.is_terminal():
                    break
                self.states.append(next_state)

            self.states = util.sample_if_large(self.states, self.task_samples)

        if experiment.num_episodes % self.report_wait == 0:

            def qval_mean(states):
                qvals = []
                for state in states:
                    qvals.append(np.max(experiment.agent.get_qvals(state)))
                return np.mean(qvals)

            return {('average_q_val', 'average_q_val'): qval_mean(self.states)}

        return None


class SpeedObserver(Observer):
    def __init__(self, report_wait=30):
        self.report_wait = report_wait
        self.prev_steps = 0
        self.prev_time = time.time()

    def observe(self, experiment):
        if experiment.num_episodes % self.report_wait != 0:
            return None
        seconds = time.time() - self.prev_time
        steps = experiment.total_steps - self.prev_steps
        self.prev_time = time.time()
        self.prev_steps = experiment.total_steps
        return {('speed', 'speed'): steps / seconds}
