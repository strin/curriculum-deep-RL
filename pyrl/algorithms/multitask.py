# code for multi task RL algorithms.
import theano
import theano.tensor as T
import random
import numpy as np
import numpy.random as npr

import pyrl.optimizers
import pyrl.layers
from pyrl.algorithms.valueiter import DeepQlearn
from pyrl.agents.agent import eval_policy_reward
import pyrl.prob as prob



class SingleLearnerSequential(object):
    def __init__(self, dqn, tasks, **kwargs):
        self.dqn = dqn
        self.tasks = tasks
        self.deepQlearn = DeepQlearn(tasks[0], dqn, **kwargs)

    def run(self, num_epochs=1, num_episodes=1):
        for ti, task in enumerate(self.tasks):
            # (TODO) this breaks away the abstraction.
            self.deepQlearn.task = task
            self.dqn.task = task
            # run training.
            self.deepQlearn.run(num_episodes)

class SingleLearnerMAB(object):
    def __init__(self, dqn, tasks, mab_gamma, mab_scale, mab_batch_size, **kwargs):
        self.dqn = dqn
        self.tasks = tasks
        self.log_weights = [0. for task in tasks]
        self.mab_gamma = mab_gamma
        self.mab_batch_size = mab_batch_size
        self.mab_scale = mab_scale
        self.deepQlearn = DeepQlearn(tasks[0], dqn, **kwargs)
        self.cumulative_epochs = 0

    def run(self, num_epochs=1, num_episodes=1):
        num_tasks = len(self.tasks)
        for epoch in range(num_epochs):
            # choose task based on weights.
            ti = -1
            if npr.rand() < self.mab_gamma:
                ti = npr.choice(range(num_tasks), 1)[0]
            else:
                p = np.exp(prob.normalize_log(self.log_weights))
                ti = npr.choice(range(num_tasks), 1, replace=True, p=p)[0]
            task = self.tasks[ti]

            # (TODO) this breaks away the abstraction.
            self.deepQlearn.task = task
            self.dqn.task = task

            # run training.
            self.deepQlearn.run(num_episodes)

            # update weights.
            self.cumulative_epochs += 1
            if self.cumulative_epochs >= self.mab_batch_size:
                self.log_weights[:] = 0.
            else:
                for ti, task in enumerate(self.tasks):
                    performance_gain = eval_policy_reward(self.dqn, task, num_episodes=10000)
                    self.log_weights[ti] += self.mab_gamma * self.mab_scale * performance_gain / num_tasks


class HordeLearnSequential(object):
    def __init__(self, horde_dqn, **kwargs):
        self.tasks = horde_dqn.tasks
        self.num_tasks = len(self.tasks)
        self.horde_dqn = horde_dqn
        self.deepQlearns = [DeepQlearn(task, dqn, **kwargs)
                            for (task, dqn) in zip(self.tasks, self.horde_dqn.dqns)]

    def run(self, num_epochs = 1, num_episodes = 1):
        for ti, task in enumerate(self.tasks):
            learner = self.deepQlearns[ti]
            learner.run(num_episodes)


