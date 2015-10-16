# code for multi task RL algorithms.
import theano
import theano.tensor as T
import random
import numpy as np
import numpy.random as npr
import numpy.linalg as npla

import pyrl.optimizers
import pyrl.layers
from pyrl.algorithms.valueiter import DeepQlearn
from pyrl.agents.agent import eval_policy_reward
from pyrl.evaluate import eval_dataset, expected_reward_tabular_normalized
import pyrl.prob as prob



class SingleLearnerSequential(object):
    def __init__(self, dqn, tasks, **kwargs):
        self.dqn = dqn
        self.tasks = tasks
        self.num_tasks = len(self.tasks)
        self.deepQlearn = DeepQlearn(tasks[0], dqn, **kwargs)
        self.t = 0

    def run(self, num_epochs=1, num_episodes=1):
        for ei in range(num_epochs):
            ti = self.t % self.num_tasks
            task = self.tasks[ti]

            # (TODO) this breaks away the abstraction.
            self.deepQlearn.task = task
            self.dqn.task = task
            # run training.
            self.deepQlearn.run(num_episodes, task)

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

class SingleLearnerGPt(object):
    def __init__(self, dqn, tasks, dist, gpt_eta, gpt_r, gpt_v, gpt_sigma, gpt_kappa, **kwargs):
        '''
        init GP-t algorithms based multi-task single learner.
        GP-t uses a Gaussian Process regression to infer progress in task space at a given time t.

        The covariance is defined as
        The function is defined as $$p(f \mid X) = \mathcal{N}(0, K)$$

        Hyper-Parameters
        ==========
        dist: distance metric between two tasks.
        gpt_eta: time-decaying factor, controlling decaying weight of out-dated progress examples.
        gpt_r: task correlation hyper-parameter.
        gpt_sigma: noise level.
        gpt_kappa: tradeoff hyper-parameter for exploration-exploitation tradeoff.
        '''
        self.gpt_eta = gpt_eta
        self.gpt_r = gpt_r
        self.gpt_sigma = gpt_sigma
        self.gpt_v = gpt_v
        self.gpt_kappa = gpt_kappa
        self.dist = dist

        # q-learning.
        self.dqn = dqn
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.deepQlearn = DeepQlearn(tasks[0], dqn, **kwargs)

        # GP.
        self.examples = []   # collect examples that measure progress.
        self.K = np.array([])
        self.y = np.array([])
        self.last_performance = eval_dataset(self.dqn, self.tasks)
        self.t = 0

        # some diagnostic information.
        self.ucb = None
        self.mu = None
        self.sigma = None
        self.last_task = tasks[0]
        self.last_task_ti = 0
        self.last_task_performance = None
        self.last_progress = None

    def _run_task(self, task, num_episodes=1):
        self.deepQlearn.task = task
        self.dqn.task = task
        # run training.
        self.deepQlearn.run(num_episodes, task)


    def run(self, num_epochs=1, num_episodes=1):
        cov_func = lambda task1, task2, t1, t2: self.gpt_v * np.exp(- (self.dist(task1, task2) ** 2 * self.gpt_r + self.gpt_eta * (t1 - t2) ** 2))
        for ei in range(num_epochs):
            # task selection.
            # complexity max(#task * history, history ** 2.3)
            if len(self.examples) == 0: # no prior experience, choose randomly.
                task = prob.choice(self.tasks, 1)[0]
            else:
                # GP-t.
                mu = np.zeros(self.num_tasks)
                sigma = np.zeros(self.num_tasks)
                ucb = np.zeros(self.num_tasks)
                # Kinv = npla.inv(self.K + self.gpt_sigma ** 2)
                # Kinv_y = np.dot(Kinv, self.y)
                Kinv_y = npla.solve(self.K + np.eye(self.t) * self.gpt_sigma ** 2, self.y)
                for ti, task in enumerate(self.tasks):
                    vec = np.zeros(self.t)
                    for ei in range(self.t):
                        (t_ei, task_ei, _) = self.examples[ei]
                        vec[ei] = cov_func(task, task_ei, self.t, t_ei)
                    mu[ti] = np.dot(vec, Kinv_y)
                    Kinv_vec = npla.solve(self.K + np.eye(self.t) * self.gpt_sigma ** 2, vec)
                    sigma[ti] = self.gpt_v + self.gpt_sigma ** 2 - np.dot(vec, Kinv_vec)
                    ucb[ti] = mu[ti] + self.gpt_kappa * sigma[ti]
                best_ti = np.argmax(ucb)
                task = self.tasks[best_ti]
                # store information for diagnosis.
                self.mu = mu
                self.sigma = sigma
                self.ucb = ucb

            # import pdb; pdb.set_trace()
            # run training.
            self._run_task(task, num_episodes=num_episodes)

            # evaluate performance.
            self.last_task_performance = np.zeros(self.num_tasks)
            for ti in range(self.num_tasks):
                self.last_task_performance[ti] = expected_reward_tabular_normalized(self.dqn, self.tasks[ti], tol=1e-4)
            performance = np.mean(self.last_task_performance)
            progress = performance - self.last_performance
            # update statistics.
            self.examples.append((self.t, task, progress))
            self.t += 1
            t = self.t

            new_K = np.zeros((t, t))
            new_y = np.zeros(t)
            if t > 1:
                new_K[:t - 1, :t - 1] = self.K
                new_y[:t - 1] = self.y
            new_K[t - 1, t - 1] = self.gpt_v
            new_y[t - 1] = progress
            for ei in range(t - 1):
                (t_ei, task_ei, _) = self.examples[ei]
                new_K[t - 1, ei] = cov_func(task_ei, task, t_ei, t - 1)
                new_K[ei, t - 1] = new_K[t - 1, ei] # symmetric.
            self.K = new_K
            self.y = new_y
            self.last_performance = performance
            self.last_progress = progress
            self.last_task = task
            self.last_task_ti = self.tasks.index(task)








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


