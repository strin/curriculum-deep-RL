import theano
import theano.tensor as T
import random
import numpy as np
import numpy.linalg as npla
import numpy.random as npr

import pyrl.optimizers as optimizers
import pyrl.layers as layers
import pyrl.prob as prob
from pyrl.utils import Timer
from pyrl.tasks.task import Task
from pyrl.agents.agent import DQN
from pyrl.agents.agent import TabularVfunc
from pyrl.algorithms.valueiter import DeepQlearn


class DQCL:
    class BasicGP(object):
        '''
        an expert is a meta-feature, which is a function past-tasks and improvements.
        '''
        def __init__(self, policy, kernel_func, train_func, eval_func, eta=1.):
            '''
            train_func: atomic operation to train an agent on a task.

            eval_func: atomic operation to evaluate an agent on a task, and get a score.

            '''
            self.im_mem = [] # improvement memory (task, im)
            self.task_count = {}
            self.policy = policy
            self.kernel_func = kernel_func
            self.train_func = train_func
            self.eval_func = eval_func
            self.eta = eta

            self.im_pred = {} # prediction of improvement per task.
            self.im_sigma = {} # uncertainty of improvement per task.
            self.im_ucb = {} # confidence upper bound.

            # some diagnostic statistics.
            self.diagnostics = {}

        def run(self, tasks, num_epochs=1):
            for ni in range(num_epochs):
                # compute the score for each task.
                # and choose the task with maximal score.
                max_pred = -float('inf')
                max_task = None

                # fit a GP first.
                sigma_n = 0.01

                im_mem_dedup = {}
                for (task, im) in self.im_mem[::-1]:
                    if task in im_mem_dedup:
                        continue
                    im_mem_dedup[task] = im

                N = len(self.im_mem)
                KXX = np.zeros((N, N))
                y = np.zeros(N)
                for (ti, (task_i, im_i)) in enumerate(im_mem_dedup.items()):
                    for (tj, (task_j, im_j)) in enumerate(im_mem_dedup.items()):
                        KXX[ti, tj] = self.kernel_func(task_i, task_j)
                for (ti, (task_i, im_i)) in enumerate(im_mem_dedup.items()):
                    y[ti] = im_i

                M = len(tasks)
                KXsX = np.zeros((M, N))
                KXsXs = np.zeros((M, M))
                for (ti, task_i) in enumerate(tasks):
                    for (tj, (task_j, im_j)) in enumerate(im_mem_dedup.items()):
                        KXsX[ti, tj] = self.kernel_func(task_i, task_j)
                    KXsXs[ti, ti] = self.kernel_func(task_i, task_i)

                KXXinv = npla.inv(KXX + sigma_n**2 * np.eye(N))
                pred_mean = np.dot(KXsX, np.dot(KXXinv, y))
                pred_cov = KXsXs - np.dot(KXsX, np.dot(KXXinv, np.transpose(KXsX)))
                pred_sigma = np.sqrt(np.diag(pred_cov))

                max_task = None
                max_ucb = -float('inf')
                for (ti, task) in enumerate(tasks):
                    self.im_pred[task] = pred_mean[ti]
                    self.im_sigma[task] = pred_sigma[ti]
                    self.im_ucb[task] = pred_mean[ti] + self.eta * pred_sigma[ti]
                    if self.im_ucb[task] > max_ucb:
                        max_ucb = self.im_ucb[task]
                        max_task = task

                task_chosen = max_task

                # train task, and estimate improvement.
                score_before = self.eval_func(self.policy, task_chosen)
                self.train_func(task_chosen)
                score_after = self.eval_func(self.policy, task_chosen)
                im = score_after - score_before

                # update the meta-mode.
                self.im_mem.append((task_chosen, im))

                if task_chosen not in self.task_count:
                    self.task_count[task_chosen] = 0
                self.task_count[task_chosen] += 1

                # collect diagnostics.
                self.diagnostics['score before'] = score_before
                self.diagnostics['score after'] = score_after
                self.diagnostics['task chosen'] = task_chosen
                if 'latest score' not in self.diagnostics:
                    self.diagnostics['latest score'] = {}
                self.diagnostics['latest score'][task_chosen] = score_after





