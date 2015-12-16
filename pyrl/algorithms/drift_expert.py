import theano
import theano.tensor as T
import random
import numpy as np
import numpy.random as npr

import pyrl.optimizers as optimizers
import pyrl.layers as layers
import pyrl.prob as prob
from pyrl.utils import Timer
from pyrl.tasks.task import Task
from pyrl.agents.agent import DQN
from pyrl.agents.agent import TabularVfunc
from pyrl.algorithms.valueiter import DeepQlearn

class DriftExpert(object):
    '''
    an expert is a meta-feature, which is a function past-tasks and improvements.
    '''
    def __init__(self, policy, meta_feature_exs, train_func, eval_func, eta=1.):
        '''
        meta_feature_exs: a list of funcs, each func is a meta-feature extractor that
            takes input of past tasks and their improvement, as well as a candidate task
            produces output

        train_func: atomic operation to train an agent on a task.

        eval_func: atomic operation to evaluate an agent on a task, and get a score.

        '''
        self.meta_feature_exs = meta_feature_exs
        self.meta_weights = [1. / len(self.meta_feature_exs)] * len(self.meta_feature_exs)
        self.im_mem = [] # improvement memory (task, im)
        self.task_count = {}
        self.policy = policy
        self.train_func = train_func
        self.eval_func = eval_func
        self.eta = eta

        self.im_pred = {} # prediction of improvement per task.
        self.im_uncertainty = {}
        self.im_ucb = {}
        self.im_feat = {}

        # some diagnostic statistics.
        self.diagnostics = {}


    def run(self, tasks, num_epochs=1):
        for ni in range(num_epochs):
            # compute the score for each task.
            # and choose the task with maximal score.
            max_pred = -float('inf')
            max_task = None

            for task in tasks:
                if task not in self.task_count:
                    self.task_count[task] = 0
                uncertainty = np.sqrt(1. / (1. + self.task_count[task])) # upper confidence bound.
                self.im_uncertainty[task] = uncertainty

                self.im_feat[task] = {}

                pred = 0.
                for (ei, extractor) in enumerate(self.meta_feature_exs):
                    val = extractor(self.im_mem, task)
                    self.im_feat[task][str(extractor)] = val
                    pred += val * self.meta_weights[ei]         # expected improvement prediction.
                self.im_pred[task] = pred

                ucb = pred + self.eta * uncertainty
                self.im_ucb[task] = ucb

                if ucb > max_pred:
                    max_pred = ucb
                    max_task = task

            task_chosen = max_task

            # train task, and estimate improvement.
            score_before = self.eval_func(self.policy, task_chosen)
            self.train_func(task_chosen)
            score_after = self.eval_func(self.policy, task_chosen)
            im = score_after - score_before

            # update the meta-mode.
            self.im_mem.append((task_chosen, im))
            self.task_count[task_chosen] += 1

            # collect diagnostics.
            self.diagnostics['score before'] = score_before
            self.diagnostics['score after'] = score_after
            self.diagnostics['task chosen'] = task_chosen
            if 'latest score' not in self.diagnostics:
                self.diagnostics['latest score'] = {}
            self.diagnostics['latest score'][task_chosen] = score_after
            self.diagnostics['im_mem'] =  self.im_mem
            self.diagnostics['im_feat'] = self.im_feat



