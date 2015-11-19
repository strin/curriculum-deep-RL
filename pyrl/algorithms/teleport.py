import theano
import theano.tensor as T
import random
import numpy as np
import numpy.linalg as npla
import numpy.random as npr

import pyrl.optimizers as optimizers
import pyrl.layers as layers
import pyrl.prob as prob
from pyrl.evaluate import qval_stochastic
from pyrl.utils import Timer
from pyrl.tasks.task import Task
from pyrl.agents.agent import DQN
from pyrl.agents.agent import TabularVfunc
from pyrl.algorithms.valueiter import DeepQlearn
from pyrl.algorithms.multitask import DeepQlearnMT


class GPv0(object):
    '''
    an expert is a meta-feature, which is a function past-tasks and improvements.
    '''
    def __init__(self, dqn, kernel_func, expand_func, train_func, eval_func,
                 eta=1., sigma_n=0.01, K0=5, K=1):
        '''
        train_func: atomic operation to train an agent on a task.

        eval_func: atomic operation to evaluate an agent on a task, and get a score.

        '''
        # some params.
        self.K0 = K0
        self.K = K
        self.sigma_n = sigma_n

        # components.
        self.kernel_func = kernel_func
        self.expand_func = expand_func
        self.train_func = train_func
        self.eval_func = eval_func

        # representation.
        self.active_tasks = []
        self.task_count = {}
        self.task_score = {}
        self.dqn = dqn
        self.eta = eta

        self.im_pred = {} # prediction of improvement per task.
        self.im_sigma = {} # uncertainty of improvement per task.
        self.im_ucb = {} # confidence upper bound.

        # some diagnostic statistics.
        self.diagnostics = {}

    def run(self, tasks, num_epochs=1):
        if len(self.active_tasks) == 0: # initial round.
            # chose a set of active tasks uniformly at random.
            self.active_tasks = prob.choice(tasks, size=self.K0, replace=True)

        # set local variables.
        active_tasks = self.active_tasks
        K = self.K

        for ni in range(num_epochs):
            # compute old score if necessary.
            for task in active_tasks:
                if task not in self.task_score:
                    self.task_score[task] = self.eval_func(task)

            # learn on each task.
            for task in active_tasks[::-1]:
                self.train_func(task)

            # evaluate improvement.
            im = {}
            for task in active_tasks:
                new_score = self.eval_func(task)
                im[task] = new_score - self.task_score[task]
                self.task_score[task] = new_score

            # create candidate set.
            candidate_set = set()
            for task in active_tasks:
                candidate_set = candidate_set.union(set(self.expand_func(task)))
            candidate_set = list(candidate_set)

            new_tasks = [task for task in candidate_set if task not in active_tasks]
            if len(new_tasks) == 0:
                print 'WARNING: new tasks is empty in GP'

            # use Gaussian Process to estimate potential function.
            N = len(im)
            KXX = np.zeros((N, N))
            y = np.zeros(N)
            for (ti, (task_i, im_i)) in enumerate(im.items()):
                for (tj, (task_j, im_j)) in enumerate(im.items()):
                    KXX[ti, tj] = self.kernel_func(task_i, task_j)
            for (ti, (task_i, im_i)) in enumerate(im.items()):
                y[ti] = im_i

            M = len(new_tasks)
            KXsX = np.zeros((M, N))
            KXsXs = np.zeros((M, M))
            for (ti, task_i) in enumerate(new_tasks):
                for (tj, (task_j, im_j)) in enumerate(im.items()):
                    KXsX[ti, tj] = self.kernel_func(task_i, task_j)
                KXsXs[ti, ti] = self.kernel_func(task_i, task_i)

            KXXinv = npla.inv(KXX + self.sigma_n**2 * np.eye(N))

            pred_mean = np.dot(KXsX, np.dot(KXXinv, y))
            pred_cov = KXsXs - np.dot(KXsX, np.dot(KXXinv, np.transpose(KXsX)))
            pred_sigma = np.sqrt(np.diag(pred_cov))

            im_pred = {}
            im_sigma = {}
            im_ucb = {}
            for (ti, task) in enumerate(new_tasks):
                im_pred[task] = pred_mean[ti]
                im_sigma[task] = pred_sigma[ti]
                im_ucb[task] = pred_mean[ti] + self.eta * pred_sigma[ti]

            new_tasks = sorted(new_tasks, key=lambda task: im_ucb[task], reverse=True)
            new_tasks_selected = new_tasks[:K]

            self.active_tasks.extend(new_tasks_selected)

            # collect diagnostics.
            self.diagnostics['im'] = im
            self.diagnostics['pred'] = im_pred
            self.diagnostics['sigma'] = im_sigma
            self.diagnostics['ucb'] = im_ucb
            self.diagnostics['score'] = self.task_score
            self.diagnostics['new-tasks'] = new_tasks
            self.diagnostics['new-tasks-selected'] = new_tasks_selected
            self.diagnostics['active-tasks'] = active_tasks


class GPv0a(object):
    '''
    an expert is a meta-feature, which is a function past-tasks and improvements.
    '''
    def __init__(self, dqn, kernel_func, expand_func, train_func, eval_func,
                 eta=1., sigma_n=0.01, K=5, K0=1):
        '''
        train_func: atomic operation to train an agent on a task.

        eval_func: atomic operation to evaluate an agent on a task, and get a score.

        '''
        # some params.
        self.K = K
        self.K0 = K0
        self.sigma_n = sigma_n

        # components.
        self.kernel_func = kernel_func
        self.expand_func = expand_func
        self.train_func = train_func
        self.eval_func = eval_func

        # representation.
        self.active_tasks = set()
        self.passive_tasks = set()
        self.task_count = {}
        self.task_score = {}
        self.dqn = dqn
        self.eta = eta

        self.im_pred = {} # prediction of improvement per task.
        self.im_sigma = {} # uncertainty of improvement per task.
        self.im_ucb = {} # confidence upper bound.

        # some diagnostic statistics.
        self.diagnostics = {}

    def run(self, tasks, num_epochs=1):
        # set local variables.
        K = self.K
        K0 = self.K0

        if len(self.active_tasks) == 0: # initial round.
            # chose a set of active tasks uniformly at random.
            self.active_tasks = set(prob.choice(tasks, size=K+K0, replace=True))

        for ni in range(num_epochs):
            active_tasks = self.active_tasks
            passive_tasks = self.passive_tasks
            curr_tasks = active_tasks.union(passive_tasks)

            # compute old score if necessary.
            for task in curr_tasks:
                if task not in self.task_score:
                    self.task_score[task] = self.eval_func(task)

            # learn on each task.
            for task in active_tasks:
                self.train_func(task)

            # evaluate improvement.
            im = {}
            for task in curr_tasks:
                new_score = self.eval_func(task)
                im[task] = new_score - self.task_score[task]
                self.task_score[task] = new_score

            # create candidate set.
            candidate_set = set()
            for task in curr_tasks:
                candidate_set = candidate_set.union(set(self.expand_func(task)))
            candidate_set = list(candidate_set)

            # new_tasks = [task for task in candidate_set if task not in active_tasks]
            new_tasks = candidate_set
            if len(new_tasks) == 0:
                print 'WARNING: new tasks is empty in GP'

            # use Gaussian Process to estimate potential function.
            N = len(im)
            KXX = np.zeros((N, N))
            y = np.zeros(N)
            for (ti, (task_i, im_i)) in enumerate(im.items()):
                for (tj, (task_j, im_j)) in enumerate(im.items()):
                    KXX[ti, tj] = self.kernel_func(task_i, task_j)
            for (ti, (task_i, im_i)) in enumerate(im.items()):
                y[ti] = im_i

            M = len(new_tasks)
            KXsX = np.zeros((M, N))
            KXsXs = np.zeros((M, M))
            for (ti, task_i) in enumerate(new_tasks):
                for (tj, (task_j, im_j)) in enumerate(im.items()):
                    KXsX[ti, tj] = self.kernel_func(task_i, task_j)
                KXsXs[ti, ti] = self.kernel_func(task_i, task_i)

            KXXinv = npla.inv(KXX + self.sigma_n**2 * np.eye(N))

            pred_mean = np.dot(KXsX, np.dot(KXXinv, y))
            pred_cov = KXsXs - np.dot(KXsX, np.dot(KXXinv, np.transpose(KXsX)))
            pred_sigma = np.sqrt(np.diag(pred_cov))

            im_pred = {}
            im_sigma = {}
            im_ucb = {}
            for (ti, task) in enumerate(new_tasks):
                im_pred[task] = pred_mean[ti]
                im_sigma[task] = pred_sigma[ti]
                im_ucb[task] = pred_mean[ti] + self.eta * pred_sigma[ti]

            new_tasks = sorted(new_tasks, key=lambda task: im_ucb[task], reverse=True)
            new_tasks_selected = new_tasks[:K]

            # randomly choose some new tasks.
            new_tasks_selected.extend(prob.choice(tasks, size=K0, replace=False))

            self.passive_tasks = passive_tasks.union(active_tasks).difference(set(new_tasks_selected))
            self.active_tasks = set(new_tasks_selected)

            # collect diagnostics.
            self.diagnostics['im'] = im
            self.diagnostics['pred'] = im_pred
            self.diagnostics['sigma'] = im_sigma
            self.diagnostics['ucb'] = im_ucb
            self.diagnostics['score'] = self.task_score
            self.diagnostics['new-tasks'] = new_tasks
            self.diagnostics['new-tasks-selected'] = new_tasks_selected
            self.diagnostics['active-tasks'] = self.active_tasks
            self.diagnostics['passive-tasks'] = self.passive_tasks

class GPv1(object):
    '''
    an expert is a meta-feature, which is a function past-tasks and improvements.

    in V1, we collect experiences only for tasks in active set.
    '''
    def __init__(self, dqn, kernel_func, expand_func, train_func, eval_func,
                 eta=1., sigma_n=0.01, K1=1, K=1, K0=1):
        '''
        train_func: atomic operation to train an agent on a task.

        eval_func: atomic operation to evaluate an agent on a task, and get a score.

        '''
        # some params.
        self.K = K
        self.K1 = K
        self.K0 = K0
        self.sigma_n = sigma_n

        # components.
        self.kernel_func = kernel_func
        self.expand_func = expand_func
        self.train_func = train_func
        self.eval_func = eval_func

        # representation.
        self.active_tasks = set()
        self.passive_tasks = set()
        self.task_count = {}
        self.task_score = {}
        self.dqn = dqn
        self.eta = eta

        self.im_pred = {} # prediction of improvement per task.
        self.im_sigma = {} # uncertainty of improvement per task.
        self.im_ucb = {} # confidence upper bound.

        # some diagnostic statistics.
        self.diagnostics = {}

    def run(self, tasks, num_epochs=1):
        if len(self.active_tasks) == 0: # initial round.
            # chose a set of active tasks uniformly at random.
            self.active_tasks = prob.choice(tasks, size=self.K, replace=True)

        # set local variables.
        active_tasks = self.active_tasks
        passive_tasks = self.passive_tasks
        K = self.K
        K0 = self.K0
        K1 = self.K1

        for ni in range(num_epochs):
            random_tasks = prob.choice(tasks, size=K0, replace=False)

            if len(passive_tasks) >= K1:
                selected_passive_tasks = prob.choice(list(self.passive_tasks), size=K1, replace=False)
                curr_tasks = list(self.active_tasks) + selected_passive_tasks + random_tasks
            else:
                curr_tasks = list(self.active_tasks) + random_tasks
            curr_tasks = list(set(curr_tasks)) # dedup.

            # compute old score if necessary.
            for task in curr_tasks:
                if task not in self.task_score:
                    self.task_score[task] = self.eval_func(task)

            # learn on each task.
            for task in curr_tasks:
                self.train_func(task)

            # evaluate improvement.
            im = {}
            for task in curr_tasks:
                new_score = self.eval_func(task)
                im[task] = new_score - self.task_score[task]
                self.task_score[task] = new_score

            # create candidate set.
            candidate_set = set()
            for task in curr_tasks:
                candidate_set = candidate_set.union(set(self.expand_func(task)))
            candidate_set = list(candidate_set)

            new_tasks = candidate_set

            # use Gaussian Process to estimate potential function.
            N = len(im)
            KXX = np.zeros((N, N))
            y = np.zeros(N)
            for (ti, (task_i, im_i)) in enumerate(im.items()):
                for (tj, (task_j, im_j)) in enumerate(im.items()):
                    KXX[ti, tj] = self.kernel_func(task_i, task_j)
            for (ti, (task_i, im_i)) in enumerate(im.items()):
                y[ti] = im_i

            M = len(new_tasks)
            KXsX = np.zeros((M, N))
            KXsXs = np.zeros((M, M))
            for (ti, task_i) in enumerate(new_tasks):
                for (tj, (task_j, im_j)) in enumerate(im.items()):
                    KXsX[ti, tj] = self.kernel_func(task_i, task_j)
                KXsXs[ti, ti] = self.kernel_func(task_i, task_i)

            KXXinv = npla.inv(KXX + self.sigma_n**2 * np.eye(N))

            pred_mean = np.dot(KXsX, np.dot(KXXinv, y))
            pred_cov = KXsXs - np.dot(KXsX, np.dot(KXXinv, np.transpose(KXsX)))
            pred_sigma = np.sqrt(np.diag(pred_cov))

            im_pred = {}
            im_sigma = {}
            im_ucb = {}
            for (ti, task) in enumerate(new_tasks):
                im_pred[task] = pred_mean[ti]
                im_sigma[task] = pred_sigma[ti]
                im_ucb[task] = pred_mean[ti] + self.eta * pred_sigma[ti]

            new_tasks = sorted(new_tasks, key=lambda task: im_ucb[task], reverse=True)
            new_tasks_selected = new_tasks[:K]

            self.passive_tasks = self.passive_tasks.union(self.active_tasks).difference(new_tasks_selected)
            self.active_tasks = new_tasks_selected

            # collect diagnostics.
            self.diagnostics['im'] = im
            self.diagnostics['pred'] = im_pred
            self.diagnostics['sigma'] = im_sigma
            self.diagnostics['ucb'] = im_ucb
            self.diagnostics['curr_tasks'] = curr_tasks
            self.diagnostics['score'] = self.task_score
            self.diagnostics['new-tasks'] = new_tasks
            self.diagnostics['new-tasks-selected'] = new_tasks_selected
            self.diagnostics['active-tasks'] = active_tasks


class GPv2(object):
    '''
    an expert is a meta-feature, which is a function past-tasks and improvements.

    in V1, we collect experiences only for tasks in active set.
    '''
    def __init__(self, dqn, kernel_func, expand_func, train_func, eval_func,
                 eta=1., sigma_n=0.01, K1=1, K=1, K0=5):
        '''
        train_func: atomic operation to train an agent on a task.

        eval_func: atomic operation to evaluate an agent on a task, and get a score.

        '''
        # some params.
        self.K = K
        self.K1 = K
        self.K0 = K0
        self.sigma_n = sigma_n

        # components.
        self.kernel_func = kernel_func
        self.expand_func = expand_func
        self.train_func = train_func
        self.eval_func = eval_func

        # representation.
        self.active_tasks = set()
        self.passive_tasks = set()
        self.task_count = {}
        self.task_score = {}
        self.dqn = dqn
        self.eta = eta

        self.im_pred = {} # prediction of improvement per task.
        self.im_sigma = {} # uncertainty of improvement per task.
        self.im_ucb = {} # confidence upper bound.

        # some diagnostic statistics.
        self.diagnostics = {}

    def run(self, tasks, num_epochs=1):
        if len(self.active_tasks) == 0: # initial round.
            # chose a set of active tasks uniformly at random.
            self.active_tasks = prob.choice(tasks, size=self.K, replace=True)

        # set local variables.
        active_tasks = self.active_tasks
        passive_tasks = self.passive_tasks
        K = self.K
        K0 = self.K0
        K1 = self.K1

        for ni in range(num_epochs):
            # compute old score if necessary.
            for task in active_tasks:
                if task not in self.task_score:
                    self.task_score[task] = self.eval_func(task)

            # learn on each task.
            for task in active_tasks:
                self.train_func(task)

            if len(passive_tasks) >= K1:
                selected_passive_tasks = prob.choice(self.passive_tasks, size=K1, replace=False)
                for task in selected_passive_tasks:
                    self.train_func(task)

            # evaluate improvement.
            im = {}
            for task in active_tasks:
                new_score = self.eval_func(task)
                im[task] = new_score - self.task_score[task]
                self.task_score[task] = new_score

            # create candidate set.
            candidate_set = set()
            for task in active_tasks:
                candidate_set = candidate_set.union(set(self.expand_func(task)))
            candidate_set = candidate_set.union(set(prob.choice(tasks, size=K0, replace=False)))
            candidate_set = list(candidate_set)

            new_tasks = candidate_set

            # use Gaussian Process to estimate potential function.
            N = len(im)
            KXX = np.zeros((N, N))
            y = np.zeros(N)
            for (ti, (task_i, im_i)) in enumerate(im.items()):
                for (tj, (task_j, im_j)) in enumerate(im.items()):
                    KXX[ti, tj] = self.kernel_func(task_i, task_j)
            for (ti, (task_i, im_i)) in enumerate(im.items()):
                y[ti] = im_i

            M = len(new_tasks)
            KXsX = np.zeros((M, N))
            KXsXs = np.zeros((M, M))
            for (ti, task_i) in enumerate(new_tasks):
                for (tj, (task_j, im_j)) in enumerate(im.items()):
                    KXsX[ti, tj] = self.kernel_func(task_i, task_j)
                KXsXs[ti, ti] = self.kernel_func(task_i, task_i)

            KXXinv = npla.inv(KXX + self.sigma_n**2 * np.eye(N))

            pred_mean = np.dot(KXsX, np.dot(KXXinv, y))
            pred_cov = KXsXs - np.dot(KXsX, np.dot(KXXinv, np.transpose(KXsX)))
            pred_sigma = np.sqrt(np.diag(pred_cov))

            im_pred = {}
            im_sigma = {}
            im_ucb = {}
            for (ti, task) in enumerate(new_tasks):
                im_pred[task] = pred_mean[ti]
                im_sigma[task] = pred_sigma[ti]
                im_ucb[task] = pred_mean[ti] + self.eta * pred_sigma[ti]

            new_tasks = sorted(new_tasks, key=lambda task: im_ucb[task], reverse=True)
            new_tasks_selected = new_tasks[:K]

            self.passive_task = self.passive_tasks.union(self.active_tasks).difference(new_tasks_selected)
            self.active_tasks = new_tasks_selected

            # collect diagnostics.
            self.diagnostics['im'] = im
            self.diagnostics['pred'] = im_pred
            self.diagnostics['sigma'] = im_sigma
            self.diagnostics['ucb'] = im_ucb
            self.diagnostics['score'] = self.task_score
            self.diagnostics['new-tasks'] = new_tasks
            self.diagnostics['new-tasks-selected'] = new_tasks_selected
            self.diagnostics['active-tasks'] = active_tasks
