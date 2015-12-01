import theano
import theano.tensor as T
import random
import numpy as np
import numpy.linalg as npla
import numpy.random as npr
import dill as pickle

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


class OracleIm(object):
    '''
    Select task based on oracle improvment.
    To save computation, it only chooses the task with best improvment among a set of pre-selected tasks.
    The set of pre-selected tasks are chosen uniformly at random.
    '''
    def __init__(self, learner, num_sample, train_func, eval_func):
        self.train_func = train_func
        self.eval_func = eval_func
        self.num_sample = num_sample
        self.learner = learner

        self.diagnostics = {}

    def _eval_im(self, task):
        score_before = self.eval_func(self.learner, task)
        new_learner = self.learner.copy()
        self.train_func(new_learner, task)
        score_after = self.eval_func(new_learner, task)
        print 'task', task, 'before', score_before, 'after', score_after
        return score_after - score_before

    def run(self, tasks, num_epochs=1):
        for ni in range(num_epochs):
            sub_tasks = prob.choice(tasks, size=self.num_sample, replace=False)
            ims = []
            self.diagnostics['im_task'] = {}
            for task in sub_tasks:
                im = self._eval_im(task)
                ims.append(im)
                self.diagnostics['im_task'][task] = im

            max_ind = np.argmax(ims)
            chosen_task = sub_tasks[max_ind]
            self.diagnostics['chosen_task'] = str(chosen_task)

            self.train_func(self.learner, chosen_task)


class GPt(object):
    def __init__(self, train_func, eval_func, kernel_func, gpt_sigma, gpt_kappa):
        '''
        GP-t algorithms based multi-task single learner.
        aims to model I(t, task) as approximation of I(M, task).

        GP-t uses a Gaussian Process regression to infer progress in task space at a given time t.

        The covariance is defined as
        The function is defined as $$p(f \mid X) = \mathcal{N}(0, K)$$

        Hyper-Parameters
        ==========
        gpt_sigma: noise level.
        gpt_kappa: tradeoff hyper-parameter for exploration-exploitation tradeoff.
        '''
        self.gpt_sigma = gpt_sigma
        self.gpt_kappa = gpt_kappa

        self.train_func = train_func
        self.eval_func = eval_func
        self.kernel_func = kernel_func

        # GP.
        self.ims = []   # collect (task, im) tuples that track progress.
        self.t = 0

        # some diagnostic information.
        self.diagnostics = {}

    def run(self, tasks, num_epochs=1, num_episodes=1):
        for ei in range(num_epochs):
            t = self.t

            # task selection.
            if t == 0: # no prior experience, choose randomly.
                task = prob.choice(tasks, 1)[0]
            else:
                # GP-t.
                N = len(self.ims)
                KXX = np.zeros((N, N))
                y = np.zeros(N)

                for (t_i, (task_i, im_i)) in enumerate(self.ims):
                    for (t_j, (task_j, im_j)) in enumerate(self.ims):
                        KXX[t_i, t_j] = self.kernel_func(t_i, task_i, t_j, task_j)

                for (ti, (task_i, im_i)) in enumerate(self.ims):
                    y[ti] = im_i

                M = len(tasks)
                KXsX = np.zeros((M, N))
                KXsXs = np.zeros((M, M))

                for (t_i, task_i) in enumerate(tasks):
                    for (t_j, (task_j, im_j)) in enumerate(self.ims):
                        KXsX[t_i, t_j] = self.kernel_func(t, task_i, t_j, task_j)
                    KXsXs[t_i, t_i] = self.kernel_func(t, task_i, t, task_i)

                KXXinv = npla.inv(KXX + self.gpt_sigma ** 2 * np.eye(N))

                pred_mean = np.dot(KXsX, np.dot(KXXinv, y))
                pred_cov = KXsXs - np.dot(KXsX, np.dot(KXXinv, np.transpose(KXsX)))
                pred_sigma = np.sqrt(np.diag(pred_cov))

                pred_ucb = pred_mean + self.gpt_kappa * pred_sigma

                best_ti = np.argmax(pred_ucb)
                task = tasks[best_ti]

                # store information for diagnosis.
                self.diagnostics['mean'] = {str(task): mean for (task, mean) in zip(tasks, pred_mean)}
                self.diagnostics['sigma'] = {str(task): sigma for (task, sigma) in zip(tasks, pred_sigma)}
                self.diagnostics['ucb'] = {str(task): ucb for (task, ucb) in zip(tasks, pred_ucb)}

            score_before = self.eval_func(task)
            self.train_func(task)
            score_after = self.eval_func(task)
            im = score_after - score_before

            self.diagnostics['chosen_task'] = str(task)
            self.diagnostics['im'] = im

            self.ims.append((task, im))
            self.t += 1



class GPv0(object):
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
                 init_tasks=None, eta=1., sigma_n=0.01, K=5, K0=1):
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
        self.init_tasks = init_tasks

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
            if self.init_tasks:
                self.active_tasks = set(prob.choice(self.init_tasks, size=K+K0, replace=True))
            else:
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


class GPv3(object):
    '''
    an expert is a meta-feature, which is a function past-tasks and improvements.

    in V1, we collect experiences only for tasks in active set.
    '''
    def __init__(self, dqn, feat_func, sample_func, kernel_func, train_func, eval_func,
                 init_setting=None, eta=1., sigma_n=0.01, K=10):
        '''
        train_func: atomic operation to train an agent on a task.

        eval_func: atomic operation to evaluate an agent on a task, and get a score.
        '''
        # some params.
        self.K = K
        self.sigma_n = sigma_n

        # components.
        self.kernel_func = kernel_func
        self.feat_func = feat_func
        self.sample_func = sample_func
        self.init_setting = init_setting
        self.train_func = train_func
        self.eval_func = eval_func

        # representation.
        self.task_im = {}
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
        all_settings = set()
        for task in tasks:
            all_settings.add(self.feat_func(task))
        all_settings = list(all_settings)

        for task in tasks:
            if task not in self.task_score:
                self.task_score[task] = self.eval_func(task)

        for ni in range(num_epochs):
            im_pred = {}
            im_sigma = {}
            im_ucb = {}

            if len(self.task_im) < 1:
                # select task based on prior.
                if not self.init_setting:
                    chosen_task = prob.choice(tasks, 1)
                else:
                    chosen_task = self.sample_func(self.init_setting)
            else:
                # select task based on GP.
                im = [(self.feat_func(task), im) for (task, im) in self.task_im.items()]

                # use Gaussian Process to estimate potential function.
                N = len(im)
                KXX = np.zeros((N, N))
                y = np.zeros(N)
                for (ti, (setting_i, im_i)) in enumerate(im):
                    for (tj, (setting_j, im_j)) in enumerate(im):
                        KXX[ti, tj] = self.kernel_func(setting_i, setting_j)
                for (ti, (setting_i, im_i)) in enumerate(im):
                    y[ti] = im_i

                M = len(all_settings)
                KXsX = np.zeros((M, N))
                KXsXs = np.zeros((M, M))
                for (ti, setting_i) in enumerate(all_settings):
                    for (tj, (setting_j, im_j)) in enumerate(im):
                        KXsX[ti, tj] = self.kernel_func(setting_i, setting_j)
                    KXsXs[ti, ti] = self.kernel_func(setting_i, setting_i)

                KXXinv = npla.inv(KXX + self.sigma_n**2 * np.eye(N))

                pred_mean = np.dot(KXsX, np.dot(KXXinv, y))
                pred_cov = KXsXs - np.dot(KXsX, np.dot(KXXinv, np.transpose(KXsX)))
                pred_sigma = np.sqrt(np.diag(pred_cov))

                for (ti, setting) in enumerate(all_settings):
                    im_pred[setting] = pred_mean[ti]
                    im_sigma[setting] = pred_sigma[ti]
                    im_ucb[setting] = pred_mean[ti] + self.eta * pred_sigma[ti]

                new_settings = sorted(all_settings, key=lambda setting: im_ucb[setting], reverse=True)
                new_setting = new_settings[0]

                chosen_task = self.sample_func(new_setting)

            self.train_func(chosen_task)

            for task in tasks:
                score = self.eval_func(task)
                self.task_im[task] = score - self.task_score[task]
                self.task_score[task] = score

            # collect diagnostics.
            self.diagnostics['task_im'] = self.task_im
            self.diagnostics['task_score'] = self.task_score
            self.diagnostics['pred'] = im_pred
            self.diagnostics['sigma'] = im_sigma
            self.diagnostics['ucb'] = im_ucb
            self.diagnostics['task'] = chosen_task
            self.diagnostics['setting'] = self.feat_func(chosen_task)


