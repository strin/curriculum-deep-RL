# code for multi task RL algorithms.
import theano
import theano.tensor as T
import random
import numpy as np
import numpy.random as npr
import numpy.linalg as npla

import pyrl.optimizers as optimizers
import pyrl.layers as layers
import pyrl.prob as prob
from pyrl.algorithms.valueiter import DeepQlearn
from pyrl.agents.agent import eval_policy_reward
from pyrl.evaluate import eval_dataset, expected_reward_tabular_normalized

class SingleLearnerSequential(object):
    def __init__(self, dqn, tasks, **kwargs):
        self.dqn = dqn
        self.tasks = tasks
        self.num_tasks = len(self.tasks)
        self.deepQlearn = DeepQlearn(tasks[0], dqn, **kwargs)
        self.t = 0

    def run(self, num_epochs=1, budget=100):
        for ei in range(num_epochs):
            ti = self.t % self.num_tasks
            task = self.tasks[ti]

            # (TODO) this breaks away the abstraction.
            self.deepQlearn.task = task
            self.dqn.task = task
            # run training.
            self.deepQlearn.run(budget, task)
            self.t += 1
            for it in range(budget):
                self.deepQlearn._update_net()

class SingleLearnerRandom(object):
    def __init__(self, dqn, tasks, **kwargs):
        self.dqn = dqn
        self.tasks = tasks
        self.num_tasks = len(self.tasks)
        self.deepQlearn = DeepQlearn(tasks[0], dqn, **kwargs)
        self.last_task_ti = 0

    def run(self, num_epochs=1, num_episodes=1):
        for ei in range(num_epochs):
            ti = npr.choice(range(self.num_tasks), 1)[0]
            self.last_task_ti = ti
            task = self.tasks[ti]

            # (TODO) this breaks away the abstraction.
            self.deepQlearn.task = task
            self.dqn.task = task
            # run training.
            self.deepQlearn.run(num_episodes, task)

class SingleLearnerCommunist(object):
    def __init__(self, dqn, tasks, **kwargs):
        self.dqn = dqn
        self.tasks = tasks
        self.num_tasks = len(self.tasks)
        self.deepQlearn = DeepQlearn(tasks[0], dqn, **kwargs)
        self.last_task_ti = 0

    def run(self, num_epochs=1, num_episodes=1):
        for ei in range(num_epochs):
            task_performance = np.zeros(self.num_tasks)
            for ti in range(self.num_tasks):
                task_performance[ti] = expected_reward_tabular_normalized(self.dqn, self.tasks[ti], tol=1e-4)
            ti = np.argmin(task_performance)
            task = self.tasks[ti]
            self.last_task_ti = ti

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

class DeepQlearnMT(object):
    '''
    A multi-task version of DeepMind's Deep Q Learning Algorithm
    In this version, each task has a seperate experience buffer, and
    backpropagation is based on experiences sampled from all buffers.
    '''
    def __init__(self, dqn_mt, gamma=0.95, l2_reg=0.0, lr=1e-3,
               memory_size=250, minibatch_size=64, epsilon=0.05, update_strategy=None):
        '''
        (TODO): task should be task info.
        we don't use all of task properties/methods here.
        only gamma and state dimension.
        and we allow task switching.
        '''
        self.dqn = dqn_mt
        self.l2_reg = l2_reg
        self.lr = lr
        self.epsilon = epsilon
        self.memory_size = memory_size
        self.minibatch_size = minibatch_size
        self.update_strategy = update_strategy
        self.gamma = gamma

        # for now, keep experience as a list of tuples
        self.ex_task = {}
        self.total_exp_by_task = {}
        self.ex_id = {}
        self.total_exp = 0

        # used for streaming updates
        self.last_state = None
        self.last_action = None

        # compile back-propagtion network
        self._compile_bp()

    def _compile_bp(self):
        states = self.dqn.states
        action_values = self.dqn.action_values
        params = self.dqn.params
        targets = T.vector('target')
        last_actions = T.lvector('action')

        # loss function.
        mse = layers.MSE(action_values[T.arange(action_values.shape[0]),
                            last_actions], targets)
        # l2 penalty.
        l2_penalty = 0.
        for param in params:
            l2_penalty += (param ** 2).sum()

        cost = mse + self.l2_reg * l2_penalty

        # back propagation.
        updates = optimizers.Adam(cost, params, alpha=self.lr)

        td_errors = T.sqrt(mse)
        self.bprop = theano.function(inputs=[states, last_actions, targets],
                                     outputs=td_errors, updates=updates)

    def _add_to_experience(self, task, s, a, ns, r, nva):
        # TODO: improve experience replay mechanism by making it harder to
        # evict experiences with high td_error, for example
        # s, ns are state_vectors.
        # nva is a list of valid_actions at the next state.
        if task not in self.ex_task:
            self.ex_task[task] = []
            self.ex_id[task] = 0
            self.total_exp_by_task[task] = 0
        self.total_exp += 1
        self.total_exp_by_task[task] += 1
        experience = self.ex_task[task]
        if len(experience) < self.memory_size:
            experience.append((s, a, ns, r, nva))
        else:
            experience[self.ex_id[task]] = (s, a, ns, r, nva)
            self.ex_id[task] += 1
            if self.ex_id[task] >= self.memory_size:
                self.ex_id[task] = 0

    def _update_net(self, task):
        '''
            sample from the memory dataset and perform gradient descent on
            (target - Q(s, a))^2
        '''
        #if self.total_exp_by_task[task] < self.memory_size:
        #    return

        # merge experience buffer.
        experience = []
        for task in self.ex_task:
            experience.extend(self.ex_task[task])

        num_iter = 1
        if self.update_strategy:
            if self.update_strategy == 'task':
                num_iter = len(self.ex_task)

        for it in range(num_iter):
            # don't update the network until sufficient experience has been
            # accumulated
            states = [None] * self.minibatch_size
            next_states = [None] * self.minibatch_size
            actions = np.zeros(self.minibatch_size, dtype=int)
            rewards = np.zeros(self.minibatch_size)
            nvas = []

            # sample and process minibatch
            # samples = random.sample(self.experience, self.minibatch_size) # draw without replacement.
            samples = prob.choice(experience, self.minibatch_size, replace=True) # draw with replacement.
            terminals = []

            for idx, sample in enumerate(samples):
                state, action, next_state, reward, nva = sample

                states[idx] = state
                actions[idx] = action
                rewards[idx] = reward
                nvas.append(nva)

                if next_state is not None:
                    next_states[idx] = next_state
                else:
                    next_states[idx] = state
                    terminals.append(idx)

            # convert states into tensor.
            states = np.array(states)
            next_states = np.array(next_states)

            # compute target reward + \gamma max_{a'} Q(ns, a')
            # Ensure target = reward when NEXT_STATE is terminal
            next_qvals = self.dqn.fprop(next_states)
            next_vs = np.zeros(self.minibatch_size)
            for idx in range(self.minibatch_size):
                if idx not in terminals:
                    next_vs[idx] = np.max(next_qvals[idx, nvas[idx]])

            targets = rewards + self.gamma * next_vs

            ## diagnostics.
            #print 'targets', targets
            #print 'next_qvals', next_qvals
            #print 'pure prop', self.dqn.fprop(states)
            #print 'prop', self.dqn.fprop(states)[range(states.shape[0]), actions]
            #print 'actions', actions
            #for it in range(10):
            error = self.bprop(states, actions, targets.flatten())
            #    print 'error', error

    def _learn(self, task, next_state, reward, next_valid_actions):
        '''
        need next_valid_actions to compute appropriate V = max_a Q(s', a).
        '''
        self._add_to_experience(task, self.last_state, self.last_action,
                                next_state, reward, next_valid_actions)
        self._update_net(task)

    def _end_episode(self, task, reward):
        if self.last_state is not None:
            self._add_to_experience(task, self.last_state, self.last_action, None,
                                    reward, [])
        self.last_state = None
        self.last_action = None

    def run(self, task, num_episodes=100, tol=1e-4, budget=None):
        '''
        task: the task to run on.
        num_episodes: how many episodes to repeat at maximum.
        tol: tolerance in terms of reward signal.
        budget: how many total steps to take.
        '''
        total_steps = 0.
        for ei in range(num_episodes):
            task.reset()

            curr_state = task.curr_state

            num_steps = 0.
            while True:
                # TODO: Hack!
                if num_steps >= np.log(tol) / np.log(self.gamma):
                    # print 'Lying and tell the agent the episode is over!'
                    self._end_episode(task, 0)
                    break

                action = self.dqn.get_action(curr_state, method='eps-greedy', epsilon=self.epsilon, valid_actions=task.valid_actions)
                self.last_state = curr_state
                self.last_action = action

                reward = task.step(action)
                next_state = task.curr_state

                num_steps += 1
                total_steps += 1

                if task.is_end():
                    self._end_episode(task, reward)
                    break
                else:
                    self._learn(task, next_state, reward, task.valid_actions)
                    curr_state = next_state

                if budget and num_steps >= budget:
                    break
        task.reset()

