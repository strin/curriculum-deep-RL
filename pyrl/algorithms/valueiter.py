# code for value iteration algorithms, such as Q-learning, SARSA, etc.
# this refractors the old dqn.py module by decoupling agent and algorithms.
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

class ValueIterationSolver(object):
    '''
    Vanilla value iteration for tabular environment
    '''
    def __init__(self, task, vfunc = None, tol=1e-3):
        self.task = task
        self.num_states = task.get_num_states()
        self.gamma = task.gamma
        self.tol = tol
        if vfunc:
            self.vfunc = vfunc
        else:
            self.vfunc = TabularVfunc(self.num_states)

    def get_action(self, state):
        '''Returns the greedy action with respect to the current policy'''
        poss_actions = self.task.get_allowed_actions(state)

        # compute a^* = \argmax_{a} Q(s, a)
        best_action = None
        best_val = -float('inf')
        for action in poss_actions:
            ns_dist = self.task.next_state_distribution(state, action)

            val = 0.
            for ns, prob in ns_dist:
                val += prob * self.gamma * self.vfunc(ns)

            if val > best_val:
                best_action = action
                best_val = val
            elif val == best_val and random.random() < 0.5:
                best_action = action
                best_val = val

        return best_action

    def learn(self):
        ''' Performs value iteration on the MDP until convergence '''
        while True:
            # repeatedly perform the Bellman backup on each state
            # V_{i+1}(s) = \max_{a} \sum_{s' \in NS} T(s, a, s')[R(s, a, s') + \gamma V(s')]
            max_diff = 0.

            # TODO: Add priority sweeping for state in xrange(self.num_states):
            for state in self.task.env.get_valid_states():
                poss_actions = self.task.get_allowed_actions(state)

                best_val = 0.
                for idx, action in enumerate(poss_actions):
                    val = 0.
                    ns_dist = self.task.next_state_distribution(state, action)
                    for ns, prob in ns_dist:
                        val += prob * (self.task.get_reward(state, action, ns) +
                                       self.gamma * self.vfunc(ns))

                    if(idx == 0 or val > best_val):
                        best_val = val

                diff = abs(self.vfunc(state) - best_val)
                self.vfunc.update(state, best_val)

                if diff > max_diff:
                    max_diff = diff

            if max_diff < self.tol:
                break


class Qlearn(object):
    def __init__(self, qfunc, gamma=0.95, alpha=1., epsilon=0.05):
        self.qfunc = qfunc
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.total_exp = 0

    def copy(self):
        # copy dqn.
        qfunc = self.qfunc.copy()
        learner = Qlearn(qfunc, gamma=self.gamma, alpha=self.alpha, epsilon=self.epsilon)
        learner.total_exp = self.total_exp
        return learner

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
                    break

                action = self.qfunc.get_action(curr_state, method='eps-greedy', epsilon=self.epsilon, valid_actions=task.valid_actions)
                reward = task.step(action)
                next_state = task.curr_state

                num_steps += 1
                total_steps += 1
                self.total_exp += 1

                self.qfunc.table[curr_state, action] *= (1 - self.alpha)
                if task.is_end():
                    self.qfunc.table[curr_state, action] += self.alpha * reward
                    break
                else:
                    self.qfunc.table[curr_state, action] += self.alpha * (reward
                                                + self.gamma * np.max(self.qfunc.table[next_state, :]))
                    curr_state = next_state

                if budget and num_steps >= budget:
                    break
        task.reset()


class QlearnReplay(object):
    '''
    traditional Qlearning except with experience replay.
    '''
    def __init__(self, qfunc, gamma=0.95, alpha=1., epsilon=0.05, memory_size=1000, minibatch_size=512):
        self.qfunc = qfunc
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.total_exp = 0

        self.memory_size = memory_size
        self.minibatch_size = minibatch_size
        self.experience = []
        self.exp_idx = 0

        # used for streaming updates
        self.last_state = None
        self.last_action = None

    def _add_to_experience(self, s, a, ns, r, nva):
        # TODO: improve experience replay mechanism by making it harder to
        # evict experiences with high td_error, for example
        # s, ns are state_vectors.
        # nva is a list of valid_actions at the next state.
        self.total_exp += 1
        if len(self.experience) < self.memory_size:
            self.experience.append((s, a, ns, r, nva))
        else:
            self.experience[self.exp_idx] = (s, a, ns, r, nva)
            self.exp_idx += 1
            if self.exp_idx >= self.memory_size:
                self.exp_idx = 0

    def _end_episode(self, reward):
        if self.last_state is not None:
            self._add_to_experience(self.last_state, self.last_action, None,
                                    reward, [])
            # self._update_net()
        self.last_state = None
        self.last_action = None

    def _learn(self, next_state, reward, next_valid_actions):
        '''
        need next_valid_actions to compute appropriate V = max_a Q(s', a).
        '''
        self._add_to_experience(self.last_state, self.last_action,
                                next_state, reward, next_valid_actions)

        samples = prob.choice(self.experience, self.minibatch_size, replace=True) # draw with replacement.

        for idx, sample in enumerate(samples):
            state, action, next_state, reward, nva = sample

            self.qfunc.table[state, action] *= (1 - self.alpha)

            if next_state is not None:
                self.qfunc.table[state, action] += self.alpha * (reward
                                            + self.gamma * np.max(self.qfunc.table[next_state, nva]))
            else:
                self.qfunc.table[state, action] += self.alpha * reward

    def copy(self):
        # copy dqn.
        qfunc = self.qfunc.copy()
        learner = Qlearn(qfunc, gamma=self.gamma, alpha=self.alpha, epsilon=self.epsilon)
        learner.total_exp = self.total_exp
        return learner

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
                    break

                action = self.qfunc.get_action(curr_state, method='eps-greedy', epsilon=self.epsilon, valid_actions=task.valid_actions)
                reward = task.step(action)
                next_state = task.curr_state

                self.last_state = curr_state
                self.last_action = action

                num_steps += 1
                total_steps += 1

                if task.is_end():
                    self._end_episode(reward)
                    break
                else:
                    self._learn(next_state, reward, task.valid_actions)
                    curr_state = next_state

                if budget and num_steps >= budget:
                    break
        task.reset()


class DeepQlearn(object):
    '''
    DeepMind's deep Q learning algorithms.
    '''
    def __init__(self, dqn_mt, gamma=0.95, l2_reg=0.0, lr=1e-3,
               memory_size=250, minibatch_size=64,
               nn_num_batch=1, nn_num_iter=2, regularizer={}):
        '''
        (TODO): task should be task info.
        we don't use all of task properties/methods here.
        only gamma and state dimension.
        and we allow task switching.
        '''
        self.dqn = dqn_mt
        self.l2_reg = l2_reg
        self.lr = lr
        self.memory_size = memory_size
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.regularizer = regularizer

        # for now, keep experience as a list of tuples
        self.experience = []
        self.exp_idx = 0
        self.total_exp = 0

        # used for streaming updates
        self.last_state = None
        self.last_action = None

        # params for nn optimization.
        self.nn_num_batch = nn_num_batch
        self.nn_num_iter = nn_num_iter

        # dianostics.
        self.diagnostics = {
            'nn-error': [] # training of neural network on mini-batches.
        }

        # compile back-propagtion network
        self._compile_bp()

    def copy(self):
        # copy dqn.
        dqn_mt = self.dqn.copy()
        learner = DeepQlearn(dqn_mt, self.gamma, self.l2_reg, self.lr, self.memory_size, self.minibatch_size)
        learner.experience = list(self.experience)
        learner.exp_idx = self.exp_idx
        learner.total_exp = self.total_exp
        learner.last_state = self.last_state
        learner.last_action = self.last_action
        learner._compile_bp()
        return learner

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

        reg_vs = []
        # mimic dqn regularizer.
        reg = self.regularizer.get('dqn-q')
        if reg:
            print '[compile-dqn] [regularizer] mimic dqn'
            dqn = reg['dqn']
            param = reg['param']
            print float(param) * self.minibatch_size / self.memory_size
            prior_action_values = T.matrix('prior_avs')
            reg_vs.append(prior_action_values)
            cost += float(param) * self.minibatch_size / self.memory_size * T.sqrt(T.mean((action_values - prior_action_values) ** 2))

        # back propagation.
        updates = optimizers.Adam(cost, params, alpha=self.lr)

        td_errors = T.sqrt(mse)
        self.bprop = theano.function(inputs=[states, last_actions, targets] + reg_vs,
                                     outputs=td_errors, updates=updates)

    def _add_to_experience(self, s, a, ns, r, meta):
        # TODO: improve experience replay mechanism by making it harder to
        # evict experiences with high td_error, for example
        # s, ns are state_vectors.
        # nva is a list of valid_actions at the next state.
        self.total_exp += 1
        if len(self.experience) < self.memory_size:
            self.experience.append((s, a, ns, r, meta))
        else:
            self.experience[self.exp_idx] = (s, a, ns, r, meta)
            self.exp_idx += 1
            if self.exp_idx >= self.memory_size:
                self.exp_idx = 0

    def _update_net(self):
        '''
            sample from the memory dataset and perform gradient descent on
            (target - Q(s, a))^2
        '''
        # don't update the network until sufficient experience has been
        # accumulated
        # removing this might cause correlation for early samples. suggested to be used in curriculums.
        #if len(self.experience) < self.memory_size:
        #    return
        for nn_bi in range(self.nn_num_batch):
            states = [None] * self.minibatch_size
            next_states = [None] * self.minibatch_size
            actions = np.zeros(self.minibatch_size, dtype=int)
            rewards = np.zeros(self.minibatch_size)
            nvas = []

            # sample and process minibatch
            # samples = random.sample(self.experience, self.minibatch_size) # draw without replacement.
            samples = prob.choice(self.experience, self.minibatch_size, replace=True) # draw with replacement.
            terminals = []
            for idx, sample in enumerate(samples):
                state, action, next_state, reward, meta = sample
                nva = meta['curr_valid_actions']

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

            if (targets > 100.).any():
                print 'error, target > 1', targets
                print 'rewards', rewards
                print 'next_vs', next_vs

            # regularizations.
            reg_vs = []
            reg = self.regularizer.get('dqn-q')
            if reg:
                dqn = reg['dqn']
                dqn_avs = dqn.fprop(states)
                reg_vs.append(dqn_avs)


            ## diagnostics.
            #print 'targets', targets
            #print 'next_qvals', next_qvals
            #print 'pure prop', self.dqn.fprop(states)
            #print 'prop', self.dqn.fprop(states)[range(states.shape[0]), actions]
            #print 'actions', actions
            nn_error = []
            for nn_it in range(self.nn_num_iter):
                error = self.bprop(states, actions, targets.flatten(), *reg_vs)
                nn_error.append(float(error))
            self.diagnostics['nn-error'].append(nn_error)

    def _learn(self, next_state, reward, next_valid_actions):
        '''
        need next_valid_actions to compute appropriate V = max_a Q(s', a).
        '''
        self._add_to_experience(self.last_state, self.last_action,
                                next_state, reward, next_valid_actions)
        self._update_net()

    def _end_episode(self, reward, meta):
        if self.last_state is not None:
            self._add_to_experience(self.last_state, self.last_action, None,
                                    reward, meta)
            # self._update_net()
        self.last_state = None
        self.last_action = None

    def run(self, task, num_episodes=100, tol=1e-4, budget=None, callback=None, **kwargs):
        '''
        task: the task to run on.
        num_episodes: how many episodes to repeat at maximum.
        tol: tolerance in terms of reward signal.
        budget: how many total steps to take.
        '''
        cum_rewards = []
        total_steps = 0.
        for ei in range(num_episodes):
            task.reset()

            curr_state = task.curr_state

            num_steps = 0.
            cum_reward = 0.
            while True:
                # TODO: Hack!
                meta = {}
                meta['last_valid_actions'] = task.valid_actions
                meta['num_actions'] = task.num_actions

                if num_steps >= np.log(tol) / np.log(self.gamma):
                    # print 'Lying and tell the agent the episode is over!'
                    meta['curr_valid_actions'] = []
                    self._end_episode(0, meta)
                    break

                action = self.dqn.get_action(curr_state, valid_actions=task.valid_actions, **kwargs)
                if 'uct' in kwargs: # update uct.
                    kwargs['uct'].visit(curr_state, action)

                self.last_state = curr_state
                self.last_action = action

                reward = task.step(action)
                cum_reward += reward

                meta['curr_valid_actions'] = task.valid_actions

                try:
                    next_state = task.curr_state
                    has_next_state = True
                except: # sessin has ended.
                    next_state = None
                    has_next_state = False

                num_steps += 1
                total_steps += 1

                # call diagnostics callback if provided.
                if callback:
                    callback(task)

                if task.is_end() or not has_next_state:
                    self._end_episode(reward, meta)
                    break
                else:
                    self._learn(next_state, reward, meta)
                    curr_state = next_state

                if budget and num_steps >= budget:
                    break
            cum_rewards.append(cum_reward)
        task.reset()
        return np.mean(cum_rewards)

def compute_tabular_value(task, tol=1e-4):
    solver = ValueIterationSolver(task, tol=tol)
    solver.learn()
    return solver.vfunc.V

def eval_tabular_value(task, func):
    V = np.zeros(task.get_num_states())
    for state in range(task.get_num_states()):
        V[state] = func(state)
    return V

def compute_tabular_values(tasks, num_cores = 8):
    ''' take a list of tabular tasks, and return states x tasks value matrix.
    '''
    vals = map(compute_tabular_value, tasks)
    return np.transpose(np.array(vals))
