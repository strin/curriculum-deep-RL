# code for value iteration algorithms, such as Q-learning, SARSA, etc.
# this refractors the old dqn.py module by decoupling agent and algorithms.
import theano
import theano.tensor as T
import random
import numpy as np
import numpy.random as npr

import pyrl.optimizers
import pyrl.layers
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
            for state in xrange(self.num_states):
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

class DeepQlearn(object):
    '''
    DeepMind's deep Q learning algorithms.
    '''
    def __init__(self, task, dqn, l2_reg=0.0, lr=1e-3,
               memory_size=250, minibatch_size=64):
        self.dqn = dqn
        self.l2_reg = l2_reg
        self.lr = lr
        self.memory_size = memory_size
        self.minibatch_size = minibatch_size

        # for now, keep experience as a list of tuples
        self.experience = []
        self.exp_idx = 0

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

    def _add_to_experience(self, s, a, ns, r):
        # TODO: improve experience replay mechanism by making it harder to
        # evict experiences with high td_error, for example
        if len(self.experience) < self.memory_size:
            self.experience.append((s, a, ns, r))
        else:
            self.experience[self.exp_idx] = (s, a, ns, r)
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
        if len(self.experience) < self.memory_size:
            return

        states = np.zeros((self.minibatch_size, self.state_dim,))
        next_states = np.zeros((self.minibatch_size, self.state_dim))
        actions = np.zeros(self.minibatch_size, dtype=int)
        rewards = np.zeros(self.minibatch_size)

        # sample and process minibatch
        samples = random.sample(self.experience, self.minibatch_size)
        terminals = []
        for idx, sample in enumerate(samples):
            state, action, next_state, reward = sample
            states[idx, :] = state.reshape(-1)
            actions[idx] = action
            rewards[idx] = reward

            if next_state is not None:
                next_states[idx, :] = next_state.reshape(-1)
            else:
                terminals.append(idx)

        # compute target reward + \gamma max_{a'} Q(ns, a')
        next_qvals = np.max(self.dqn.apply(next_states), axis=1)

        # Ensure target = reward when NEXT_STATE is terminal
        next_qvals[terminals] = 0.

        targets = rewards + self.gamma * next_qvals

        self.bprop(states, actions, targets.flatten())

    def _learn(self, next_state, reward):
        self._add_to_experience(self.last_state, self.last_action,
                                next_state, reward)
        self._update_net()

    def _end_episode(self, reward):
        if self.last_state is not None:
            self._add_to_experience(self.last_state, self.last_action, None,
                                    reward)
        self.last_state = None
        self.last_action = None

    def run(self, num_episodes = 1):
        task = self.task
        agent = self.dqn

        for episode in xrange(num_episodes):
            while task.is_terminal():
                task.reset()

            curr_state = task.get_current_state()

            num_steps = 0.
            while True:
                # TODO: Hack!
                if num_steps >= 200:
                    # print 'Lying and tell the agent the episode is over!'
                    agent.end_episode(0)
                    num_steps = 0.

                action = agent.get_action(curr_state)
                next_state, reward = task.perform_action(action)

                if task.is_terminal():
                    self._end_episode(reward)
                    break
                else:
                    self._learn(next_state, reward)
                    curr_state = next_state

                num_steps += 1








