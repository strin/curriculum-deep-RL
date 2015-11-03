import theano
import theano.tensor as T
import random
import numpy as np
import numpy.random as npr
import json

import pyrl.optimizers as optimizers
import pyrl.layers as layers
import pyrl.prob as prob
from pyrl.utils import Timer
from pyrl.tasks.task import Task
from pyrl.agents.agent import DQN
from pyrl.agents.agent import TabularVfunc

class MetaModelTabular(object):
    def __init__(self, bonus=1., decay=0.9):
        self.model = dict()
        self.bonus = bonus
        self.decay = decay

    def learn(self, feat, val):
        feat = json.dumps(feat)
        if feat not in self.model:
            self.model[feat] = self.bonus
        self.model[feat] *= self.decay
        self.model[feat] += val * (1- self.decay)
        print 'val', val

    def get(self, feat):
        feat = json.dumps(feat)
        if feat in self.model:
            val = self.model[feat]
        else:
            val = self.bonus
        return val

class DQCL_LocalEdit(object):
    '''
    DQCL = DeepQCurriculumLearning.
    learns a function E(task) that tracks average TD error.
    for each task, make local edits to it and sample based on E(task).
    '''
    def __init__(self, dqn, edit_func, feat_func, meta_model, gamma=0.95, l2_reg=0.0, lr=1e-3,
               memory_size=250, minibatch_size=64, epsilon=0.05):
        '''
        (TODO): task should be task info.
        we don't use all of task properties/methods here.
        only gamma and state dimension.
        and we allow task switching.
        '''
        self.dqn = dqn
        self.edit_func = edit_func
        self.feat_func = feat_func
        self.meta_model = meta_model
        self.l2_reg = l2_reg
        self.lr = lr
        self.epsilon = epsilon
        self.memory_size = memory_size
        self.minibatch_size = minibatch_size
        self.gamma = gamma

        # for now, keep experience as a list of tuples
        self.experience = []
        # add tags to each experience example.
        # for example, one tag could be {'task': task} to keep track of which task
        # the experience came from.
        self.experience_tags = []
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

    def _add_to_experience(self, s, a, ns, r, **tags):
        # TODO: improve experience replay mechanism by making it harder to
        # evict experiences with high td_error, for example
        # s, ns are state_vectors.
        if len(self.experience) < self.memory_size:
            self.experience.append((s, a, ns, r))
            self.experience_tags.append(tags)
        else:
            self.experience[self.exp_idx] = (s, a, ns, r)
            self.experience_tags[self.exp_idx] = tags
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

        states = [None] * self.minibatch_size
        next_states = [None] * self.minibatch_size
        actions = np.zeros(self.minibatch_size, dtype=int)
        rewards = np.zeros(self.minibatch_size)

        # sample and process minibatch
        samples = random.sample(self.experience, self.minibatch_size)
        terminals = []
        for idx, sample in enumerate(samples):
            state, action, next_state, reward = sample

            states[idx] = state
            actions[idx] = action
            rewards[idx] = reward

            if next_state is not None:
                next_states[idx] = next_state
            else:
                next_states[idx] = state
                terminals.append(idx)

        # convert states into tensor.
        states = np.array(states)
        next_states = np.array(next_states)

        # compute target reward + \gamma max_{a'} Q(ns, a')
        next_qvals = np.max(self.dqn.fprop(next_states), axis=1)

        # Ensure target = reward when NEXT_STATE is terminal
        next_qvals[terminals] = 0.

        targets = rewards + self.gamma * next_qvals

        self.bprop(states, actions, targets.flatten())

    def _learn(self, next_state, reward, **tags):
        self._add_to_experience(self.last_state, self.last_action,
                                next_state, reward, **tags)
        self._update_net()

    def _end_episode(self, reward, **tags):
        if self.last_state is not None:
            self._add_to_experience(self.last_state, self.last_action, None,
                                    reward, **tags)
        self.last_state = None
        self.last_action = None

    def _filter_experience_by_task(self, task):
        return [ex for (ex, tag) in zip(self.experience, self.experience_tags)
                if tag['task'] == task]

    def _average_td_error(self, ex_buffer):
        rewards = np.zeros(len(ex_buffer))
        q = np.zeros(len(ex_buffer))
        next_q = np.zeros(len(ex_buffer))

        for idx_offset in range(0, len(ex_buffer), self.minibatch_size):
            next_offset = min(idx_offset + self.minibatch_size, len(ex_buffer))
            states = []
            next_states = []
            terminals = []
            for idx in range(idx_offset, next_offset):
                state, action, next_state, reward = ex_buffer[idx]
                rewards[idx] = reward
                states.append(state)
                if next_state is not None:
                    next_states.append(next_state)
                else:
                    next_states.append(state)
                    terminals.append(idx)

            q[idx_offset:next_offset] = np.max(self.dqn.fprop(states), axis=1)
            next_q[idx_offset:next_offset] = np.max(self.dqn.fprop(next_states), axis=1)
            next_q[terminals] = 0.

        targets = rewards + self.gamma * next_q
        error = np.abs(q - targets)

        td = np.mean(error)
        return td

    def reset(self, task):
        self.last_task = task

    def run(self, task=None, num_epochs=10, num_episodes=100, tol=1e-4):
        if task:
            self.reset(task)

        task = self.last_task
        for ei in range(num_epochs):
            # run DQN on task for #episodes.
            self.run_task(task, num_episodes=num_episodes, tol=tol)
            task.reset()

            # compute average td error after learning.
            ex_buffer = self._filter_experience_by_task(task)
            td = self._average_td_error(ex_buffer)

            # learn the meta-model.
            feat = self.feat_func(task)
            self.meta_model.learn(feat, td)

            # sample a new task based on the meta-model.
            task_nb = self.edit_func(task)
            task_nb.append(task) # include this task.
            val_nb = []
            for new_task in task_nb:
                new_task_feat = self.feat_func(new_task)
                val_nb.append(self.meta_model.get(new_task_feat))
            print 'val_nb', val_nb

            log_prob = prob.normalize_log(np.array(val_nb) * 1.)
            p = np.exp(log_prob)
            print 'probability', p

            next_task = prob.choice(task_nb, 1, replace=True, p=p)[0]
            print 'new_task', next_task
            task = next_task

    def run_task(self, task, num_episodes=100, tol=1e-4):
        tags = {
            'task': task
        }
        total_steps = 0.
        for ei in range(num_episodes):
            task.reset()

            curr_state = task.curr_state

            num_steps = 0.
            while True:
                # TODO: Hack!
                if num_steps >= np.log(tol) / np.log(self.gamma):
                    # print 'Lying and tell the agent the episode is over!'
                    self._end_episode(0, **tags)
                    break

                action = self.dqn.get_action(curr_state, method='eps-greedy', epsilon=self.epsilon)
                self.last_state = curr_state
                self.last_action = action

                reward = task.step(action)
                next_state = task.curr_state

                if task.is_end():
                    self._end_episode(reward, **tags)
                    break
                else:
                    self._learn(next_state, reward, **tags)
                    curr_state = next_state

                num_steps += 1
                total_steps += 1
