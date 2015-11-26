# implements policy distllation
# DeepMind paper ICLR 2016.
import theano
import theano.tensor as T
import numpy as np

import pyrl.optimizers as optimizers
import pyrl.prob as prob
from pyrl.layers import SoftMax

class DeepDistill(object):
    def __init__(self, dqn_mt, memory_size=128, lr=1e-3, l2_reg=0., minibatch_size=128):
        '''
        '''
        self.memory_size = memory_size
        self.l2_reg = l2_reg
        self.minibatch_size = minibatch_size
        self.lr = lr

        # experience is a dict: task -> experience buffer.
        # an experience buffer is a list of tuples (s, q, va)
        # s = state, q = list of normalized qvals, va = corresponding valid actions.
        self.experience = {}
        self.ex_id = {}
        self.total_exp = 0
        self.dqn_mt = dqn_mt

        self._compile_bp()


    def _compile_bp(self):
        states = self.dqn_mt.states
        params = self.dqn_mt.params
        targets = T.matrix('target')
        is_valid = T.matrix('is_valid')

        # compute softmax for action_values
        # numerical stability in mind.
        action_values = self.dqn_mt.action_values
        action_values -= (1 - is_valid) * 1e10
        action_values_softmax = SoftMax(action_values)
        action_values_softmax += (1 - is_valid)

        # loss function: KL-divergence.
        kl = T.sum(targets * is_valid * (T.log(targets) - T.log(action_values_softmax)))

        # l2 penalty.
        l2_penalty = 0.
        for param in params:
            l2_penalty += (param ** 2).sum()

        cost = kl + self.l2_reg * l2_penalty

        updates = optimizers.Adam(cost, params, alpha=self.lr)
        self.bprop = theano.function(inputs=[states, targets, is_valid],
                                     outputs=kl / states.shape[0], updates=updates)

    def collect(self, dqn, task, temperature, num_episodes=1, budget=300):
        '''
        online data collection process.

        Parameters
        ==========
        dqn (DQN):
            deep Q-network

        task (Task):
            the target task to collect experiences on.

        num_episodes:
            number of episodes to collect

        budget:
            the maximum number of steps for each episode.
        '''
        if task not in self.experience:
            self.experience[task] = []
            self.ex_id[task] = 0

        exbuff = self.experience[task]

        for ni in range(num_episodes):
            task.reset()

            num_steps = 0

            while not task.is_end() and num_steps < budget:
                curr_state = task.curr_state
                valid_actions = task.valid_actions
                probs = dqn._get_softmax_action_distribution(curr_state, temperature=temperature, valid_actions=valid_actions)

                is_valid = np.zeros(task.num_actions)
                is_valid[valid_actions] = 1.

                full_probs = np.zeros(task.num_actions)
                full_probs[valid_actions] = probs

                ex = (curr_state, full_probs, is_valid)

                if len(exbuff) >= self.memory_size:
                    exbuff[self.ex_id[task] % self.memory_size] = ex
                    self.ex_id[task] += 1
                else:
                    exbuff.append(ex)

                self.total_exp += 1

                # action = dqn.get_action(curr_state, method='eps-greedy', epsilon=0.05, valid_actions=task.valid_actions)
                action = dqn.get_action(curr_state, method='softmax', temperature=temperature, valid_actions=task.valid_actions)

                task.step(action)
                num_steps += 1

        task.reset()

    def train(self, num_iter=100):
        '''
        supervised learning on the experience buffer.
        '''
        states = [None] * self.minibatch_size
        is_valids = [None] * self.minibatch_size
        probs = [None] * self.minibatch_size

        experience = sum(self.experience.values(), [])

        for it in range(num_iter):
            samples = prob.choice(experience, self.minibatch_size, replace=True)
            for idx, sample in enumerate(samples):
                state, p, is_valid = sample

                states[idx] = state
                is_valids[idx] = is_valid
                probs[idx] = p

            # convert into numpy array.
            states = np.array(states)
            is_valids = np.array(is_valids)
            probs = np.array(probs)

            error = self.bprop(states, probs, is_valids)
            print 'error', error









