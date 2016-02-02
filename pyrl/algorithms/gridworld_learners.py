### warning: this module is very specialized to solving gridworld tasks.
from pyrl.common import *
import pyrl.optimizers as optimizers
import pyrl.layers as layers
import pyrl.prob as prob
from pyrl.utils import Timer
from pyrl.tasks.task import Task
from pyrl.agents.agent import DQN
from pyrl.agents.agent import TabularVfunc
from pyrl.algorithms.valueiter import DeepQlearn


class DeepQMultigoal(object):
    '''
    DeepMind's deep Q learning algorithms.
    '''
    def __init__(self, dqns, gamma=0.95, l2_reg=0.0, lr=1e-3,
               memory_size=250, minibatch_size=64, epsilon=0.05,
               nn_num_batch=1, nn_num_iter=2):
        '''
        (TODO): task should be task info.
        we don't use all of task properties/methods here.
        only gamma and state dimension.
        and we allow task switching.
        '''
        self.dqns = dqns
        self.l2_reg = l2_reg
        self.lr = lr
        self.epsilon = epsilon
        self.memory_size = memory_size
        self.minibatch_size = minibatch_size
        self.gamma = gamma

        # for now, keep experience as a list of tuples
        self.experiences = [] # one buffer for each dqn
        self.experience = []
        self.exp_idx = 0
        self.total_exp = 0

        # used for streaming updates
        self.last_state = None
        self.last_action = None
        self.last_phase = 0

        # params for nn optimization.
        self.nn_num_batch = nn_num_batch
        self.nn_num_iter = nn_num_iter

        # dianostics.
        self.diagnostics = {
            'nn-error': [] # training of neural network on mini-batches.
        }

        # compile back-propagtion network
        self.bprops = []
        for self.dqn in self.dqns:
            self._compile_bp()
            self.bprops.append(self.bprop)
            self.experiences.append([])

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

    def _add_to_experience(self, p, s, a, np, ns, r, nva):
        # TODO: improve experience replay mechanism by making it harder to
        # evict experiences with high td_error, for example
        # s, ns are state_vectors.
        # nva is a list of valid_actions at the next state.
        self.total_exp += 1
        if len(self.experience) < self.memory_size:
            self.experience.append((p, s, a, np, ns, r, nva))
        else:
            self.experience[self.exp_idx] = (p, s, a, np, ns, r, nva)
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
            next_phases = [None] * self.minibatch_size
            actions = np.zeros(self.minibatch_size, dtype=int)
            rewards = np.zeros(self.minibatch_size)
            cross_phase_idx = []
            nvas = []

            # sample and process minibatch
            # samples = random.sample(self.experience, self.minibatch_size) # draw without replacement.
            samples = prob.choice(self.experience, self.minibatch_size, replace=True) # draw with replacement.
            terminals = []
            for idx, sample in enumerate(samples):
                phase, state, action, next_phase, next_state, reward, nva = sample

                states[idx] = state
                actions[idx] = action
                rewards[idx] = reward
                next_phases[idx] = next_phase

                nvas.append(nva)

                if phase != next_phase:
                    cross_phase_idx.append(idx)

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
                    if idx in cross_phase_idx:
                        next_qval = self.dqns[next_phases[idx]].fprop(np.array([next_states[idx]]))
                        next_vs[idx] = np.max(next_qval[0, nvas[idx]])
                    else:
                        next_vs[idx] = np.max(next_qvals[idx, nvas[idx]])

            targets = rewards + self.gamma * next_vs

            ## diagnostics.
            #print 'targets', targets
            #print 'next_qvals', next_qvals
            #print 'pure prop', self.dqn.fprop(states)
            #print 'prop', self.dqn.fprop(states)[range(states.shape[0]), actions]
            #print 'actions', actions
            nn_error = []
            for nn_it in range(self.nn_num_iter):
                error = self.bprop(states, actions, targets.flatten())
                nn_error.append(float(error))
            self.diagnostics['nn-error'].append(nn_error)

    def _learn(self, next_phase, next_state, reward, next_valid_actions):
        '''
        need next_valid_actions to compute appropriate V = max_a Q(s', a).
        '''
        self._add_to_experience(self.last_phase, self.last_state, self.last_action,
                                next_phase, next_state, reward, next_valid_actions)
        self._update_net()

    def _end_episode(self, reward):
        if self.last_state is not None:
            self._add_to_experience(self.last_phase, self.last_state, self.last_action,  -1, None,
                                    reward, [])
            # self._update_net()
        self.last_state = None
        self.last_action = None
        self.last_phase = 0

    def run(self, task, num_episodes=100, tol=1e-4, budget=None, test=False, callback=None):
        '''
        task: the task to run on.
        num_episodes: how many episodes to repeat at maximum.
        tol: tolerance in terms of reward signal.
        budget: how many total steps to take.
        '''
        total_steps = 0.
        avg_reward = 0
        for ei in range(num_episodes):
            task.reset()

            curr_state = task.curr_state

            num_steps = 0.
            while True:
                # TODO: Hack!
                if num_steps >= np.log(tol) / np.log(self.gamma):
                    # print 'Lying and tell the agent the episode is over!'
                    self._end_episode(0)
                    break

                self.dqn = self.dqns[task.phase]
                self.bprop = self.bprops[task.phase]
                self.experience = self.experiences[task.phase]

                action = self.dqn.get_action(curr_state, method='eps-greedy', epsilon=self.epsilon, valid_actions=task.valid_actions)
                self.last_state = curr_state
                self.last_action = action
                self.last_phase = task.phase

                reward = task.step(action)
                avg_reward += reward / float(num_episodes)

                if reward > 0:
                    print 'phase', task.phase

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
                    callback(curr_state, action, next_state, reward)

                if not test:
                    if task.is_end() or not has_next_state:
                        self._end_episode(reward)
                        break
                    else:
                        self._learn(task.phase, next_state, reward, task.valid_actions)
                        curr_state = next_state

                if budget and num_steps >= budget:
                    break
        task.reset()
        return avg_reward
