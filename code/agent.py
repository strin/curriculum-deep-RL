import random
import numpy as np
import theano
import theano.tensor as T


class OnlineAgent(object):

    def get_action(self, state):
        raise NotImplementedError()

    def learn(self, reward):
        raise NotImplementedError()


class ValueIterationSolver(OnlineAgent):

    def __init__(self, mdp, tol=1e-3):
        self.mdp = mdp
        self.num_states = mdp.get_num_states()
        self.gamma = mdp.gamma
        self.tol = tol

        # Tabular representation of state-value function initialized uniformly
        self.V = [1. for s in xrange(self.num_states)]

    def get_action(self, state):
        '''Returns the greedy action with respect to the current policy'''
        poss_actions = self.mdp.get_allowed_actions(state)

        # compute a^* = \argmax_{a} Q(s, a)
        best_action = None
        best_val = -float('inf')
        for action in poss_actions:
            ns_dist = self.mdp.next_state_distribution(state, action)

            val = 0.
            for ns, prob in ns_dist:
                val += prob * self.gamma * self.V[ns]

            if val > best_val:
                best_action = action
                best_val = val
            elif val == best_val and random.random() < 0.5:
                best_action = action
                best_val = val

        return best_action

    def learn(self):
        ''' Performs value iteration on the MDP until convergence '''
        iter_num = 0
        while True:
            print 'iteration', iter_num
            iter_num += 1
            # repeatedly perform the Bellman backup on each state
            # V_{i+1}(s) = \max_{a} \sum_{s' \in NS} T(s, a, s')[R(s, a, s') + \gamma V(s')]
            max_diff = 0.

            # TODO: Add priority sweeping
            for state in xrange(self.num_states):
                poss_actions = self.mdp.get_allowed_actions(state)

                # terminal state
                if len(poss_actions) == 0:
                    self.V[state] = 0.
                    continue

                best_val = -float('inf')
                for action in poss_actions:
                    val = 0.
                    ns_dist = self.mdp.next_state_distribution(state, action)
                    for ns, prob in ns_dist:
                        val += prob * (self.mdp.get_reward(state, action, ns) +
                                       self.gamma * self.V[ns])

                    if val > best_val:
                        best_val = val

                diff = abs(self.V[state] - best_val)
                self.V[state] = best_val

                if diff > max_diff:
                    max_diff = diff

            if max_diff < self.tol:
                break


class TdLearner(OnlineAgent):
    ''' Tabular TD-learning (Q-learning, SARSA, etc.)

        TODO: Add prioritized sweeping, planning, eligibility traces
    '''

    def __init__(self, task, update='q_learning', epsilon=0.05, alpha=0.1):
        # task related set-up
        self.task = task
        self.num_states = task.get_num_states()
        self.num_actions = task.get_num_actions()
        self.gamma = task.gamma

        # exploration policy
        self.epsilon = epsilon

        # learning rate
        self.alpha = alpha

        # Q-learning or SARSA
        if update not in ['q_learning', 'sarsa']:
            raise NotImplementedError()
        self.update = update

        # Tabular representation of the Q-function initialized uniformly
        self.Q = [[1 for a in xrange(task.get_allowed_actions(s))] for s in xrange(self.num_states)]

        # used for streaming updates
        self.s0 = None
        self.a0 = None
        self.r = None
        self.s1 = None
        self.a1 = None

    def _greedy_action(self, state):
        ''' a^* = argmax_{a} Q(s, a) '''
        q_vals = self.Q[state]
        return q_vals.index(max(q_vals))

    def get_action(self, state):
        poss_actions = self.task.get_allowed_actions()

        # e-greedy exploration policy
        if random.random() < self.epsilon:
            action = random.choice(poss_actions)
        else:
            action = self._greedy_action(state)

        self.s0 = self.s1
        self.a0 = self.a1
        self.s1 = state
        self.a1 = action

        return action

    def learn(self, reward):
        if(self.r is not None):
            if self.update == 'q_learning':
                next_action = self._greedy_action(self.s1)
            elif self.update == 'sarsa':
                # on-policy
                next_action = self.a1
            else:
                raise NotImplementedError()

            self._td_update(self.s0, self.a0, self.r, self.s1, next_action)

            # TODO: add planning here (sample experience from estimated model)

        self.r = reward

    def _td_update(self, s0, a0, r, s1, a1):
        #Q(s0, a0) + \alpha(r + \gamma Q(s1, a1) - Q(s0, a0))
        td_error = r + self.gamma * self.Q[s1][a1] - self.Q[s0][a0]
        self.Q[s0][a0] = self.Q[s0][a0] + self.alpha * td_error


class DQN(OnlineAgent):
    ''' Q-learning with a neural network function approximator

        TODO: Clean up the neural net code (make it more modular)
              Incorporate the online optimizer
                    - Tuning learning rates
                    - Regularization
                    - Gradient clipping (this is important for RL applications)
    '''

    def __init__(self, task, options, lr=1e-1):
        self.task = task
        self.state_dim = task.get_state_dimension()
        self.num_actions = task.get_num_actions()
        self.gamma = task.gamma

        # set-up the neural network. Simple 2-layer network for now
        self.dim_hidden = (self.state_dim + self.num_actions) / 2.

        # declare symbolic variables
        target = T.scalar('target')  # r + \gamma * \max_{a} Q(s, a)
        s = T.vector('s')
        a = T.scalar('a')
        W1 = theano.shared(0.1 * np.random.rand(self.dim_hidden, self.state_dim), name='W1')
        b1 = theano.shared(0.1 * np.random.rand(self.dim_hidden), name='b1')
        W2 = theano.shared(0.1 * np.random.rand(self.num_actions, self.dim_hidden), name='W2')
        b2 = theano.shared(0.1 * np.random.rand(self.num_actions), name='b2')

        # Construct expression graph
        layer1 = T.nnet.relu(W1.dot(s) + b1)
        q_vals = T.nnet.softmax(W2.dot(layer1) + b2)
        cost = (target - q_vals[a])**2

        params = [W1, b1, W2, b2]

        grads = T.grad(cost, params)

        print "Compiling fprop"
        self.fprop = theano.function(inputs=[s], outputs=[q_vals],
                                     name='fprop')

        print "Compiling bprop"
        updates = [(param_i, param_i - lr * gparam_i)
                   for param_i, gparam_i in zip(params, grads)]
        self.bprop = theano.function([s, a, target], updates=updates)

        # used for streaming updates
        self.s0 = None
        self.a0 = None
        self.r = None
        self.ns = None

    def _greedy_action(self, state):
        ''' a^* = argmax_{a} Q(s, a) '''
        q_vals = self.fprop(state)
        return q_vals.index(max(q_vals))

    def get_action(self, state):
        # epsilon greedy w.r.t the current policy
        if(random.random() < self.epsilon):
            action = random.randint(0, self.num_actions)
        else:
            action = self._greedy_action(state)

        self.s0 = self.ns
        self.a0 = self.na
        self.ns = state

        return action

    def learn(self, reward):
        if(self.r is not None):
            # max_{a'} Q(ns, a')
            next_qsa = np.max(self.fprop(self.ns))
            target = self.r + self.gamma * next_qsa

            # gradient descent on target - Q(s, a)
            self.bprop(self.s0, self.a0, target)

            # TODO: add experience replay mechanism here

        self.r = reward


class ReinforceAgent(OnlineAgent):
    ''' Policy Gradient with a neural network function approximator '''

    def __init__(self, options, task):
        pass

    def get_action(self, state):
        pass

    def learn(self, reward):
        pass
