import random
import numpy as np
import theano
import theano.tensor as T


class OnlineAgent(object):

    def get_action(self, state):
        raise NotImplementedError()

    def learn(self, next_state, reward):
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
        while True:
            # repeatedly perform the Bellman backup on each state
            # V_{i+1}(s) = \max_{a} \sum_{s' \in NS} T(s, a, s')[R(s, a, s') + \gamma V(s')]
            max_diff = 0.

            # TODO: Add priority sweeping
            for state in xrange(self.num_states):
                poss_actions = self.mdp.get_allowed_actions(state)

                best_val = 0
                for idx, action in enumerate(poss_actions):
                    val = 0.
                    ns_dist = self.mdp.next_state_distribution(state, action)
                    for ns, prob in ns_dist:
                        val += prob * (self.mdp.get_reward(state, action, ns) +
                                       self.gamma * self.V[ns])

                    if(idx == 0 or val > best_val):
                        best_val = val

                diff = abs(self.V[state] - best_val)
                self.V[state] = best_val

                if diff > max_diff:
                    max_diff = diff

            if max_diff < self.tol:
                break


class TDLearner(OnlineAgent):
    ''' Tabular TD-learning (Q-learning, SARSA, etc.)

        TODO: Add prioritized sweeping, planning, eligibility traces
              To add planning:
                1) Store a model of experiences (either store samples or an
                   ML estimate of the dynamics + rewards)
                2) On each update in learn, sample up to N experiences and
                   use them to perform updates
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
        self.Q = [[1. for a in task.get_allowed_actions(s)] for s in xrange(self.num_states)]

        # used for streaming updates
        self.last_state = None
        self.last_action = None

    def reset_episode(self):
        self.last_state = None
        self.last_action = None

    def _sample_from_policy(self, state):
        '''epsilon-greedy exploration policy '''
        poss_actions = self.task.get_allowed_actions(state)

        if len(poss_actions) == 0:
            return None

        if random.random() < self.epsilon:
            return random.choice(poss_actions)

        # a^* = argmax_{a} Q(s, a)
        q_vals = self.Q[state]
        return q_vals.index(max(q_vals))

    def get_action(self, state):
        action = self._sample_from_policy(state)
        self.last_state = state
        self.last_action = action
        return action

    def _td_update(self, s0, a0, r, s1):
        if self.update == 'q_learning':
            # Q(s0, a0) + \alpha(r + \gamma \max_{a} Q(s1, a) - Q(s0, a0))
            qmax = 0.
            for idx, qval in enumerate(self.Q[s1]):
                if(idx == 0 or qval > qmax):
                    qmax = qval
            td_error = r + self.gamma * qmax - self.Q[s0][a0]
        elif self.update == 'sarsa':
            # Q(s0, a0) + \alpha(r + \gamma Q(s1, a1) - Q(s0, a0))
            a1 = self._sample_from_policy(s1)

            if a1 is None:  # handle terminal states
                td_error = r - self.Q[s0][a0]
            else:
                td_error = r + self.gamma * self.Q[s1][a1] - self.Q[s0][a0]
        else:
            raise NotImplementedError()
        self.Q[s0][a0] += self.alpha * td_error

    def learn(self, next_state, reward):
        self._td_update(self.last_state, self.last_action, reward, next_state)

        # TODO: Add planning HERE;


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
