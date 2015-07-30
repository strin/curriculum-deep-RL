import random
import numpy as np
import theano
import theano.tensor as T
import layers


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
        self.Q = [[0.5 for a in task.get_allowed_actions(s)] for s in xrange(self.num_states)]

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

        TODO:       - Add experience replay
                            - This will speed up the model since we can
                               batch updates together for the forward
                               and backward passes
                                - Also, if we don't clone the next, then there
                                  is no need to 2 do separate passes
                    - Incorporate the online optimizer
                    - Tuning learning rates
                    - Regularization
                    - Gradient clipping (this is important for RL applications)
    '''

    def __init__(self, task, hidden_dim=128, l2_reg=0.0, lr=1e-1, epsilon=0.05):
        self.task = task
        self.state_dim = task.get_state_dimension()
        self.num_actions = task.get_num_actions()
        self.gamma = task.gamma
        self.hidden_dim = hidden_dim
        self.l2_reg = l2_reg
        self.lr = lr
        self.epsilon = epsilon

        self._initialize_net()

        # used for streaming updates
        self.last_state = None
        self.last_action = None

    def _initialize_net(self):
        # simple 2 layer net with l2-loss
        state = T.col('state')
        hidden_layer1 = layers.FullyConnected(inputs=state,
                                              input_dim=self.state_dim,
                                              output_dim=self.hidden_dim,
                                              activation='relu')

        hidden_layer2 = layers.FullyConnected(inputs=hidden_layer1.output,
                                              input_dim=self.hidden_dim,
                                              output_dim=self.hidden_dim,
                                              activation='relu')

        linear_layer = layers.FullyConnected(inputs=hidden_layer2.output,
                                             input_dim=self.hidden_dim,
                                             output_dim=self.num_actions,
                                             activation=None)

        action_values = linear_layer.output

        target = T.scalar('target')
        last_action = T.lscalar('action')
        MSE = layers.MSE(action_values[last_action], target)

        params = hidden_layer1.params + hidden_layer2.params + linear_layer.params

        l2_penalty = 0.
        for param in params:
            l2_penalty += (param ** 2).sum()

        cost = MSE.output + self.l2_reg * l2_penalty

        grads = T.grad(cost, params)

        # vanilla SGD update
        updates = [(param, param - self.lr * gparam) for param, gparam
                   in zip(params, grads)]

        print "Compiling fprop"
        self.fprop = theano.function(inputs=[state], outputs=[action_values],
                                     name='fprop')

        # takes a single gradient step
        print "Compiling backprop"
        self.bprop = theano.function(inputs=[state, last_action, target],
                                     outputs=[cost], updates=updates)

    def reset_episode(self):
        self.last_state = None
        self.last_action = None

    def _greedy_action(self, state):
        ''' a^* = argmax_{a} Q(s, a) '''
        q_vals = self.fprop(state)
        return q_vals.index(max(q_vals))

    def get_action(self, state):
        # epsilon greedy w.r.t the current policy
        if(random.random() < self.epsilon):
            action = random.randint(0, self.num_actions - 1)
        else:
            action = self._greedy_action(state)

        self.last_state = state
        self.last_action = action

        return action

    def learn(self, next_state, reward):
        # max_{a'} Q(ns, a')
        next_qsa = np.max(self.fprop(next_state))
        target = reward + self.gamma * next_qsa

        # gradient descent on target - Q(s, a)
        self.bprop(self.last_state, self.last_action, target)

        # TODO: add experience replay mechanism here


class ReinforceAgent(OnlineAgent):
    ''' Policy Gradient with a neural network function approximator '''

    def __init__(self, options, task):
        pass

    def get_action(self, state):
        pass

    def learn(self, reward):
        pass
