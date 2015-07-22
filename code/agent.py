import numpy as np


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
        self.V = [(1. / self.num_states) for s in xrange(self.num_states)]

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

        return best_action

    def learn(self):
        ''' Performs value iteration on the MDP until convergence '''
        while True:
            # repeatedly perform the Bellman backup on each state
            # V_{i+1}(s) = \max_{a} \sum_{s' \in NS} T(s, a, s')[R(s, a, s') + \gamma V(s')]
            max_diff = 0.
            for state in xrange(self.num_states):
                poss_actions = self.mdp.get_allowed_actions(state)
                if len(poss_actions) == 0:
                    self.V[state] = 0.

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
    ''' Tabular TD-learning (Q-learning, SARSA, etc.) '''

    def __init__(self, options, environment):
        pass

    def get_action(self, state):
        # e-greedy exploration policy
        # use the estimate of Q-learning
        pass

    def learn(self, reward):
        # Q-learning updates?
        pass


class DQN(OnlineAgent):
    ''' Q-learning with a neural network function approximator '''

    def __init__(self, options, environment):
        pass

    def get_action(self, state):
        pass

    def learn(self, reward):
        pass


class ReinforceAgent(OnlineAgent):
    ''' Policy Gradient with a neural network function approximator '''

    def __init__(self, options, environment):
        pass

    def get_action(self, state):
        pass

    def learn(self, reward):
        pass
