class OnlineAgent(object):

    def get_action(self, state):
        raise NotImplementedError()

    def learn(self, reward):
        raise NotImplementedError()


class MDPSolver(object):

    def get_policy(self, state):
        raise NotImplementedError()

    def solve_mdp(self):
        raise NotImplementedError()


class ValueIterationSolver(MDPSolver):

    def __init__(self, options, mdp):
        pass

    def get_policy(self, state):
        pass

    def solve_mdp(self):
        ''' Performs value iteration on the MDP until convergence '''
        pass


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
