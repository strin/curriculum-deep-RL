class Environment(object):

    def get_num_states(self):
        # returns the number of state for a tabular representation
        raise NotImplementedError()

    def get_state_dimension(self):
        # returns the dimension of each observation for use with a neural
        # network function approximator
        raise NotImplementedError()

    def get_num_actions(self):
        # returns the cardinality of the action set
        raise NotImplementedError()

    def get_allowed_actions(self, state):
        # optional method to return a subset of the actions based on the
        # current state
        return self.get_max_allowed_actions()

    def get_starting_state(self):
        # return the initial state (this may be stochastic)
        raise NotImplementedError()

    def perform_action(self, action):
        # performs the specified action and updates the internal state of
        # the environment
        # returns the next state in the environment
        raise NotImplementedError()


class MDP(Environment):

    def next_state_distribution(self, state, action):
        # returns a list of (next_state, probability) pairs based in s, a
        raise NotImplementedError()

    def get_reward(self, state, action, next_state):
        # returns the reward based on the (s, a, s') triple
        raise NotImplementedError()


class Task(object):
    ''' A task associates a reward function with an environment. The
        task mediates the agent's interaction with the underlying environment'''

    def __init__(self, environment):
        self.env = environment

    def perform_action(self, action):
        # performs ACTION on the underlying environment, computes the
        # reward for the observed transition, and returns a
        # (reward, next_state) pair
        raise NotImplementedError()
