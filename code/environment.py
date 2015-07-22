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

    def get_current_state(self):
        # return the current state
        raise NotImplementedError()

    def get_starting_state(self):
        # Note: initial state may be stochastic!
        raise NotImplementedError()

    def perform_action(self, action):
        # performs the specified action and updates the internal state of
        # the environment
        # returns the next state in the environment
        raise NotImplementedError()

    def reset(self):
        ''' Optional method for episodic tasks '''
        pass


class MDP(object):

    def __init__(self, environment, gamma):
        self.env = environment
        self.gamma = gamma

    def get_num_states(self):
        return self.env.get_num_states()

    def get_state_dimension(self):
        return self.env.get_state_dimension()

    def get_num_actions(self):
        return self.env.get_num_actions()

    def get_allowed_actions(self, state):
        return self.env.get_allowed_actions(state)

    def get_starting_state(self):
        return self.env.get_starting_state()

    def perform_action(self, action):
        return self.env.perform_action(action)

    def reset(self):
        self.env.reset()

    def get_gamma(self):
        return self.gamma

    def next_state_distribution(self, state, action):
        # returns a list of (next_state, probability) pairs based in s, a
        raise NotImplementedError()

    def get_reward(self, state, action, next_state):
        # returns the reward based on the (s, a, s') triple
        raise NotImplementedError()


class Task(object):
    ''' A task associates a reward function with an environment. The
        task mediates the agent's interaction with the underlying environment'''

    def __init__(self, environment, gamma):
        self.env = environment
        self.gamma = gamma

    def get_num_states(self):
        return self.env.get_num_states()

    def get_state_dimension(self):
        return self.env.get_state_dimension()

    def get_num_actions(self):
        return self.env.get_num_actions()

    def get_allowed_actions(self, state):
        return self.env.get_allowed_actions(state)

    def get_current_state(self):
        return self.env.get_current_state()

    def reset(self):
        self.env.reset()

    def get_gamma(self):
        return self.gamma()

    def get_reward(self, state, action, next_state):
        # returns the reward based on the (s, a, s') triple
        raise NotImplementedError()

    def perform_action(self, action):
        next_state = self.env.perform_action(action)
        reward = self.get_reward(next_state)
        return (next_state, reward)
