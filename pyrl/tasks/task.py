class Environment(object):
    '''
    An Environment defines initial states and transition distributions
    '''
    def get_num_actions(self):
        # returns the cardinality of the action set
        raise NotImplementedError()

    def get_allowed_actions(self, state):
        # Return a subset of the actions based on the current state
        raise NotImplementedError()

    def get_current_state(self):
        # return the current state
        raise NotImplementedError()

    def get_current_state_vector(self):
        # get a vector representation of the current state.
        raise NotImplementedError()

    def get_start_state(self):
        # Note: initial state may be stochastic!
        return self.get_current_state()

    def perform_action(self, action):
        # performs the specified action and updates the internal state of
        # the environment
        # returns the next state in the environment
        raise NotImplementedError()

    def reset(self):
        ''' Optional method for episodic tasks '''
        pass

    def support_tabular(self):
        ''' tabular support is usually off when the representation high-dimensional '''
        return True

    def get_num_states(self):
        # returns the number of state for a tabular representation
        raise NotImplementedError()

    def get_state_dimension(self):
        # returns the dimension of each observation for use with a neural
        # network function approximator
        raise NotImplementedError()

class Task(object):
    ''' A task associates a reward function with an environment. The
        task mediates the agent's interaction with the underlying environment'''

    def __init__(self, environment, gamma):
        self.env = environment
        self.__dict__.update(self.env.__dict__) # populate namespace with those from env.
        self.gamma = gamma

    def get_gamma(self):
        return self.gamma()

    def is_terminal(self):
        # Is the environment in a terminal state, i.e. are there no possible
        # successors
        state = self.env.get_current_state()
        poss_actions = self.get_allowed_actions(state)
        return len(poss_actions) == 0

    def perform_action(self, action):
        curr_state = self.env.get_current_state()
        next_state = self.env.perform_action(action)
        reward = self.get_reward(curr_state, action, next_state)
        return (next_state, reward)

    def reset(self):
        # resets the current state to the starting state
        raise NotImplementedError()

    def get_reward(self, state, action, next_state):
        # returns the reward based on the (s, a, s') triple
        raise NotImplementedError()

    def visualize(self):
        '''
            Returns a matrix that can visualize the current state of the game
            board.
        '''
        pass
