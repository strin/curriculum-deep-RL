class _MDP(object):
    '''
    Shared MDP behavior.
    MDP defines initial states and transition distributions
    '''
    @property
    def num_actions(self):
        # returns the cardinality of the action set
        raise NotImplementedError()

    @property
    def valid_actions(self, state):
        # returns the cardinality of the action set
        raise NotImplementedError()

    def start_state(self):
        # Note: initial state may be stochastic!
        raise NotImplementedError()

    def step(self, state, action):
        # performs the specified action and updates the internal state of
        # the environment
        # returns the next state in the environment
        raise NotImplementedError()

    def reward(self, state, action, next_state):
        # reward function of the MDP.
        raise NotImplementedError()

    def end_state(self, state):
        # verify if a state ends an episode.
        # by default, the MDP is not episodic.
        raise False


class DiscreteMDP(_MDP):
    '''
    A discrete MDP is an MDP where states are discrete and can be represented
    in a tabular form.
    '''
    @property
    def num_states(self):
        # returns the number of state for a tabular representation
        raise NotImplementedError()

class ContinuousMDP(_MDP):
    '''
    A continuous MDP is an MDP where states are continuous and can be represented
    in a vector/matrix/tensor form.
    '''
    @property
    def state_shape(self):
        # return a shape tuple.
        raise NotImplementedError()

class Task:
    '''
    In Task, states are memorized and not immediately writable by
    external code.
    '''
    @property
    def curr_state(self):
        raise NotImplementedError()

    def is_end(self):
        # verify if a state ends an episode.
        # by default, the MDP is not episodic.
        raise False

    @property
    def num_actions(self):
        # returns the cardinality of the action set
        raise NotImplementedError()

    @property
    def valid_actions(self):
        # returns the cardinality of the action set
        raise NotImplementedError()

    def step(self, action):
        # performs the specified action and updates the internal state of
        # the environment
        # returns reward.
        raise NotImplementedError()

class MDPTask(Task):
    def __init__(self, mdp):
        self.mdp = mdp
        self._curr_state = mdp.start_state()

    @property
    def curr_state(self):
        return self._curr_state

    def is_end(self):
        return self.mdp.end_state(self._curr_state)

    @property
    def num_actions(self):
        return self.mdp.num_actions

    @property
    def valid_actions(self):
        return self.mdp.valid_actions(self._curr_state)

    def step(self, action):
        next_state = self.mdp.step(self._curr_state, action)
        reward = self.mdp.reward(self._curr_state, action, next_state)
        self._curr_state = next_state
        return reward

    @property
    def num_states(self):
        return self.mdp.num_states

    @property
    def state_shape(self):
        return self.mdp.state_shape

