
# coding: utf-8

# In[ ]:

import scriptinit
import numpy as np
import unittest
import numpy.testing
from agent import DQN, RecurrentReinforceAgent
import theano


# In[ ]:

# Chain MDP from https://github.com/spragunr/deep_q_rl and modified to fit our DQN and API

##
# For now, the DQN assumes vector representation of state. Modify the state representation here to
# get tensor representation later.
##
class ChainMDP(object):
    """Simple markov chain style MDP.  Three "rooms" and one absorbing
    state. States are encoded for the q_network as arrays with
    indicator entries. E.g. [1, 0, 0, 0] encodes state 0, and [0, 1,
    0, 0] encodes state 1.  The absorbing state is [0, 0, 0, 1]

    Action 0 moves the agent left, departing the maze if it is in state 0.
    Action 1 moves the agent to the right, departing the maze if it is in
    state 2.

    The agent receives a reward of .7 for departing the chain on the left, and
    a reward of 1.0 for departing the chain on the right.

    Assuming deterministic actions and a discount rate of .5, the
    correct Q-values are:

    .7|.25,  .35|.5, .25|1.0,  0|0
    """

    def __init__(self, success_prob=1.0, reward_left=0.7, reward_right=1.0):
        self.num_actions = 2
        self.num_states = 5
        self.gamma = .5
        self.success_prob = success_prob

        self.actions = [np.array([[0]], dtype='int32'),
                        np.array([[1]], dtype='int32')]

        self.reward_zero = 0 
        self.reward_left = reward_left
        self.reward_right = reward_right

        self.states = []
        for i in range(self.num_states):
            self.states.append(np.zeros((self.num_states, 1),
                                        dtype=theano.config.floatX))
            self.states[-1][i, 0] = 1

    def act(self, state, action_index):

        """
        action 0 is left, 1 is right.
        """
        state_index =  np.nonzero(state[:, 0])[0][0]
        
        if state_index == self.num_states - 1: # terminal state
            return self.reward_zero, self.states[-1], np.array([[True]])
        
        if state_index == self.num_states - 2: # first time in absorbing state
            return self.reward_zero, self.states[-1], np.array([[False]])

        next_index = state_index
        if np.random.random() < self.success_prob:
            next_index = state_index + action_index * 2 - 1

        # Exit left
        if next_index == -1:
            return self.reward_left, self.states[-2], np.array([[False]])

        # Exit right
        if next_index == self.num_states - 2:
            return self.reward_right, self.states[-2], np.array([[False]])

        if np.random.random() < self.success_prob:
            return (self.reward_zero,
                    self.states[state_index + action_index * 2 - 1],
                    np.array([[False]]))
        else:
            return (self.reward_zero, self.states[state_index],
                    np.array([[False]]))
    
    def get_state_dimension(self):
        return self.num_states
    
    def get_num_actions(self):
        return self.num_actions


# In[ ]:

####
# Unit test for the Deep Q-Network
####

class DQNConvergenceTest(unittest.TestCase):
    def setUp(self):
        self.mdp = ChainMDP()

    def all_q_vals(self, net):
        """ Helper method to get the entire Q-table """

        q_vals = np.zeros((self.mdp.num_states, self.mdp.num_actions))
        for i in range(self.mdp.num_states):
            q_vals[i, :] = net.fprop(self.mdp.states[i].T)
        return q_vals

    def train(self, net, steps):
        mdp = self.mdp
        curr_state =mdp.states[np.random.randint(0, mdp.num_states-1)]
        for step in xrange(steps):
            action = net.get_action(curr_state)
            reward, next_state, terminal = mdp.act(curr_state, action)
            if terminal:
                net.end_episode(reward)
                curr_state = mdp.states[np.random.randint(0, mdp.num_states-1)]
            else:
                net.learn(next_state, reward)
                curr_state = next_state

    def test_convergence_sgd(self):
        dqn = DQN(self.mdp, hidden_dim=128, l2_reg=0.0, epsilon=0.2)
        self.train(dqn, 1000)
        
        # there is a secret "5-th" state corresponding to the second visit
        # to the absorbing state (to avoid infinite looping), so only check the
        # first four states
        numpy.testing.assert_almost_equal(self.all_q_vals(dqn)[:4],
                                          [[.7, .25], [.35, .5],
                                           [.25, 1.0], [0., 0.]], 3)


# In[ ]:

# Testing the DQN
suite = unittest.TestLoader().loadTestsFromTestCase(DQNConvergenceTest)
unittest.TextTestRunner(verbosity=2).run(suite)


# In[ ]:

#####
# Unit test for the Recurrent Policy Gradient Implementation
#####
class ReinforceActionTest(unittest.TestCase):
    def setUp(self):
        self.mdp = ChainMDP(reward_left=-20, reward_right=20)

    def all_action_distribution(self, net):
        """ Helper method to get the entire Q-table """

        action_probs = np.zeros((self.mdp.num_states, self.mdp.num_actions))
        for i in range(self.mdp.num_states):
            net.reset_net()
            action_probs[i, :] = net.fprop(self.mdp.states[i].T).flatten()
        return action_probs

    def train(self, net, steps):
        mdp = self.mdp
        curr_state =mdp.states[np.random.randint(0, mdp.num_states-1)]
        for step in xrange(steps):
            action = net.get_action(curr_state)
            reward, next_state, terminal = mdp.act(curr_state, action)
            if terminal:
                net.end_episode(reward)
                curr_state = mdp.states[np.random.randint(0, mdp.num_states-1)]
            else:
                net.learn(next_state, reward)
                curr_state = next_state

    def test_convergence_sgd(self):
        rr_agent = RecurrentReinforceAgent(self.mdp, hidden_dim=128, num_samples=10, mode='fast_compile')
        self.train(rr_agent, 50000)
        diffs = (self.all_action_distribution(rr_agent)[:4, 1] - [0.95]*4) < 0
        self.assertEqual(sum(diffs), 0)


# In[ ]:

# Testing the Recurrent Reinforce Agent
suite = unittest.TestLoader().loadTestsFromTestCase(ReinforceActionTest)
unittest.TextTestRunner(verbosity=2).run(suite)


# In[ ]:



