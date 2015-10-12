import random
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
import layers
import optimizers
import cPickle as pickle
from util import unit_vec

class OnlineAgent(object):

    def get_action(self, state):
        raise NotImplementedError()

    def learn(self, next_state, reward):
        raise NotImplementedError()

    def reset(self):
        '''
            Optional method.
        '''

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

                best_val = 0.
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

    def end_episode(self, reward):
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
        if self.last_state is not None:
            self._td_update(self.last_state, self.last_action, reward,
                            next_state)

        # TODO: Add planning HERE;


class Trajectory(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def add_sample(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def zero_pad(self, max_len):
        diff = max_len - len(self.states)
        if diff <= 0:
            return

        # assumes state is a numpy array
        zero_states = [np.zeros_like(self.states[0])] * diff
        self.states += zero_states

        # zero pad actions and rewards (assumes they are scalar!)
        zeros = [0] * diff
        self.rewards += zeros
        self.actions += zeros

    def length(self):
        return len(self.states)


class RecurrentReinforceAgent(OnlineAgent):
    ''' Policy Gradient with a LSTM function approximator '''
    def __init__(self, task, hidden_dim=1024, truncate_gradient=-1, num_samples=5,
                 max_trajectory_length=float('inf'), mode=None, options=None):
        self.task = task
        self.state_dim = task.get_state_dimension()  # input dimensionaliy
        self.num_actions = task.get_num_actions()  # softmax output dimensionality
        self.hidden_dim = hidden_dim  # number of LSTM cells (in both the controller and baseline for now)
        self.truncate_gradient = truncate_gradient  # how many steps to backprop through
        self.gamma = task.gamma

        # Streaming counter of current trajectory
        self.max_trajectory_length = max_trajectory_length
        self.curr_trajectory_length = 0.

        # store of samples for the current parameters
        self.num_samples = num_samples
        self.curr_traj_idx = 0
        self.trajectories = [Trajectory()]

        # theano compilation directives
        if mode is None:
            self.mode = 'FAST_RUN'
        elif mode is 'fast_compile':
            self.mode = 'FAST_COMPILE'
        else:
            raise NotImplementedError()

        self.model = self._initialize_net()

        # used for streaming updates
        self.last_state = None
        self.last_action = None

    def _initialize_net(self):
        '''
            For now, just initialize the LSTM controller. Eventually, we will
            want to have another LSTM attempt to predict the value function
            for the current state (i.e. use a baseline to estimate the
            advantage function)
        '''
        # initialize Reinforce layers
        lstm_layer = layers.LSTMLayer(self.state_dim, self.hidden_dim)
        fc_layer = layers.FullyConnected(self.hidden_dim, self.num_actions)

        model = {'lstm_layer': lstm_layer, 'fc_layer': fc_layer}

        # build the Reinforce computation graph for a single time step
        def reinforce_step(state, h, prev_memory_cell):
            next_cell, next_hidden_layer = lstm_layer(state, h, prev_memory_cell)
            outputs = fc_layer(next_hidden_layer)
            action_probs = layers.SoftMax(outputs)
            return next_hidden_layer, next_cell, action_probs

        # use these variables to keepy track of h and memory_cell during
        # trajectory sampling
        h = theano.shared(value=lstm_layer.h0.get_value())
        cell = theano.shared(value=lstm_layer.cell_0.get_value())

        # forward prop takes a single step, updates the hidden layer,
        # and returns the action probabilities conditioned on the past
        curr_state = T.row('curr_state')
        next_h, next_cell, action_probs = reinforce_step(curr_state, h, cell)

        print 'Compiling fprop'
        step_updates = [(h, next_h), (cell, next_cell)]
        self.fprop = theano.function(inputs=[curr_state], outputs=action_probs,
                                     updates=step_updates, name='fprop',
                                     mode=self.mode)

        # backprop is more involved... we use REINFORCE to compute a descent
        # direction
        state_sequences = T.tensor3('state_sequences')
        action_sequences = T.lmatrix('action_sequences')  # must be integers
        reward_sequences = T.matrix('reward_sequences')
        num_trajectories = state_sequences.shape[1]

        # scan returns a list of \pi(a_t \mid o_{1:t}) for all t
        [hidden_states, memory_cells, action_probs], _ = theano.scan(
            fn=reinforce_step,
            sequences=state_sequences,
            outputs_info=[T.extra_ops.repeat(lstm_layer.h0,
                                             num_trajectories, axis=0),
                          T.extra_ops.repeat(lstm_layer.cell_0,
                                             num_trajectories, axis=0),
                          None],
            truncate_gradient=self.truncate_gradient)

        # action probs is a tensor, where dim 1 is the time, dim 2 is the
        # sample number and dim 3 is the actual vector of probabilities
        # This is into a matrix where dim 1 is sample number, and dim 2 is
        # is the log probability of the action taken at each time step
        log_action_probs = T.log(action_probs[T.arange(action_probs.shape[0]),
                                              T.arange(action_probs.shape[1]).reshape((-1, 1)),
                                              action_sequences])

        # reward[i, 0] = R^i_0:T, reward[i, 1] = R^i_1:T, etc.
        rewards = T.cumsum(reward_sequences[:, ::-1], axis=1)[:, ::-1]

        # baselines is a matrix with baseline[i, j] = baseline for sample i
        # at time j
        baselines, updates = self._get_baseline(state_sequences, action_sequences, rewards, model)

        # take a gradient descent step for the reinforce agent
        cost = -T.sum(log_action_probs * (rewards - baselines)) / num_trajectories
        params = lstm_layer.params + fc_layer.params
        updates += optimizers.Adam(cost, params)

        print 'Compiling backprop'
        self.bprop = theano.function(inputs=[state_sequences, action_sequences,
                                             reward_sequences],
                                     outputs=[cost, action_probs],
                                     updates=updates, name='bprop',
                                     mode=self.mode)

        # reset the hidden layer and the memory cell
        print 'Compiling reset'
        reset_updates = lstm_layer.reset(h, cell)
        self.reset_net = theano.function(inputs=[], updates=reset_updates,
                                         mode=self.mode)

        print 'done'
        return model

    def _get_baseline(self, state_sequences, action_sequences, rewards, model):
        '''
            TODO: Currently the baseline only depends on the state sequence
                  and not the action sequence!

                  To make this change, need to get a one-hot representation
                  of the actions. And use a new state representation that
                  consists of the concatenated state/actions
        '''
        # initialize baseline layers
        baseline_lstm_layer = layers.LSTMLayer(self.state_dim,  # + self.action_dim,
                                               self.hidden_dim)
        baseline_fc_layer = layers.FullyConnected(self.hidden_dim, 1)

        model['baseline_lstm'] = baseline_lstm_layer
        model['baseline_fc'] = baseline_fc_layer

        # build the baseline computation graph
        def baseline_step(state, h, prev_memory_cell):
            # concatenate the current state and the previous action
            # sa_vec = T.concatenate([state, prev_action], axis=0)
            next_cell, next_hidden_layer = baseline_lstm_layer(state, h, prev_memory_cell)
            reward_estimate = baseline_fc_layer(next_hidden_layer)
            return next_hidden_layer, next_cell, reward_estimate

        # compute the baselines
        [hidden_states, memory_cells, reward_estimates], _ = theano.scan(
            fn=baseline_step,
            sequences=state_sequences,
            outputs_info=[T.extra_ops.repeat(T.tanh(baseline_lstm_layer.cell_0),
                                             state_sequences.shape[1], axis=0),
                          T.extra_ops.repeat(baseline_lstm_layer.cell_0,
                                             state_sequences.shape[1], axis=0),
                          None],
            truncate_gradient=self.truncate_gradient)

        baselines = reward_estimates[:, :, 0].T

        baseline_cost = layers.MSE(rewards, baselines)
        baseline_params = baseline_lstm_layer.params + baseline_fc_layer.params
        updates = optimizers.Adam(baseline_cost, baseline_params)

        return baselines, updates

    def reset(self):
        self.reset_net()

        # erase all of the samples
        self.trajectories = [Trajectory()]
        self.curr_traj_idx = 0

    def end_episode(self, reward):
        '''
            Updates the network
        '''
        self.trajectories[self.curr_traj_idx].add_sample(self.last_state,
                                                         self.last_action,
                                                         reward)

        if len(self.trajectories) >= self.num_samples:
            self._update_net()

            # erase all of the samples
            self.trajectories = [Trajectory()]
            self.curr_traj_idx = 0
        else:
            # reset the hidden states
            self.reset_net()

            # add another trajectory slot
            self.trajectories.append(Trajectory())
            self.curr_traj_idx += 1

    def _update_net(self):
        # Find the longest trajectory so we can zero pad
        max_len = 0.
        for traj in self.trajectories:
            if traj.length() > max_len:
                max_len = traj.length()

        # Batch the trajectories
        state_seq = []
        action_seq = []
        reward_seq = []
        for traj in self.trajectories:

            # For now, handle sequences of different lengths by zero padding
            # and using a dense tensor
            if traj.length() < max_len:
                traj.zero_pad(max_len)

            tensor_slice = np.rollaxis(np.dstack(traj.states), -1)
            state_seq.append(tensor_slice)

            action_seq.append(traj.actions)
            reward_seq.append(traj.rewards)

        # states are a 3D tensor with 1st D as time, 2nd D as trajectory, etc.
        state_tensor = np.concatenate(state_seq, axis=1)

        # rows are the trajectory number, columns are the data
        action_matrix = np.row_stack(action_seq)
        reward_matrix = np.row_stack(reward_seq)

        # take a gradient step
        cost, action_probs = self.bprop(state_tensor, action_matrix, reward_matrix)

    def get_action(self, state):
        '''
            Uses a soft-action selection policy, where each action is
            selected according with probability p(a_t \mid s_{1:t}, a_{1:t-1})
        '''
        state = state.reshape(1, -1)  # states are stored as row-vectors
        action_probs = self.fprop(state).flatten()
        action = np.random.choice(np.arange(self.num_actions), p=action_probs)
        self.last_state = state
        self.last_action = action
        return action

    def learn(self, next_state, reward):
        '''
            NEXT_STATE is unused, but appears in the prototype to keep the
            same api
        '''
        self.curr_trajectory_length += 1
        if self.curr_trajectory_length > self.max_trajectory_length:

            # treat this experience as an "end of episode event"
            self.end_episode(reward)
            self.curr_trajectory_length = 0

        else:
            # add the most recent experience to the dataset
            self.trajectories[self.curr_traj_idx].add_sample(self.last_state,
                                                             self.last_action,
                                                             reward)

    def save_params(self, path):
        assert path is not None
        print 'Saving params to ', path
        params = {}
        for name, layer in self.model.iteritems():
            params[name] = layer.get_params()
        pickle.dump(params, file(path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def load_params(self, path):
        assert path is not None
        print 'Restoring params from ', path
        params = pickle.load(file(path, 'r'))
        for name, layer in self.model.iteritems():
            layer.set_params(params[name])

        self.reset_net()


class DecompositionAgent(OnlineAgent):
    '''
            THIS NETWORK IS A HACK FOR THE APPLES TASK ON THE SQUARE!
    '''

    def __init__(self, task, hidden_dim=128, l2_reg=0.0, lr=1e-1, epsilon=0.05,
                 memory_size=250, minibatch_size=64):
        self.task = task
        self.state_dim = task.get_state_dimension()
        self.num_actions = task.get_num_actions()

        self.gamma = task.gamma
        self.hidden_dim = hidden_dim
        self.l2_reg = l2_reg
        self.lr = lr
        self.epsilon = epsilon
        self.memory_size = memory_size  # number of experiences to store
        self.minibatch_size = minibatch_size

        self.model = self._initialize_net()

        # for now, keep experience as a list of tuples
        self.experience = []
        self.exp_idx = 0

        # used for streaming updates
        self.last_state = None
        self.last_action = None

    def get_decomposed_state(self, state):
        '''
            HACK! THIS HARDCODES A SINGLE TASK INTO THE NETWORK!
        '''
        s = state[:2]
        g = state[2:6]
        r = state[6:]

        s_g = [np.concatenate([s, g * unit_vec(4, idx), r * unit_vec(4, idx)]) for idx in xrange(4)]

        return np.concatenate(s_g)

    def _initialize_net(self):
        '''
            THIS NETWORK IS A HACK FOR THE APPLES TASK ON THE SQUARE!

            Assumes all (s, g_i) pairs are the same length and that the
            input state vector is [(s, g_1), (s, g_2), (s, g_3), (s, g_4)]
        '''
        # initialize layers
        fc1 = layers.FullyConnected(input_dim=self.state_dim,
                                    output_dim=self.hidden_dim,
                                    activation='relu')

        # HACK! (Concatenation)
        fc2 = layers.FullyConnected(input_dim=4*self.hidden_dim,
                                    output_dim=self.hidden_dim,
                                    activation='relu')

        linear_layer = layers.FullyConnected(input_dim=self.hidden_dim,
                                             output_dim=self.num_actions,
                                             activation=None)

        model = {'fc1': fc1, 'fc2': fc2, 'linear': linear_layer}

        # construct computation graph for forward pass
        states = T.matrix('states')

        # HACK!
        ####
        splits = [self.state_dim] * 4
        v1, v2, v3, v4 = T.split(states, splits, n_splits=4, axis=1)
        h1, h2, h3, h4 = fc1(v1), fc1(v2), fc1(v3), fc1(v4)

        # CONCAT
        # fix fc2 input dimensions
        hidden_1 = T.concatenate([h1, h2, h3, h4], axis=1)

        # SUM
        # hidden_1 = h1 + h2 + h3 + h4

        # h11, h22, h33, h44 = fc2(h1), fc2(h2), fc2(h3), fc2(h4)

        # H2 CONCATE. Fix linear_layer
        # hidden_2 = T.concatenate([h11, h22, h33, h44], axis=1)

        # SUM
        # hidden_2 = h11 + h22 + h33 + h44

        ######
        hidden_2 = fc2(hidden_1)
        action_values = linear_layer(hidden_2)

        print "Compiling DECOMPOSITION FPROP"
        self.fprop = theano.function(inputs=[states], outputs=action_values,
                                     name='fprop')

        # build computation graph for backward pass (using the variables
        # introduced previously for forward)
        targets = T.vector('target')
        last_actions = T.lvector('action')
        mse = layers.MSE(action_values[T.arange(action_values.shape[0]),
                         last_actions], targets)

        # regularization
        params = fc1.params + fc2.params + linear_layer.params
        l2_penalty = 0.
        for param in params:
            l2_penalty += (param ** 2).sum()

        cost = mse + self.l2_reg * l2_penalty

        updates = optimizers.Adam(cost, params)

        # takes a single gradient step
        print "Compiling DECOMPOSITION bprop"
        td_errors = T.sqrt(mse)
        self.bprop = theano.function(inputs=[states, last_actions, targets],
                                     outputs=td_errors, updates=updates)

        print 'done'
        return model

    def end_episode(self, reward):
        if self.last_state is not None:
            self._add_to_experience(self.last_state, self.last_action, None,
                                    reward)
        self.last_state = None
        self.last_action = None

    def get_qvals(self, state):
        state = self.get_decomposed_state(state)

        state = state.reshape(1, -1)
        return self.fprop(state)

    def get_action(self, state):
        state = self.get_decomposed_state(state)

        # transpose since the DQN expects row vectors
        state = state.reshape(1, -1)

        # epsilon greedy w.r.t the current policy
        if(random.random() < self.epsilon):
            action = np.random.randint(0, self.num_actions)
        else:
            # a^* = argmax_{a} Q(s, a)
            action = np.argmax(self.fprop(state))

        self.last_state = state
        self.last_action = action

        return action

    def _add_to_experience(self, s, a, ns, r):
        # TODO: improve experience replay mechanism by making it harder to
        # evict experiences with high td_error, for example
        if len(self.experience) < self.memory_size:
            self.experience.append((s, a, ns, r))
        else:
            self.experience[self.exp_idx] = (s, a, ns, r)
            self.exp_idx += 1
            if self.exp_idx >= self.memory_size:
                self.exp_idx = 0

    def _update_net(self):
        '''
            sample from the memory dataset and perform gradient descent on
            (target - Q(s, a))^2
        '''

        # don't update the network until sufficient experience has been
        # accumulated
        if len(self.experience) < self.memory_size:
            return

        #HACK
        ##
        states = np.zeros((self.minibatch_size, 4*self.state_dim,))
        next_states = np.zeros((self.minibatch_size, 4*self.state_dim))
        ##
        actions = np.zeros(self.minibatch_size, dtype=int)
        rewards = np.zeros(self.minibatch_size)

        # sample and process minibatch
        samples = random.sample(self.experience, self.minibatch_size)
        terminals = []
        for idx, sample in enumerate(samples):
            state, action, next_state, reward = sample
            states[idx, :] = state.ravel()
            actions[idx] = action
            rewards[idx] = reward

            if next_state is not None:
                next_states[idx, :] = next_state.ravel()
            else:
                terminals.append(idx)

        # compute target reward + \gamma max_{a'} Q(ns, a')
        next_qvals = np.max(self.fprop(next_states), axis=1)

        # Ensure target = reward when NEXT_STATE is terminal
        next_qvals[terminals] = 0.

        targets = rewards + self.gamma * next_qvals

        self.bprop(states, actions, targets.flatten())

    def learn(self, next_state, reward):
        next_state = self.get_decomposed_state(next_state)

        self._add_to_experience(self.last_state, self.last_action,
                                next_state, reward)
        self._update_net()

    def save_params(self, path):
        assert path is not None
        print 'Saving params to ', path
        params = {}
        for name, layer in self.model.iteritems():
            params[name] = layer.get_params()
        pickle.dump(params, file(path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def load_params(self, path):
        assert path is not None
        print 'Restoring params from ', path
        params = pickle.load(file(path, 'r'))
        for name, layer in self.model.iteritems():
            layer.set_params(params[name])


class OracleAgent(OnlineAgent):
    '''
    an oracle agent knows the value of each state, and acts based on the values (i.e. off-policy).

    notice, however these values might not necessarily be the ground truth values,
    instead, it could any value function approximated by a model like neural networks.
    '''
    def __init__(self, value_func, task,
        strategy={
            'name': 'softmax',
            'temperature': 1.},
        tol=1e-3):
        '''
        values (dict): a mapping from state to values
        '''
        self.value_func = value_func
        self.task = task
        self.strategy = strategy
        self.num_states = task.get_num_states()
        self.tol = tol
        self.V = [1. for s in xrange(self.num_states)]

    def get_actions_with_probs(self, state):
        available_actions = self.task.get_allowed_actions(state)
        available_values = []
        probs = []
        if available_actions:
            for action in available_actions:
                val = 0.
                ns_dist = self.task.next_state_distribution(state, action)
                for ns, prob in ns_dist:
                    val += prob * (self.task.get_reward(state, action, ns) +
                            self.task.gamma * float(self.value_func(ns)))
                available_values.append(val)
            if self.strategy['name'] == 'softmax':
                probs = np.array(available_values)
                probs = np.exp(probs / self.strategy['temperature'])
            elif self.strategy['name'] == 'eps-greedy':
                if npr.rand() < self.strategy['eps']:
                    probs = np.array([1.0] * len(available_actions))
                else:
                    action_i = np.argmax(available_values)
                    probs = np.zeros(len(available_actions))
                    probs[action_i] = 1.
            probs /= sum(probs)
        return (available_actions, probs)

    def get_action(self, state):
        (available_actions, probs) = self.get_actions_with_probs(state)
        action = npr.choice(available_actions, 1, replace=True, p = probs)[0]
        return action

    def learn(self, max_iter = 1000):
        ''' Performs value iteration on the MDP until convergence '''
        for it in range(max_iter):
            max_diff = 0.

            for state in xrange(self.num_states):
                (available_actions, probs) = self.get_actions_with_probs(state)
                if not available_actions:
                    total_val = 0.
                else:
                    total_val = 0.
                    for idx, action in enumerate(available_actions):
                        val = 0.
                        ns_dist = self.task.next_state_distribution(state, action)
                        for ns, prob in ns_dist:
                            val += prob * (self.task.get_reward(state, action, ns) +
                                           self.task.gamma * self.V[ns])
                        total_val += probs[idx] * val

                diff = abs(self.V[state] - total_val)
                self.V[state] = total_val

                if diff > max_diff:
                    max_diff = diff

            if max_diff < self.tol:
                return
        print 'warning: value iteration not terminated'


