import random
import numpy as np
import theano
import theano.tensor as T
import layers
import optimizers


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


class DQN(OnlineAgent):
    ''' Q-learning with a neural network function approximator

        TODO:   - reward clipping
                - epsilon decay
    '''

    def __init__(self, task, hidden_dim=128, l2_reg=0.0, lr=1e-1, epsilon=0.05,
                 memory_size=250, minibatch_size=32):
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

        self._initialize_net()

        # for now, keep experience as a list of tuples
        self.experience = []
        self.exp_idx = 0

        # used for streaming updates
        self.last_state = None
        self.last_action = None

    def _initialize_net(self):
        '''
            Attaches fprop and bprop functions to the class
        '''
        # simple 2 layer net with l2-loss
        states = T.matrix('states')
        hidden_layer1 = layers.FullyConnected(inputs=states,
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

        targets = T.vector('target')
        last_actions = T.lvector('action')
        MSE = layers.MSE(action_values[last_actions,
                         T.arange(action_values.shape[1])], targets)

        params = hidden_layer1.params + hidden_layer2.params + linear_layer.params

        l2_penalty = 0.
        for param in params:
            l2_penalty += (param ** 2).sum()

        cost = MSE.output + self.l2_reg * l2_penalty

        updates = optimizers.Adam(cost, params)

        print "Compiling fprop"
        self.fprop = theano.function(inputs=[states], outputs=action_values,
                                     name='fprop')

        # takes a single gradient step
        print "Compiling backprop"
        td_errors = T.sqrt(MSE.output)
        self.bprop = theano.function(inputs=[states, last_actions, targets],
                                     outputs=td_errors, updates=updates)

        return params

    def end_episode(self, reward):
        if self.last_state is not None:
            self._add_to_experience(self.last_state, self.last_action, None,
                                    reward)
        self.last_state = None
        self.last_action = None

    def get_action(self, state):
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

        states = np.zeros((self.state_dim, self.minibatch_size))
        next_states = np.zeros((self.state_dim, self.minibatch_size))
        actions = np.zeros(self.minibatch_size, dtype=int)
        rewards = np.zeros(self.minibatch_size)

        # sample and process minibatch
        samples = random.sample(self.experience, self.minibatch_size)
        terminals = []
        for idx, sample in enumerate(samples):
            state, action, next_state, reward = sample
            states[:, idx] = state.ravel()
            actions[idx] = action
            rewards[idx] = reward

            if next_state is not None:
                next_states[:, idx] = next_state.ravel()
            else:
                terminals.append(idx)

        # compute target reward + \gamma max_{a'} Q(ns, a')
        next_qvals = np.max(self.fprop(next_states), axis=0)

        # Ensure target = reward when NEXT_STATE is terminal
        next_qvals[terminals] = 0.

        targets = rewards + self.gamma * next_qvals

        self.bprop(states, actions, targets.flatten())

    def learn(self, next_state, reward):
        self._add_to_experience(self.last_state, self.last_action,
                                next_state, reward)
        self._update_net()


class RecurrentReinforceAgent(OnlineAgent):
    ''' Policy Gradient with a LSTM function approximator '''
    def __init__(self, task, hidden_dim=1024, options=None):
        self.task = task
        self.state_dim = task.get_state_dimension()  # input dimensionaliy
        self.num_actions = task.get_num_actions()  # softmax output dimensionality
        self.hidden_dim = hidden_dim  # number of RNN cells
        self.gamma = task.gamma

        # store of previous trajectories.. for now it is just 1.
        self.trajectory = {'states': [], 'actions': [], 'rewards': []}

        self._initialize_net()

        # used for streaming updates
        self.last_state = None
        self.last_action = None

    def _initialize_net(self):
        '''
            For now, just initialize the RNN controller. Soon, the RNN will
            become an LSTM. Also,  we will want to have another LSTM attempt
            to predict the value function for the current state (i.e. use a
            baseline to estimate the advantage function)
        '''
        rnn_layer = layers.RNNLayer(self.state_dim, self.hidden_dim,
                                    self.num_actions,
                                    hidden_activation='tanh',
                                    output_activation=None)

        h = theano.shared(value=np.zeros((1, self.hidden_dim)), name='h')

        # the initial hidden state isn't a param for now
        params = rnn_layer.params

        def step(state, hidden_vector):
            next_h, outputs = rnn_layer(state, hidden_vector)
            action_probs = layers.SoftMax(outputs).outputs
            return next_h, action_probs

        # forward prop takes a single step, updates the hidden layer,
        # and returns the action probabilities conditioned on the past
        curr_state = T.row('curr_state')
        next_h, action_probs = step(curr_state, h)

        print 'Compiling fprop'
        self.fprop = theano.function(inputs=[curr_state], outputs=action_probs,
                                     updates=[(h, next_h)], name='fprop')

        # backprop is more involved... we use REINFORCE to compute a descent
        # direction
        start_h = h * 0.
        state_sequence = T.matrix('state_sequence')
        action_sequence = T.lvector('action_sequence')  # must be integers
        reward_sequence = T.vector('reward_sequence')

        # scan returns a list of \pi(a_t \mid o_{1:t}) for all t
        [hidden_states, action_probs], _ = theano.scan(fn=step,
                                                       sequences=state_sequence,
                                                       outputs_info=[start_h, None]
                                                       )

        # remove the column dimension to get a matrix
        action_probs = action_probs[:, 0, :]

        # get the log probabilities for the actions taken along this path
        log_action_probs = T.log(action_probs[T.arange(action_probs.shape[0]),
                                 action_sequence])

        # reward[0] = 0:T, reward[1] = 1:T, etc.
        rewards = T.cumsum(reward_sequence[::-1])[::-1]

        cost = -T.sum(log_action_probs * rewards)

        # take a gradient descent step
        updates = optimizers.Adam(cost, params)

        print 'Compiling backprop'
        self.bprop = theano.function(inputs=[state_sequence, action_sequence,
                                             reward_sequence],
                                     outputs=[cost, action_probs],
                                     updates=updates, name='bprop')

        # reset simply sets the hidden state to zero for now.
        # TODO: backprop through hidden state
        print 'Compiling reset'
        self.reset_net = theano.function(inputs=[], updates=[(h, start_h)])

    def end_episode(self, reward):
        '''
            Updates the network
        '''
        self._add_to_experience(self.last_state, self.last_action, reward)

        # take a single gradient step
        # for now, we are only using a single sampled trajectory
        state_seq = np.vstack(self.trajectory['states'])  # matrix w/ states as rows
        action_seq = self.trajectory['actions']
        reward_seq = self.trajectory['rewards']

        cost, action_probs = self.bprop(state_seq, action_seq, reward_seq)

        self.reset_net()
        self.trajectory['states'] = []
        self.trajectory['actions'] = []
        self.trajectory['rewards'] = []

    def get_action(self, state):
        '''
            Uses a soft-action selection policy, where each action is
            selected according with probability p(a_t \mid s_{1:t}, a_{1:t-1})
        '''
        action_probs = self.fprop(state.T).flatten()
        action = np.random.choice(np.arange(self.num_actions), p=action_probs)
        self.last_state = state
        self.last_action = action
        return action

    def _add_to_experience(self, state, action, reward):
        '''
            Note: states are stored as row-vectors
        '''
        self.trajectory['states'].append(state.T)
        self.trajectory['actions'].append(action)
        self.trajectory['rewards'].append(reward)

    def learn(self, next_state, reward):
        '''
            NEXT_STATE is unused, but appears in the prototype to keep the
            same api
        '''
        # add the most experience to the dataset
        self._add_to_experience(self.last_state, self.last_action, reward)

        # TODO: Don't wait until the end of the episode to update params, just
        # do it every T steps
