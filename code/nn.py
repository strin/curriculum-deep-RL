"""
Some util functions for constructing neural networks.
"""

import theano
import theano.tensor as T
import numpy as np

import layers
import optimizers
from utils import make_minibatch_x_y

class FeedforwardNet(object):
    def __init__(self, arch_func, l2_reg=0.0, epsilon=0.05, batch_size=32,
                 step_size = 1e-3, tensor_type=T.matrix):
        '''
        initialize the neural network with architecture.

        Params
        =====
            - arch: (function) inputs -> (params, outputs)
            - l2_reg: l2 regularization hyper-parameter
            - step_size: the step_size used in optimizers
        '''
        self.arch_func = arch_func
        self.l2_reg = l2_reg
        self.step_size = step_size
        self.batch_size = batch_size
        self.tensor_type = tensor_type

        self._initialize_net()


    def _initialize_net(self):
        '''
        Based on the architecture, create a neural network with fprop and backprop
        '''
        # construct computational graph.
        states = self.tensor_type('states')
        (params, final_output, model) = self.arch_func(states)
        self.model = model

        self.fprop = theano.function(inputs=[states], outputs=final_output, name='fprop')

        # compute back propagation.
        targets = T.matrix('targets')
        mse = layers.MSE(final_output, targets)

        l2_penalty = 0.
        for param in params:
            l2_penalty += (param ** 2).sum()

        cost = mse + self.l2_reg * l2_penalty

        updates = optimizers.Adam(cost, params, alpha=self.step_size)

        td_errors = T.sqrt(mse)

        self.bprop = theano.function(inputs=[states, targets],
                                     outputs=td_errors,
                                     updates=updates)


    def train(self, data, targets, num_iter = 1):
        '''
        train the network using given data.
            - data is a N x D matrix, where N is number of data points, and D is the dimension.
            - targets is a N x D' marix, where N is the number of data points, and D' is the dimension.
        '''
        mini_batches = make_minibatch_x_y(data, targets, self.batch_size, num_iter)
        for (data_batch, target_batch) in mini_batches:
            self.bprop(data_batch, target_batch)

    def predict(self, data):
        '''
        give predictions for given data matrix.
        '''
        return self.fprop(data)

    def mse(self, data, targets):
        preds = self.predict(data)
        return np.sum((preds - targets) ** 2) / preds.shape[0] / preds.shape[1]

    def __call__(self, data):
        return self.fprop(data)


def FCN(arch, l2_reg=0.0, epsilon=0.05, batch_size=32, step_size=1e-3):
    def arch_func(states):
        params = []
        model = []
        for (li, layer) in enumerate(arch[1:-1]):
            fc = layers.FullyConnected(arch[li], arch[li+1], activation='relu')
            params.extend(fc.params)
            model.append(fc)
        linear_layer = layers.FullyConnected(arch[-2], arch[-1], activation=None)
        model.append(linear_layer)
        params.extend(linear_layer.params)

        # construct computational graph.
        hidden = [model[0](states)]
        for (fi, fc) in enumerate(model[1:-1]):
            hidden.append(model[fi+1](hidden[-1]))
        final_output = model[-1](hidden[-1])
        return (params, final_output, {
                'layers': model
            })
    return FeedforwardNet(arch_func, l2_reg=l2_reg, epsilon=epsilon, batch_size=batch_size, step_size=step_size)


def GridWorldUltimateFCN(input_dim, l2_reg=0.0, epsilon=0.05, batch_size=32, step_size=1e-3):
    '''
    dim: dimension of the grid vector (also agent vec, demons vec, goal vec).
    '''
    def arch_func(states):
        params = []
        ## agent.
        H_AGENT_DIM1 = 5
        fc_agent1 = layers.FullyConnected(input_dim, H_AGENT_DIM1, activation='relu')
        params.extend(fc_agent1.params)
        h_agent1 = fc_agent1(states[:, :input_dim])
        ## grid.
        H_GRID_DIM1 = 10
        fc_grid1 = layers.FullyConnected(input_dim, H_GRID_DIM1, activation='relu')
        params.extend(fc_grid1.params)
        h_grid1 = fc_grid1(states[:, input_dim:2*input_dim])
        # (TODO:) grid layer 2.
        ## goal.
        H_GOAL_DIM1 = H_AGENT_DIM1 # symmetric
        fc_goal1 = layers.FullyConnected(input_dim, H_GOAL_DIM1, activation='relu')
        params.extend(fc_goal1.params)
        h_goal1 = fc_goal1(states[:, 2*input_dim:3*input_dim])
        ##demons.
        H_DEMONS_DIM1 = 5
        fc_demons1 = layers.FullyConnected(input_dim, H_DEMONS_DIM1, activation='relu')
        params.extend(fc_demons1.params)
        h_demons1 = fc_demons1(states[:, 3*input_dim:4*input_dim])
        ## combine them all!
        v_joint = T.concatenate([h_agent1, h_grid1, h_goal1, h_demons1], axis=1)
        H_JOINT_DIM1 = 5
        fc_joint1 = layers.FullyConnected(H_AGENT_DIM1+H_GRID_DIM1+H_GOAL_DIM1+H_DEMONS_DIM1, H_JOINT_DIM1, activation='relu')
        params.extend(fc_joint1.params)
        h_joint1 = fc_joint1(v_joint) 
        linear_layer = layers.FullyConnected(H_JOINT_DIM1, 1, activation=None)
        params.extend(linear_layer.params)
        output = linear_layer(h_joint1)

        return (params, output, {
                'fc_agent1': fc_agent1,
                'fc_grid1': fc_grid1,
                'fc_goal1': fc_goal1,
                'fc_demons1': fc_demons1,
                'fc_joint1': fc_joint1,
                'linear_layer': linear_layer
            })
    return FeedforwardNet(arch_func, l2_reg=l2_reg, epsilon=epsilon, batch_size=batch_size, step_size=step_size)




