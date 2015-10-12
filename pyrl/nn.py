"""
Some util functions for constructing neural networks.
"""

import theano
import theano.tensor as T
from theano.printing import pydotprint
import numpy as np

import layers
import optimizers
import arch
from utils import make_minibatch_x_y

class FeedforwardNet(object):
    DEFAULT_PARAMS = dict(l2_reg=0.0, epsilon=0.05, batch_size=32, step_size=1e-3)

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

    def visualize_net(self):
        pydotprint(self.output)

    def _initialize_net(self):
        '''
        Based on the architecture, create a neural network with fprop and backprop
        '''
        # construct computational graph.
        states = self.tensor_type('states')
        (final_output, model) = self.arch_func(states)
        self.model = model
        params = arch.model_params(model)

        self.output = final_output
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


def FCN(arch, **kwargs):
    args = dict(FeedforwardNet.DEFAULT_PARAMS).update(kwargs)
    return FeedforwardNet(arch.fully_connected, **args)


def GridWorldUltimateFCN(input_dim, **kwargs):
    args = dict(FeedforwardNet.DEFAULT_PARAMS).update(kwargs)
    return FeedforwardNet(lambda states: arch.GridWorld_5x5_FCN(states, input_dim),
                          **args)




