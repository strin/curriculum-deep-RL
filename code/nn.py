"""
Some neural networks
"""

import theano
import theano.tensor as T
import numpy as np

import layers
import optimizers
from utils import make_minibatch_x_y

class FCN(object):
    """
    fully connected network
    """
    def __init__(self, arch, l2_reg=0.0, epsilon=0.05, batch_size=32,
                 step_size = 1e-3):
        '''
        initialize the neural network with architecture.

        Params
        =====
            - arch: [num_vis, num_hidden_1, num_hidden_2, ...]
            - l2_reg: l2 regularization hyper-parameter
            - step_size: the step_size used in optimizers
        '''
        self.arch = arch
        self.l2_reg = l2_reg
        self.step_size = step_size
        self.batch_size = batch_size

        self.model = self._initialize_net()


    def _initialize_net(self):
        '''
        Based on the parameters, create a neural network with fprop and backprop
        '''
        # construct model.
        model = []
        params = []
        for (li, layer) in enumerate(self.arch[1:-1]):
            fc = layers.FullyConnected(self.arch[li], self.arch[li+1], activation='relu')
            params.extend(fc.params)
            model.append(fc)
        linear_layer = layers.FullyConnected(self.arch[-2], self.arch[-1], activation=None)
        model.append(linear_layer)

        # construct computational graph.
        states = T.matrix('states')
        hidden = [model[0](states)]
        for (fi, fc) in enumerate(model[1:-1]):
            hidden.append(model[fi+1](hidden[-1]))
        final_output = model[-1](hidden[-1])

        print "Compiling fprop"
        self.fprop = theano.function(inputs=[states], outputs=final_output, name='fprop')

        # compute back propagation.
        targets = T.matrix('targets')
        mse = layers.MSE(final_output, targets)

        l2_penalty = 0.
        for param in params:
            l2_penalty += (param ** 2).sum()

        cost = mse + self.l2_reg * l2_penalty

        updates = optimizers.Adam(cost, params, alpha=self.step_size)

        print 'Compiling backprop'
        td_errors = T.sqrt(mse)

        self.bprop = theano.function(inputs=[states, targets],
                                     outputs=td_errors,
                                     updates=updates)

        print 'done'
        return model

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

