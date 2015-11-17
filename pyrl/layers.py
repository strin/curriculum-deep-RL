import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d

class FullyConnected(object):
    def __init__(self, input_dim, output_dim, activation='relu'):
        '''
            TODO: choice of initialization
                    - Currently uses the initialization scheme for relu
                      proposed in "Delving Deep into Rectifiers: Surpassing
                      Human-Level Performance on ImageNet Classification"

                        - zero-mean Gaussian with variance 2/(input_dim)

                  Other choices of activations
                  - if None, then defaults to a linear layer
        '''
        if activation is None:
            self.act = lambda x: x
        elif activation is 'relu':
            self.act = lambda x: T.maximum(x, 0)
        elif activation is 'tanh':
            self.act = lambda x: T.tanh(x)
        else:
            raise NotImplementedError()

        # initialize weight matrix W of size (input_dim, output_dim)
        std_dev = np.sqrt(0.2 / input_dim)
        W_init = std_dev * np.random.randn(input_dim, output_dim)
        W = theano.shared(value=W_init, name='W')

        # initialize bias vector b of size (output_dim, 1)
        b_init = np.zeros((1, output_dim))
        b = theano.shared(value=b_init, name='b', broadcastable=(True, False))

        # store parameters
        self.W = W
        self.b = b
        self.params = [self.W, self.b]

    def __call__(self, inputs):
        # set-up the outputs
        return self.act(inputs.dot(self.W) + self.b)

    def get_params(self):
        params = {}
        for p in self.params:
            params[p.name] = p.get_value()
        return params

    def set_params(self, params):
        for p in self.params:
            p.set_value(params[p.name], borrow=True)


class Conv(object):
    '''
    convolution layers.
    '''
    def __init__(self, input_dim, output_dim, filter_size = (2, 2), pool_size = (2, 2), activation='relu', border_mode='valid'):
        '''
            TODO: choice of initialization
                    - Currently uses the initialization scheme for relu
                      proposed in "Delving Deep into Rectifiers: Surpassing
                      Human-Level Performance on ImageNet Classification"

                        - zero-mean Gaussian with variance 2/(input_dim)

                  Other choices of activations
                  - if None, then defaults to a linear layer
        '''
        if activation is None:
            self.act = lambda x: x
        elif activation is 'relu':
            self.act = lambda x: T.maximum(x, 0)
        elif activation is 'tanh':
            self.act = lambda x: T.tanh(x)
        else:
            raise NotImplementedError()

        # initialize weight matrix W of size (input_dim, output_dim)
        std_dev = np.sqrt(2. / (input_dim * filter_size[0] * filter_size[1]))
        W_init = std_dev * np.random.randn(output_dim, input_dim, filter_size[0], filter_size[1])
        W = theano.shared(value=W_init, name='W')

        # initialize bias vector b of size (output_dim, 1)
        b_init = np.zeros((output_dim))
        b = theano.shared(value=b_init, name='b')

        # store parameters
        self.W = W
        self.b = b
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.border_mode = border_mode
        self.params = [self.W, self.b]

    def __call__(self, inputs):
        # set-up the outputs
        conv_out = T.nnet.conv.conv2d(inputs, self.W, border_mode=self.border_mode) + self.b.dimshuffle('x', 0, 'x', 'x')
        pool_out = max_pool_2d(input=conv_out, ds=self.pool_size, ignore_border=True)
        return pool_out

    def get_params(self):
        params = {}
        for p in self.params:
            params[p.name] = p.get_value()
        return params

    def set_params(self, params):
        for p in self.params:
            p.set_value(params[p.name], borrow=True)

def orth(A):
    '''
        Returns an orthonormal basis for A
    '''
    return np.linalg.svd(A)[0]


class RNNLayer(object):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 hidden_activation='relu', output_activation=None):
        '''
            input_dim: input dimensionality
            hidden_dim: hidden layer dimensionality
            output_dim: output layer dimensionality (number of classes inputs
                                                     a single layer model)
        '''
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        init_scale = 0.001

        # input to hidden layer weight matrix
        W_x_init = np.random.rand(input_dim, hidden_dim) * 2 * init_scale - init_scale
        self.W_x = theano.shared(value=W_x_init, name='W_x')

        # hidden layer to output matrix
        W_o_init = np.random.rand(hidden_dim, output_dim) * 2 * init_scale - init_scale
        self.W_o = theano.shared(value=W_o_init, name='W_o')

        # hidden layer to hidden layer matrix
        W_h_init = orth(np.random.rand(hidden_dim, hidden_dim) * 2 * init_scale - init_scale)
        self.W_h = theano.shared(value=W_h_init, name='W_h')

        # biases
        b_h_init = np.zeros((1, hidden_dim))
        self.b_h = theano.shared(value=b_h_init, name='b_h',
                                 broadcastable=(True, False))

        b_o_init = np.zeros((1, output_dim))
        self.b_o = theano.shared(value=b_o_init, name='b_o',
                                 broadcastable=(True, False))

        # store params
        self.params = [self.W_x, self.W_o, self.W_h, self.b_h, self.b_o]

    def _apply_nonlinearity(self, activations, nonlinearity):
         # set-up the outputs
        if nonlinearity is None:
            return activations
        elif nonlinearity is 'relu':
            return T.maximum(activations, 0)
        elif nonlinearity is 'tanh':
            return T.tanh(activations)
        else:
            raise NotImplementedError()

    def __call__(self, x, h):
        '''
            Take a single RNN step. Requires as input the current state x
            and previous hidden state h

            Return the output and the new hidden state.
        '''
        new_h = self._apply_nonlinearity(x.dot(self.W_x) + h.dot(self.W_h) +
                                         self.b_h, self.hidden_activation)

        output = self._apply_nonlinearity(new_h.dot(self.W_o) + self.b_o,
                                          self.output_activation)

        return [new_h, output]

    def get_params(self):
        params = {}
        for p in self.params:
            params[p.name] = p.get_value()
        return params

    def set_params(self, params):
        for p in self.params:
            p.set_value(params[p.name], borrow=True)


class LSTMLayer(object):
    '''
        Modification of the standard RNN architecture with long-term
        short-term memory units.

        For speed of implementation, we modify the standard update equation
        for the output gate at time t to remove dependence on the memory
        unit at time t.

        This allows us to concatenate all of separate weight matrices into
        a single weight matrix, which speeds up computation via batching.

        For simplicity, we use tanh activations by default.
    '''
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        init_scale = 0.001

        # initial hidden params
        memory_cell_init = np.zeros((1, hidden_dim))
        cell_0 = theano.shared(value=memory_cell_init, name='mc_0')
        h0 = theano.shared(value=np.tanh(cell_0.get_value()), name='h0')

        # memory cell weights
        W_x_init = np.random.rand(input_dim, 4 * hidden_dim) * 2 * init_scale - init_scale
        W_x = theano.shared(value=W_x_init, name='W_x')

        def recurrent_matrix(dim):
            return orth(np.random.rand(dim, dim) * 2 * init_scale - init_scale)

        U_h_init = np.concatenate([recurrent_matrix(hidden_dim),
                                   recurrent_matrix(hidden_dim),
                                   recurrent_matrix(hidden_dim),
                                   recurrent_matrix(hidden_dim)], axis=1)

        U_h = theano.shared(value=U_h_init, name='U_h')

        b_init = np.zeros((1, 4 * hidden_dim))
        b = theano.shared(value=b_init, name='b', broadcastable=(True, False))

        # mapping from memory cell unit to slice
        self.units = {'i': 0, 'c': 1, 'f': 2, 'o': 3}
        self.unit_size = self.hidden_dim

        # store params
        self.cell_0 = cell_0
        self.h0 = h0    # note we don't backprop through this param!
        self.W_x = W_x
        self.U_h = U_h
        self.b = b
        self.params = [self.W_x, self.U_h, self.b, self.cell_0]

    def __call__(self, x, h, prev_cell):
        z = x.dot(self.W_x) + h.dot(self.U_h) + self.b

        def _get_unit(matrix, unit, dim):
            slice_num = self.units[unit]
            # assume all slices have the same dimension
            return matrix[:, slice_num * dim: (slice_num + 1) * dim]

        # input gate
        i = T.nnet.sigmoid(_get_unit(z, 'i', self.unit_size))

        # candidate memory cell
        candidate = T.tanh(_get_unit(z, 'c', self.unit_size))

        # forget gate
        f = T.nnet.sigmoid(_get_unit(z, 'f', self.unit_size))

        # output gate (note it doesn't involve the current memory cell)
        o = T.nnet.sigmoid(_get_unit(z, 'o', self.unit_size))

        next_cell = i * candidate + f * prev_cell

        h = o * T.tanh(next_cell)

        return [next_cell, h]

    def reset(self, h, cell):
        '''
            Resets these shared variables to their default values
        '''
        # update the starting layer since we don't backprop through it
        self.h0.set_value(np.tanh(self.cell_0.get_value()))
        return [(h, self.h0), (cell, self.cell_0)]

    def get_params(self):
        params = {}
        for p in self.params:
            params[p.name] = p.get_value()
        return params

    def set_params(self, params):
        for p in self.params:
            p.set_value(params[p.name], borrow=True)


#######################
#  OBJECTIVES         #
#######################
def SVR(inputs, targets, eps=0.3):
    '''
        Computes the MSE between inputs and targets
    '''
    delta = T.sqrt(T.sqr(inputs - targets))
    return T.mean((delta - eps) * (delta > eps))

def MSE(inputs, targets):
    '''
        Computes the MSE between inputs and targets
    '''
    return T.mean(T.sqr(inputs - targets))


def SoftMax(inputs):
    # return T.nnet.softmax(self.inputs)

    # improves numerical stability
    e_x = T.exp(inputs - inputs.max(axis=1, keepdims=True))
    currentLayerValues = e_x / e_x.sum(axis=1, keepdims=True)

    return currentLayerValues
