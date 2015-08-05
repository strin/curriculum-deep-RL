import numpy as np
import theano
import theano.tensor as T


class FullyConnected(object):
    def __init__(self, inputs, input_dim, output_dim, activation='relu'):
        '''
            TODO: choice of initialization
                    - Currently uses the initialization scheme for relu
                      proposed in "Delving Deep into Rectifiers: Surpassing
                      Human-Level Performance on ImageNet Classification"

                        - zero-mean Gaussian with variance 2/(input_dim)

                  Other choice of activations
                  - if None, then defaults to a linear layer
        '''

        self.input = inputs

        # initialize weight matrix W of size (output_dim, input_dim)
        std_dev = np.sqrt(2. / input_dim)
        W_init = std_dev * np.random.randn(output_dim, input_dim)
        W = theano.shared(value=W_init, name='W')

        # initialize bias vector b of size (output_dim, 1)
        b_init = np.zeros((output_dim, 1))
        b = theano.shared(value=b_init, name='b', broadcastable=(False, True))

        # set-up the outputs
        if activation is None:
            self.output = W.dot(inputs) + b
        elif activation is 'relu':
            self.output = T.maximum(W.dot(inputs) + b, 0)
        elif activation is 'tanh':
            self.output = T.tanh(W.dot(inputs) + b)
        else:
            raise NotImplementedError()

        # store parameters
        self.W = W
        self.b = b
        self.params = [self.W, self.b]


class MSE(object):
    def __init__(self, inputs, targets):
        '''
            Computes the MSE between inputs and targets
        '''
        self.input = inputs
        self.targets = targets
        self.output = T.mean(T.sqr(inputs - targets))


class SoftMax(object):
    def __init__(self, inputs):
        self.inputs = inputs
        # self.outputs = T.nnet.softmax(self.inputs)
        e_x = T.exp(self.inputs - self.inputs.max(axis=1, keepdims=True))
        currentLayerValues = e_x / e_x.sum(axis=1, keepdims=True)
        self.outputs = currentLayerValues


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
        r = 0.001

        def orth(A):
            '''
                Returns an orthonormal basis for A
            '''
            return np.linalg.svd(A)[0]

        # input to hidden layer weight matrix
        W_x_init = orth(np.random.rand(input_dim, hidden_dim) * 2 * r - r)
        self.W_x = theano.shared(value=W_x_init, name='W_x')

        # hidden layer to output matrix
        W_o_init = orth(np.random.rand(hidden_dim, output_dim) * 2 * r - r)
        self.W_o = theano.shared(value=W_o_init, name='W_o')

        # hidden layer to hidden layer matrix
        W_h_init = orth(np.random.rand(hidden_dim, hidden_dim) * 2 * r - r)
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
