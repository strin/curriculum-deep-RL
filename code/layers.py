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
        b = theano.shared(value=b_init, name='b')

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
        self.output = T.mean((inputs - targets)**2)
