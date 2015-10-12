# reproduce DeepMind's paper on Universal Value Function Approximation.
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
import layers
import optimizers

class UVFA(object):
    '''
    a two-stream architecture that learns from multiple tasks.
    '''
    def __init__(self):




