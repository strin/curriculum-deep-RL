import os
import numpy as np
import theano

floatX_name = os.environ.get('floatX')
if floatX_name and floatX_name == 'float64':
    floatX = np.float64
    theano.config.floatX = floatX_name
else:
    floatX = np.float32
    theano.config.floatX = 'float32'
