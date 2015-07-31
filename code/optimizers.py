import numpy as np
import theano
import theano.tensor as T


def Adam(cost, params, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    '''
        Follows the psuedo-code from
            ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION
                http://arxiv.org/pdf/1412.6980v8.pdf
    '''
    updates = []
    t = theano.shared(value=1., name='t')
    grads = T.grad(cost, params)

    alpha_t = alpha * T.sqrt(1. - beta_2**t) / (1. - beta_1**t)
    for param, gparam in zip(params, grads):
        value = param.get_value(borrow=True)
        # initialize first and second moment updates parameter-wise
        m = theano.shared(value=np.zeros(value.shape, dtype=value.dtype),
                          broadcastable=param.broadcastable, name='m')
        v = theano.shared(value=np.zeros(value.shape, dtype=value.dtype),
                          broadcastable=param.broadcastable, name='v')

        # update biased first/second moment estimates
        m_t = beta_1 * m + (1. - beta_1) * gparam
        v_t = beta_2 * v + (1. - beta_2) * T.sqr(gparam)

        # use the efficient update from sec. 2 of the paper to avoid
        # computing the unbiased estimates
        g_t = m_t / (T.sqrt(v_t) + epsilon)
        param_t = param - alpha_t * g_t

        # store changes to the shared variables
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((param, param_t))

    updates.append((t, t + 1))
    return updates


def Adagrad(cost, params, base_lr=1e-2, epsilon=1e-8):
    '''
        Follows the psuedo-code from:
            http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf
    '''
    updates = []
    grads = T.grad(cost, params)
    for param, gparam in zip(params, grads):
        value = param.get_value(borrow=True)
        # cache for sum of squared historical gradients
        cache = theano.shared(value=np.zeros(value.shape, dtype=value.dtype),
                              broadcastable=param.broadcastable, name='cache')
        cache_t = cache + T.sqr(gparam)

        # per parameter adaptive learning rate
        param_t = param - base_lr * gparam / (T.sqrt(cache_t) + epsilon)

        # store changes to the shared variables
        updates.append((cache, cache_t))
        updates.append((param, param_t))

    return updates


def RMSProp(cost, params, base_lr=1e-2, decay_rate=0.99, epsilon=1e-8):
    '''
        Typical values of DECAY_RATE are [0.9, 0.99, 0.999].
    '''
    updates = []
    grads = T.grad(cost, params)
    for param, gparam in zip(params, grads):
        value = param.get_value(borrow=True)
        # leaky cache of sum of squared historical gradients
        cache = theano.shared(value=np.zeros(value.shape, dtype=value.dtype),
                              broadcastable=param.broadcastable, name='cache')
        cache_t = decay_rate * cache + (1. - decay_rate) * T.sqr(gparam)

        # per parameter adaptive learning rate
        param_t = param - base_lr * gparam / (T.sqrt(cache_t) + epsilon)

        # store changes to the shared variables
        updates.append((cache, cache_t))
        updates.append((param, param_t))

    return updates


def SGD(cost, params, lr=1e-2):
    '''
        Returns updates for vanilla SGD
        TODO: add momentum + options for learning rate decay schedule
    '''
    grads = T.grad(cost, params)
    return [(param, param - lr * gparam) for param, gparam in zip(params, grads)]
