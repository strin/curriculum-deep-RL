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
    for param, gparam in zip(params, grads):
        # initialize first and second moment updates parameter-wise
        m = theano.shared(value=param.get_value() * 0., name='m')
        v = theano.shared(value=param.get_value() * 0., name='v')

        # update biased first/second moment estimates
        m_t = beta_1 * m + (1. - beta_1) * gparam
        v_t = beta_2 * v + (1. - beta_2) * T.sqr(gparam)

        # correct estimates for bias
        m_t /= 1. - beta_1**t
        v_t /= 1. - beta_2**t

        # update parameter vector
        param_t = param - ((alpha * m_t) / (T.sqrt(v_t) + epsilon))

        # remember changes to the shared variables
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
        # cache for sum of squared historical gradients
        cache = theano.shared(value=param.get_value() * 0., name='cache')
        cache_t = cache + T.sqr(gparam)

        # per parameter adaptive learning rate
        param_t = param - base_lr * gparam / (T.sqrt(cache_t) + epsilon)

        # remember changes to the shared variables
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
        # leaky cache of sum of squared historical gradients
        cache = theano.shared(param.get_value() * 0., name='cache')
        cache_t = decay_rate * cache + (1. - decay_rate) * T.sqr(gparam)

        # per parameter adaptive learning rate
        param_t = param - base_lr * gparam / (T.sqrt(cache_t) + epsilon)

        # remember changes to the shared variables
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
