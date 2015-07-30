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

        # compute biased corrected estimates
        m_t /= 1. - beta_1**t
        v_t /= 1. - beta_2**t

        # update parameter vector
        param_t = param - ((alpha * m_t) / (T.sqrt(v_t) + epsilon))

        # propogate changes to the shared variables
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((param, param_t))

    updates.append((t, t + 1))
    return updates


def Adagrad(cost, params, lr=1e-2):
    '''
        TODO!!
    '''
    raise NotImplementedError()


def SGD(cost, params, lr=1e-2):
    '''
        Returns updates for vanilla SGD
        TODO: add momentum and nesterov momentum too!
    '''
    grads = T.grad(cost, params)
    return [(param, param - lr * gparam) for param, gparam in zip(params, grads)]
