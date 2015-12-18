import numpy.random as npr
import numpy as np
import time
import os

def make_minibatch_x(data, batch_size, num_iter):
    '''
    assume data is a N x D matrix, this method creates mini-batches
    by draw each mini-batch with replacement from the data for num_iter runs.
    '''
    N = data.shape[0]
    D = data.shape[1]
    mini_batch = np.zeros((batch_size, D))
    assert batch_size <= N
    for it in range(num_iter):
        ind = npr.choice(range(N), size=batch_size, replace=True)
        mini_batch[:, :] = data[ind, :]
        yield mini_batch

def make_minibatch_x_y(data, targets, batch_size, num_iter):
    '''
    assume data is a N x D matrix, this method creates mini-batches
    by draw each mini-batch with replacement from the data for num_iter runs.
    '''
    N = data.shape[0]
    Np = targets.shape[0]
    D = data.shape[1]
    Dp = targets.shape[1]
    batch_shape = list(data.shape)
    batch_shape[0] = batch_size
    mini_batch = np.zeros(batch_shape)
    mini_batch_targets = np.zeros((batch_size, Dp))
    assert N == Np
    assert batch_size <= N
    for it in range(num_iter):
        ind = npr.choice(range(N), size=batch_size, replace=True)
        if len(mini_batch.shape) == 2: # matrix data.
            mini_batch[:, :] = data[ind, :]
        elif len(mini_batch.shape) == 4: # tensor data.
            mini_batch[:, :, :, :] = data[ind, :, :, :]
        mini_batch_targets[:, :] = targets[ind, :]
        yield mini_batch, mini_batch_targets

def train_test_split(dataset, training_ratio = 0.6):
    indices = npr.choice(range(len(dataset)), int(len(dataset) * training_ratio), replace=False)
    train_set = [dataset[ind] for ind in indices]
    test_set = [dataset[ind] for ind in range(len(dataset)) if ind not in indices]
    return (train_set, test_set)

def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        print '[%s]' % self.name, 'Start'
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight = False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(unicode(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

MESSAGE_DEPTH = 0
class Message(object):
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        global MESSAGE_DEPTH #pylint: disable=W0603
        print colorize('\t'*MESSAGE_DEPTH + '=: ' + self.msg,'magenta')
        self.tstart = time.time()
        MESSAGE_DEPTH += 1

    def __exit__(self, etype, *args):
        global MESSAGE_DEPTH #pylint: disable=W0603
        MESSAGE_DEPTH -= 1
        maybe_exc = "" if etype is None else " (with exception)"
        print colorize('\t'*MESSAGE_DEPTH + "done%s in %.3f seconds"%(maybe_exc, time.time() - self.tstart), 'magenta')

def outdir_from_environ():
    '''
    parse experiment output dir from environment variables
    create directory if necessary
    '''
    outdir = os.environ.get('outdir')
    outdir = outdir if outdir else ''
    if os.path.exists(outdir):
        raise Exception('output directory already exists!')
    mkdir_if_not_exist(outdir)
    return outdir

