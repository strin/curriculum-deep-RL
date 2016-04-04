import numpy.random as npr
import numpy as np
import time
import os
import sys
from datetime import datetime

from StringIO import StringIO
from pprint import pprint

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
    if path == '':
        return
    if not os.path.exists(path):
        os.makedirs(path)

def get_runid():
    return datetime.now().strftime('%m-%d-%y-%H-%M-%S.%f')


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


class Timer(object):
    def __init__(self, name=None, output=sys.stdout):
        self.name = name
        if output and type(output) == str:
            self.output = open(output, 'w')
        else:
            self.output = output

    def __enter__(self):
        if self.name:
            print >>self.output, colorize('[%s]\t' % self.name, 'green'),
        print >>self.output, colorize('Start', 'green')
        self.tstart = time.time()
        self.output.flush()

    def __exit__(self, type, value, traceback):
        if self.name:
            print >>self.output, colorize('[%s]\t' % self.name, 'green'),
        print >>self.output, colorize('Elapsed: %s' % (time.time() - self.tstart),
                                      'green')
        self.output.flush()


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


def to_string(obj):
    if type(obj) == dict:
       buf = StringIO()
       pprint(obj, buf)
       buf.seek(0)
       res = buf.read()
       buf.close()
       return res
    else:
        raise TypeError('Unsupported type %s for to_string' % str(type(obj)))

