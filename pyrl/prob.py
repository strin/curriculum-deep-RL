# probability utils.
import numpy as np
import numpy.random as npr

def normalize_log(arr):
    arr = np.array(arr)
    max_ele = np.max(arr)
    arr -= max_ele
    arr -= np.log(np.sum(np.exp(arr)))
    return arr

def choice(objs, size, replace=True, p=None):
    all_inds = range(len(objs))
    inds = npr.choice(all_inds, size=size, replace=replace, p=p)
    return [objs[ind] for ind in inds]

def TV(logprob1, logprob2):
    return np.sum(np.abs(np.exp(logprob1) - np.exp(logprob2)))
