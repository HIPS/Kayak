# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import numpy        as np
import numpy.random as npr
import itertools    as it

from . import EPSILON

from constants import Parameter

def checkgrad(input, output, epsilon=1e-4):
    if not isinstance(input, Parameter):
        raise Exception("Cannot evaluate gradient in terms of non-Parameter type %s", (type(input)))

    # Need to make sure all evals have the same random number generation.
    rng_seed = 1

    value   = output.value(reset=True, rng=npr.RandomState(rng_seed))
    an_grad = output.grad(input, reset=False)
    fd_grad = np.zeros(an_grad.shape)

    for in_dims in it.product(*map(range, input.shape(reset=False))):
        small_vect = np.zeros(input.shape(reset=False))
        small_vect[in_dims] = epsilon/2.0

        input.add(small_vect)
        fn_up = output.value(reset=True, rng=npr.RandomState(rng_seed))
        input.add(- 2 * small_vect)
        fn_dn = output.value(reset=True, rng=npr.RandomState(rng_seed))
        input.add(small_vect)
            
        fd_grad[in_dims] = (fn_up - fn_dn)/epsilon

        #print in_dims, (an_grad[in_dims] - fd_grad[in_dims])/np.abs(fd_grad[in_dims]), an_grad[in_dims], fd_grad[in_dims]
        
    return np.mean(np.abs((an_grad - fd_grad)/(fd_grad+EPSILON)))

def broadcast(shape1, shape2):
    shape1 = list(shape1)
    shape2 = list(shape2)
    d1 = 1 if len(shape1) == 0 else shape1.pop()
    d2 = 1 if len(shape2) == 0 else shape2.pop()
    if d1 > 1 and d2 > 1 and d1 != d2:
        raise Exception("Invalid shapes for broadcast.")
    if len(shape1) == 0 and len(shape2) == 0:
        return (max(d1, d2),)
    else:
        return tuple(list(broadcast(shape1, shape2)) + [max(d1,d2),])

def logsumexp(X, axis=None):
    maxes = np.expand_dims(np.max(X, axis=axis), axis=axis)
    return np.expand_dims(np.log(np.sum(np.exp(X - maxes), axis=axis)), axis=axis) + maxes

def onehot(T, num_labels=None):
    if num_labels is None:
        num_labels = np.max(T)+1
    labels = np.zeros((T.shape[0], num_labels), dtype=bool)
    labels[np.arange(T.shape[0], dtype=int), T] = 1
    return labels

