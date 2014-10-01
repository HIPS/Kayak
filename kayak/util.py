# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import numpy        as np
import numpy.random as npr
import itertools    as it

from . import EPSILON

from constants import Parameter

def checkgrad(variable, output, epsilon=1e-4, verbose=False):
    if not isinstance(variable, Parameter):
        raise Exception("Cannot evaluate gradient in terms of non-Parameter type %s", (type(variable)))

    # Need to make sure all evals have the same random number generation.
    rng_seed = 1

    value = output.value
    an_grad = output.grad(variable)
    fd_grad = np.zeros(an_grad.shape)

    for in_dims in it.product(*map(range, variable.shape)):
        small_array = np.zeros(variable.shape)
        small_array[in_dims] = epsilon/2.0
        variable.value += small_array
        fn_up = output.value
        variable.value -= 2 * small_array
        fn_dn = output.value
        variable.value += small_array
        fd_grad[in_dims] = (fn_up - fn_dn)/epsilon
        
        # Print finite diffs for individual parameters
        if verbose:
            print np.abs((an_grad[in_dims] - fd_grad[in_dims])/(fd_grad[in_dims]+EPSILON)), an_grad[in_dims], fd_grad[in_dims]

    return np.mean(np.abs((an_grad - fd_grad)/(fd_grad+EPSILON)))

def logsumexp(X, axis=None):
    maxes = np.max(X, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(X - maxes), axis=axis, keepdims=True)) + maxes

def onehot(T, num_labels=None):
    if num_labels is None:
        num_labels = np.max(T)+1
    labels = np.zeros((T.shape[0], num_labels), dtype=bool)
    labels[np.arange(T.shape[0], dtype=int), T] = 1
    return labels

