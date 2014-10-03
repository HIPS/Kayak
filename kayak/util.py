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
    fd_grad = np.zeros(variable.shape)
    base_value = variable.value.copy()
    for in_dims in it.product(*map(range, variable.shape)):
        small_array = np.zeros(variable.shape)
        small_array[in_dims] = epsilon

        variable.value = base_value - 2*small_array
        fn_l2 = output.value
        variable.value = base_value - small_array
        fn_l1 = output.value
        variable.value = base_value + small_array
        fn_r1 = output.value
        variable.value = base_value + 2*small_array
        fn_r2 = output.value

        fd_grad[in_dims] = ((fn_l2 - fn_r2)/12. + (- fn_l1 + fn_r1)*2./3.) /epsilon # 2nd order method
        # fd_grad[in_dims] = (- fn_l1/2. + fn_r1/2.) /epsilon # 1st order method

        if verbose:
            print np.abs((an_grad[in_dims] - fd_grad[in_dims])/(fd_grad[in_dims]+EPSILON)), an_grad[in_dims], fd_grad[in_dims]

    variable.value = base_value
    print "Mean finite difference", np.mean(np.abs((an_grad - fd_grad)/(fd_grad+EPSILON)))
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

