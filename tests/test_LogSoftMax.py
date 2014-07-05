import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_logsoftmax_values_1():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):

        np_X = npr.randn(5,6)
        X    = kayak.Parameter(np_X)
        Y    = kayak.LogSoftMax(X)

        np_Y = np.exp(np_X)
        np_Y = np_Y / np.sum(np_Y, axis=1)[:,np.newaxis]
        np_Y = np.log(np_Y)

        assert Y.shape() == np_X.shape
        assert np.all(close_float(Y.value(True), np_Y))

def test_logsoftmax_values_2():
    npr.seed(2)

    for ii in xrange(NUM_TRIALS):

        np_X = npr.randn(5,6)
        X    = kayak.Parameter(np_X)
        Y    = kayak.LogSoftMax(X, axis=0)

        np_Y = np.exp(np_X)
        np_Y = np_Y / np.sum(np_Y, axis=0)[np.newaxis,:]
        np_Y = np.log(np_Y)

        assert Y.shape() == np_X.shape
        assert np.all(close_float(Y.value(True), np_Y))

def test_logsoftmax_grad_1():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):

        np_X = npr.randn(5,6)
        X    = kayak.Parameter(np_X)
        Y    = kayak.LogSoftMax(X)
        Z    = kayak.MatSum(Y)

        assert kayak.util.checkgrad(X, Z) < MAX_GRAD_DIFF
        
def test_logsoftmax_grad_2():
    npr.seed(4)

    for ii in xrange(NUM_TRIALS):

        np_X = npr.randn(5,6)
        X    = kayak.Parameter(np_X)
        Y    = kayak.LogSoftMax(X, axis=0)
        Z    = kayak.MatSum(Y)

        assert kayak.util.checkgrad(X, Z) < MAX_GRAD_DIFF

def test_logsoftmax_grad_3():
    npr.seed(5)

    for ii in xrange(NUM_TRIALS):

        np_X = npr.randn(5,6)
        np_T = npr.randint(0, 10, np_X.shape)
        X    = kayak.Parameter(np_X)
        T    = kayak.Targets(np_T)
        Y    = kayak.LogSoftMax(X)
        Z    = kayak.MatSum(kayak.LogMultinomialLoss(Y, T))

        assert kayak.util.checkgrad(X, Z) < MAX_GRAD_DIFF
        
