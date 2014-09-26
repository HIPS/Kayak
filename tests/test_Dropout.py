import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_nondropout_values():
    npr.seed(1)
    # First sanity check: don't actually drop anything out.
    # Make sure we get everything back.

    np_X = npr.randn(10,20)
    X    = kayak.Parameter(np_X)
    Y    = kayak.Dropout(X, drop_prob=0.0, rng=1)
    
    assert np.all(close_float(Y.value(), np_X))

def test_alldropout_values():
    npr.seed(2)
    # Drop everything out.

    np_X = npr.randn(10,20)
    X    = kayak.Parameter(np_X)
    Y    = kayak.Dropout(X, drop_prob=1.0, rng=1)
    
    assert np.all(Y.value() == 0.0)

def test_dropout_values():
    # Drop some things out.
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        prob = npr.rand()
        scale = 1.0 / (1.0 - prob)

        np_X = npr.randn(5,6)
        X    = kayak.Parameter(np_X)
        Y    = kayak.Dropout(X, drop_prob=prob, rng=1)

        Y.value()

        assert np.all(np.logical_xor(Y.value(reset=False) == 0.0,
                                     close_float(Y.value(reset=False), scale*np_X)))

def test_nondropout_grad():
    npr.seed(4)

    np_X = npr.randn(10,20)
    X    = kayak.Parameter(np_X)
    Y    = kayak.Dropout(X, drop_prob=0.0, rng=1)
    Z    = kayak.MatSum(Y)
    
    Z.value()
    assert Z.grad(X).shape == np_X.shape
    assert kayak.util.checkgrad(X, Z) < MAX_GRAD_DIFF

def test_alldropout_grad():
    npr.seed(5)

    np_X = npr.randn(10,20)
    X    = kayak.Parameter(np_X)
    Y    = kayak.Dropout(X, drop_prob=1.0, rng=1)
    Z    = kayak.MatSum(Y)
    
    Z.value()
    assert Z.grad(X).shape == np_X.shape
    assert kayak.util.checkgrad(X, Z) < MAX_GRAD_DIFF

def test_dropout_grad():
    # Drop some things out.
    npr.seed(6)

    for ii in xrange(NUM_TRIALS):
        prob = npr.rand()
        scale = 1.0 / (1.0 - prob)

        np_X = npr.randn(5,6)
        X    = kayak.Parameter(np_X)
        Y    = kayak.Dropout(X, drop_prob=prob, rng=1)
        Z    = kayak.MatSum(Y)

        Z.value()
        assert Z.grad(X).shape == np_X.shape
        assert kayak.util.checkgrad(X, Z) < MAX_GRAD_DIFF

