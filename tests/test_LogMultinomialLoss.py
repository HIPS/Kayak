import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_vector_value():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        np_pred = npr.randn(1,10)
        np_targ = npr.randn(1,10)
        
        pred = kayak.Parameter(np_pred)
        targ = kayak.Targets(np_targ)
        out  = kayak.LogMultinomialLoss(pred, targ)

        assert close_float(out.value, -np.sum(np_pred * np_targ))

def test_vector_grad():
    npr.seed(4)

    for ii in xrange(NUM_TRIALS):
        np_pred = npr.randn(1,10)
        np_targ = npr.randn(1,10)
        
        pred = kayak.Parameter(np_pred)
        targ = kayak.Targets(np_targ)
        out  = kayak.LogMultinomialLoss(pred, targ)

        assert np.all(close_float(out.grad(pred), -np_targ))
        assert kayak.util.checkgrad(pred, out) < MAX_GRAD_DIFF

def test_matrix_value_1():
    npr.seed(5)

    for ii in xrange(NUM_TRIALS):
        np_pred = npr.randn(10,20)
        np_targ = npr.randn(10,20)
        
        pred = kayak.Parameter(np_pred)
        targ = kayak.Targets(np_targ)
        out  = kayak.LogMultinomialLoss(pred, targ)

        assert np.all(close_float(out.value, -np.sum(np_pred * np_targ, axis=1, keepdims=True)))

def test_matrix_grad():
    npr.seed(6)

    for ii in xrange(NUM_TRIALS):
        np_pred = npr.randn(10,20)
        np_targ = npr.randn(10,20)

        pred = kayak.Parameter(np_pred)
        targ = kayak.Targets(np_targ)
        out  = kayak.MatSum(kayak.LogMultinomialLoss(pred, targ))

        assert kayak.util.checkgrad(pred, out) < MAX_GRAD_DIFF

def test_matrix_value_2():
    npr.seed(7)

    for ii in xrange(NUM_TRIALS):
        np_pred = npr.randn(10,20)
        np_targ = npr.randn(10,20)
        
        pred = kayak.Parameter(np_pred)
        targ = kayak.Targets(np_targ)
        out  = kayak.LogMultinomialLoss(pred, targ, axis=0)

        assert np.all(close_float(out.value, -np.sum(np_pred * np_targ, axis=0)))
