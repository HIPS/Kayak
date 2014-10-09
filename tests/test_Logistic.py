import numpy        as np
import numpy.random as npr

import kayak

from . import *
from nose.tools import assert_less

def test_logistic_values():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):
        np_X = npr.randn(6,5)
        X    = kayak.Parameter(np_X)
        Y    = kayak.Logistic(X)

        assert np.all(close_float(1.0/(1.0+np.exp(-np_X)), Y.value))
        
def test_logistic_grad():
    npr.seed(2)

    for ii in xrange(NUM_TRIALS):
        np_X = npr.randn(6,5)
        X    = kayak.Parameter(np_X)
        Y    = kayak.Logistic(X)
        Z    = kayak.MatSum(Y)

        Z.value
        assert np.all( Z.grad(X) >= 0.0 )
        assert_less(kayak.util.checkgrad(X, Z), MAX_GRAD_DIFF)
