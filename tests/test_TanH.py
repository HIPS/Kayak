import numpy        as np
import numpy.random as npr

import kayak

from . import *
from nose.tools import assert_less

def test_tanh_values():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):
        np_X = npr.randn(6,5)
        X    = kayak.Parameter(np_X)
        Y    = kayak.TanH(X)

        assert np.all(close_float(np.tanh(np_X), Y.value))
        
def test_tanh_grad():
    npr.seed(2)

    for ii in xrange(NUM_TRIALS):
        np_X = npr.randn(6,5)
        X    = kayak.Parameter(np_X)
        Y    = kayak.TanH(X)
        Z    = kayak.MatSum(Y)

        Z.value
        assert np.all( Z.grad(X) >= 0.0 )
        assert_less(kayak.util.checkgrad(X, Z), MAX_GRAD_DIFF)
