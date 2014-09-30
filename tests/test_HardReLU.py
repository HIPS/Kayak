import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_hardrelu_values():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):
        np_X = npr.randn(6,5)
        X    = kayak.Parameter(np_X)
        Y    = kayak.HardReLU(X)

        assert np.all( Y.value >= 0.0 )
        assert np.all(np.maximum(np_X, 0.0) == Y.value)
        
def test_hardrelu_grad():
    npr.seed(2)

    # Needs to be small due to non-differentiability.
    epsilon = 1e-6

    for ii in xrange(NUM_TRIALS):
        np_X = npr.randn(6,5)
        X    = kayak.Parameter(np_X)
        Y    = kayak.HardReLU(X)
        Z    = kayak.MatSum(Y)

        Z.value
        assert np.all( Z.grad(X) >= 0.0 )
        print "CHECKGRAD: ", ii, kayak.util.checkgrad(X, Z, epsilon)
        assert kayak.util.checkgrad(X, Z, epsilon) < MAX_GRAD_DIFF
