import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_softrelu_values():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):
        np_X = npr.randn(6,5)
        X    = kayak.Parameter(np_X)
        Y    = kayak.SoftReLU(X)

        assert np.all( Y.value() >= 0.0 )
        assert np.all(close_float(np.log(1.0 + np.exp(np_X)), Y.value()))
        
def test_softrelu_grad():
    npr.seed(2)

    for ii in xrange(NUM_TRIALS):
        np_X = npr.randn(6,5)
        X    = kayak.Parameter(np_X)
        Y    = kayak.SoftReLU(X)
        Z    = kayak.MatSum(Y)

        Z.value()
        assert np.all( Z.grad(X) >= 0.0 )
        assert kayak.util.checkgrad(X, Z) < MAX_GRAD_DIFF
