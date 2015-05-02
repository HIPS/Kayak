import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_batchnorm_values_1():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):

        np_X = npr.randn(5,4)
        np_A = npr.randn(4,2)
        A    = kayak.Parameter(np_A)
        X    = kayak.Parameter(np_X)
        Y    = kayak.BatchNormalize(X)        
        J    = kayak.TanH(kayak.MatMult(Y,A))
        Z    = kayak.MatSum(J)

        mu   = np.mean(np_X, axis=0, keepdims=True)
        sig  = np.mean((np_X - mu)**2, axis=0, keepdims=True) + 1e-6
        np_Y = (np_X - mu) / np.sqrt(sig)

        assert np.all(close_float(Y.value, np_Y))
        assert kayak.util.checkgrad(X, Z, verbose=True) < MAX_GRAD_DIFF
