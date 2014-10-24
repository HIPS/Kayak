import numpy        as np
import numpy.random as npr

import kayak

from . import *
from nose.tools import assert_less

def test_indexing_values():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):
        np_X = npr.randn(6,10)
        inds = npr.permutation(10)[:5]
        X    = kayak.Parameter(np_X)
        Y    = kayak.Take(X, inds,axis=1)
        assert(np.array_equal(Y.value, np.take(np_X, inds,axis=1)))
        
def test_indexing_grad():
    npr.seed(2)

    for ii in xrange(NUM_TRIALS):
        np_X = npr.randn(6,20)
        inds = npr.permutation(20)[:5]
        X    = kayak.Parameter(np_X)
        Y    = kayak.Take(X, inds,axis=1)
        Z    = kayak.MatSum(Y)

        Z.value
        assert_less(kayak.util.checkgrad(X, Z), MAX_GRAD_DIFF)
