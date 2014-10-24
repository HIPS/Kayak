import numpy        as np
import numpy.random as npr

import kayak

from . import *
from nose.tools import assert_less

def test_stacking_values():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):
        np_A = npr.randn(6,10)
        np_B = npr.randn(6,5)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        Y    = kayak.Hstack(A,B)

        assert(np.array_equal(Y.value[:, :A.shape[1]], np_A))
        assert(np.array_equal(Y.value[:, A.shape[1]:], np_B))

def test_stacking_grad():
    npr.seed(2)

    for ii in xrange(NUM_TRIALS):
        np_A = npr.randn(6,10)
        np_B = npr.randn(6,5)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        Y    = kayak.Hstack(A, B)
        Z    = kayak.MatSum(Y)

        Z.value
        assert_less(kayak.util.checkgrad(A, Z), MAX_GRAD_DIFF)
        assert_less(kayak.util.checkgrad(B, Z), MAX_GRAD_DIFF)
