import numpy        as np
import numpy.random as npr
import scipy.linalg as spla
import kayak

from . import *
from nose.tools import assert_less, assert_equal

def test_matdet_values_1():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(12,6)
        A    = np.dot(np_A.T, np_A) + 1e-6*np.eye(6)
        B    = kayak.Parameter(A)
        D    = kayak.MatDet(B)

        assert_less((D.value - spla.det(A))**2, 1e-6)

def test_matdet_grad_1():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(12,6)
        A    = np.dot(np_A.T, np_A) + 1e-6*np.eye(6)
        B    = kayak.Parameter(A)
        D    = kayak.MatDet(B)

        assert_less((D.value - spla.det(A))**2, 1e-6)

        assert_equal(D.grad(B).shape, B.shape)
        assert_less(kayak.util.checkgrad(B, D), MAX_GRAD_DIFF)

