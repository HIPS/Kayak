import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_elempower_values_1():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        C    = kayak.ElemPower(A, 2)

        assert C.shape == np_A.shape
        assert np.all( close_float(C.value, np.power(np_A, 2)))

def test_elempower_values_2():
    npr.seed(2)

    for ii in xrange(NUM_TRIALS):

        # Only nonnegative values allowed
        np_A = -np.log(npr.rand(1))
        A    = kayak.Parameter(np_A)
        D    = kayak.ElemPower(A, 0.5)

        assert D.shape == np_A.shape
        assert np.all( close_float(D.value, np.power(np_A, 0.5)))

def test_elempower_values_3():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):

        np_A = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        C    = kayak.ElemPower(A, -1)

        assert C.shape == np_A.shape
        assert np.all( close_float(C.value, np.power(np_A, -1)))

def test_elempower_values_4():
    npr.seed(2)

    for ii in xrange(NUM_TRIALS):

        np_A = npr.randn(1)
        A    = kayak.Parameter(np_A)
        D    = kayak.ElemPower(A, 3.)

        assert D.shape == np_A.shape
        assert np.all( close_float(D.value, np.power(np_A, 3.)))

def test_elempower_grad_1():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)

        # Avoid small values where the inverse is unstable
        err = np.where(abs(np_A) < 1e-2)
        np_A[err] = 1e-2 * np.sign(np_A[err])

        A    = kayak.Parameter(np_A)
        C    = kayak.ElemPower(A, -1)
        D    = kayak.MatSum(C)

        D.value
        assert kayak.util.checkgrad(A, D) < MAX_GRAD_DIFF

def test_elempower_grad_2():
    npr.seed(9)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(1)
        A    = kayak.Parameter(np_A)
        D    = kayak.ElemPower(A, 2)
        E    = kayak.MatSum(D)

        E.value
        assert kayak.util.checkgrad(A, E) < MAX_GRAD_DIFF
