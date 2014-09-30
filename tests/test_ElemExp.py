import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_elemexp_values_1():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        C    = kayak.ElemExp(A)

        assert C.shape == np_A.shape
        assert np.all( close_float(C.value, np.exp(np_A)))

def test_elemexp_values_2():
    npr.seed(2)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(1)
        A    = kayak.Parameter(np_A)
        D    = kayak.ElemExp(A)

        assert D.shape == np_A.shape
        assert np.all( close_float(D.value, np.exp(np_A)))

def test_elemexp_grad_1():
    npr.seed(8)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        C    = kayak.ElemExp(A)
        D    = kayak.MatSum(C)

        D.value
        assert kayak.util.checkgrad(A, D) < MAX_GRAD_DIFF

def test_elemexp_grad_2():
    npr.seed(9)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(1)
        A    = kayak.Parameter(np_A)
        D    = kayak.ElemExp(A)
        E    = kayak.MatSum(D)

        E.value
        assert kayak.util.checkgrad(A, E) < MAX_GRAD_DIFF
