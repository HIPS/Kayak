import numpy        as np
import numpy.random as npr

import kayak

from . import *
from nose.tools import assert_equals, assert_less

def test_convolve1d_1():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(6,7)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.Convolve1d(A, B, ncolors=1)

        # If the filters are the same size as the data
        assert C.value.shape == (5,7)

def test_convolve1d_2():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,20)
        np_B = npr.randn(6,4)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.Convolve1d(A, B, ncolors=1)

        assert_equals(C.value.shape, (5,(20-6+1)*4))

def test_convolve1d_3():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,50)
        np_B = npr.randn(6*5,4)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.Convolve1d(A, B, ncolors=5)

        assert_equals(C.value.shape, (5,(10-6+1)*4))

def test_convolve1d_grad_1():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(6,7)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.Convolve1d(A, B)
        D    = kayak.MatSum(C)

        D.value
        assert_equals(D.grad(A).shape, (5,6))
        assert_equals(D.grad(B).shape, (6,7))
        assert_less(kayak.util.checkgrad(A, D), MAX_GRAD_DIFF)
        assert_less(kayak.util.checkgrad(B, D), MAX_GRAD_DIFF)

def test_pool_1():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Pool(A, width=2)
        C    = kayak.MatSum(B)

        C.value
        assert_equals(C.grad(A).shape, (5,6))
        assert_equals(C.grad(B).shape, (5,3))
        assert_less(kayak.util.checkgrad(A, C), MAX_GRAD_DIFF)

def test_pool_2():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5, 6*4)
        A    = kayak.Parameter(np_A)
        B    = kayak.Pool(A, width=2, ncolors=4)
        C    = kayak.MatSum(B)

        C.value
        assert_equals(C.grad(A).shape, (5, 6*4))
        assert_equals(C.grad(B).shape, (5, 12))
        assert_equals(B.shape, (5, 12))
        assert_less(kayak.util.checkgrad(A, C), MAX_GRAD_DIFF)

def test_pool_offwidth_1():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,7)
        A    = kayak.Parameter(np_A)
        B    = kayak.Pool(A, width=3)
        C    = kayak.MatSum(B)

        C.value
        assert_equals(C.grad(A).shape, (5,7))
        assert_equals(C.grad(B).shape, (5,3))
        assert_less(kayak.util.checkgrad(A, C), MAX_GRAD_DIFF)

def test_pool_offwidth_2():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5, 7*4)
        A    = kayak.Parameter(np_A)
        B    = kayak.Pool(A, width=3, ncolors=4)
        C    = kayak.MatSum(B)

        C.value
        assert_equals(C.grad(A).shape, (5, 7*4))
        assert_equals(C.grad(B).shape, (5, 12))
        assert_equals(B.shape, (5, 12))
        assert_less(kayak.util.checkgrad(A, C), MAX_GRAD_DIFF)

def test_topkpool_1():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,9)
        A    = kayak.Parameter(np_A)
        B    = kayak.TopKPool(A, k=5)
        C    = kayak.MatSum(B)

        C.value
        assert_equals(C.grad(A).shape, (5,9))
        assert_equals(C.grad(B).shape, (5,5))
        assert_less(kayak.util.checkgrad(A, C), MAX_GRAD_DIFF)

def test_convolve1d_grad_2():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,50)
        np_B = npr.randn(6,7)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.Convolve1d(A, B)
        D    = kayak.MatSum(C)

        D.value
        assert_equals(D.grad(A).shape, (5,50))
        assert_equals(D.grad(B).shape, (6,7))
        assert_less(kayak.util.checkgrad(A, D), MAX_GRAD_DIFF)
        assert_less(kayak.util.checkgrad(B, D), MAX_GRAD_DIFF)

def test_convolve1d_grad_2():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,50)
        np_B = npr.randn(6*5,4)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.Convolve1d(A, B, ncolors=5)
        D    = kayak.MatSum(C)

        D.value
        assert_equals(D.grad(A).shape, (5,50))
        assert_equals(D.grad(B).shape, (6*5,4))
        assert_less(kayak.util.checkgrad(A, D), MAX_GRAD_DIFF)
        assert_less(kayak.util.checkgrad(B, D), MAX_GRAD_DIFF)