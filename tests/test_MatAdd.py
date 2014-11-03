import numpy        as np
import numpy.random as npr

import kayak

from . import *

# These behaviors requires prepending singeltons. Do we want to keep them?
# def test_0d_plus_2d_scalar_value():
#     npr.seed(1)

#     for ii in xrange(NUM_TRIALS):
#         npX1 = npr.randn(1, 1)
#         X1 = kayak.Parameter( npX1 )
#         npX2 = np.sum(npr.randn()) # generates a scalar with shape ()
#         X2= kayak.Parameter( npX2 )
#         Y = kayak.MatAdd(X1, X2)

#         # Verify that a scalar is reproduced.
#         assert close_float(Y.value, npX1 + npX2)


# def test_0d_plus_2d_scalar_grad():
#     npr.seed(2)
#     for ii in xrange(NUM_TRIALS):
#         npX1 = npr.randn(1, 1)
#         X1 = kayak.Parameter( npX1 )
#         npX2 = np.sum(npr.randn()) # generates a scalar with shape ()
#         X2= kayak.Parameter( npX2 )
#         Y = kayak.MatAdd(X1, X2)

#         # Verify that the gradient is one.
#         assert Y.grad(X1) == 1.0
#         assert Y.grad(X2) == 1.0
#         assert kayak.util.checkgrad(X1, Y) < MAX_GRAD_DIFF
#         assert kayak.util.checkgrad(X2, Y) < MAX_GRAD_DIFF

def test_matadd_values_1():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.MatAdd(A, B)

        assert C.shape == np_A.shape
        assert np.all( close_float(C.value, np_A+np_B))

def test_matadd_values_2():
    npr.seed(2)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,6)
        np_C = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.Parameter(np_C)
        D    = kayak.MatAdd(A, B, C)

        assert D.shape == np_A.shape
        assert np.all( close_float(D.value, np_A+np_B+np_C))

def test_matadd_values_3():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(1,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.MatAdd(A, B)

        assert C.shape == (5,6)
        assert np.all( close_float(C.value, np_A+np_B))

def test_matadd_values_4():
    npr.seed(4)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,1)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.MatAdd(A, B)

        assert C.shape == (5,6)
        assert np.all( close_float(C.value, np_A+np_B))

def test_matadd_values_5():
    npr.seed(5)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(1,6)
        np_B = npr.randn(5,1)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.MatAdd(A, B)

        assert C.shape == (5,6)
        assert np.all( close_float(C.value, np_A+np_B))

def test_matadd_values_6():
    npr.seed(6)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(1,1)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.MatAdd(A, B)

        assert C.shape == (5,6)
        assert np.all( close_float(C.value, np_A+np_B))

def test_matadd_values_7():
    npr.seed(7)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        D    = kayak.MatAdd(A, B, A)

        assert D.shape == (5,6)
        assert np.all( close_float(D.value, 2*np_A + np_B))

def test_matadd_grad_1():
    npr.seed(8)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.MatAdd(A, B)
        D    = kayak.MatSum(C)

        D.value
        assert kayak.util.checkgrad(A, D) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, D) < MAX_GRAD_DIFF

def test_matadd_grad_2():
    npr.seed(9)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,6)
        np_C = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.Parameter(np_C)
        D    = kayak.MatAdd(A, B, C)
        E    = kayak.MatSum(D)

        E.value
        assert kayak.util.checkgrad(A, E) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, E) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(C, E) < MAX_GRAD_DIFF

def test_matadd_grad_3():
    npr.seed(10)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(1,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.MatAdd(A, B)
        D    = kayak.MatSum(C)

        D.value
        print np_A.shape, D.grad(A).shape
        print np_B.shape, D.grad(B).shape
        assert D.grad(A).shape == np_A.shape
        assert D.grad(B).shape == np_B.shape
        assert kayak.util.checkgrad(A, D) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, D) < MAX_GRAD_DIFF

def test_matadd_grad_4():
    npr.seed(11)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,1)
        np_B = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.MatAdd(A, B)
        D    = kayak.MatSum(C)

        D.value
        assert D.grad(A).shape == np_A.shape
        assert D.grad(B).shape == np_B.shape
        assert kayak.util.checkgrad(A, D) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, D) < MAX_GRAD_DIFF

def test_matadd_grad_5():
    npr.seed(12)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,1)
        np_B = npr.randn(1,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.MatAdd(A, B)
        D    = kayak.MatSum(C)

        D.value
        assert D.grad(A).shape == np_A.shape
        assert D.grad(B).shape == np_B.shape
        assert kayak.util.checkgrad(A, D) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, D) < MAX_GRAD_DIFF

def test_matadd_grad_6():
    npr.seed(13)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(1,1)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.MatAdd(A, B)
        D    = kayak.MatSum(C)

        D.value
        assert D.grad(A).shape == np_A.shape
        assert D.grad(B).shape == np_B.shape
        assert kayak.util.checkgrad(A, D) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, D) < MAX_GRAD_DIFF

def test_matadd_grad_7():
    npr.seed(14)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        D    = kayak.MatAdd(A, B, A)
        E    = kayak.MatSum(D)

        E.value
        assert E.grad(A).shape == np_A.shape
        assert E.grad(B).shape == np_B.shape
        assert kayak.util.checkgrad(A, E) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, E) < MAX_GRAD_DIFF

def test_matadd_grad_8():
    npr.seed(15)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        D    = kayak.MatAdd(A, A)
        E    = kayak.MatSum(D)

        E.value
        assert E.grad(A).shape == np_A.shape
        assert kayak.util.checkgrad(A, E) < MAX_GRAD_DIFF
