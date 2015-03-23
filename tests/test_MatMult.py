import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_matmult_values_1():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(6,7)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.MatMult(A, B)

        assert C.value.shape == (5,7)
        assert np.all(close_float(C.value, np.dot(np_A, np_B)))

def test_matmult_values_2():
    npr.seed(2)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,5)
        A    = kayak.Parameter(np_A)
        C    = kayak.MatMult(A, A)

        assert C.value.shape == (5,5)
        assert np.all(close_float(C.value, np.dot(np_A, np_A)))

def test_matmult_values_3():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(6,7)
        np_C = npr.randn(7,8)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.Parameter(np_C)
        D    = kayak.MatMult(A, B, C)

        assert D.value.shape == (5,8)
        assert np.all(close_float(D.value, np.dot(np_A, np.dot(np_B, np_C))))

def test_matmult_grad_1():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(6,7)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.MatMult(A, B)
        D    = kayak.MatSum(C)

        D.value
        assert D.grad(A).shape == (5,6)
        assert D.grad(B).shape == (6,7)
        assert kayak.util.checkgrad(A, D) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, D) < MAX_GRAD_DIFF

def test_matmult_grad_2():
    npr.seed(4)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,5)
        A    = kayak.Parameter(np_A)
        C    = kayak.MatMult(A, A)
        D    = kayak.MatSum(C)

        D.value
        assert D.grad(A).shape == (5,5)
        assert kayak.util.checkgrad(A, D) < MAX_GRAD_DIFF

def test_matmult_grad_3():
    npr.seed(5)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(6,7)
        np_C = npr.randn(7,8)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.Parameter(np_C)
        D    = kayak.MatMult(A, B, C)
        E    = kayak.MatSum(kayak.SoftReLU(D))

        assert E.grad(A).shape == (5,6)
        assert E.grad(B).shape == (6,7)
        assert E.grad(C).shape == (7,8)
        assert kayak.util.checkgrad(A, E) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, E) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(C, E) < MAX_GRAD_DIFF

def test_matmult_grad_mat_vect():
    npr.seed(5)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(6)
        np_C = npr.randn(5,)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.Parameter(np_C)
        D    = kayak.MatMult(A, B)
        E    = kayak.MatSum(kayak.ElemMult(C, D))

        assert E.grad(A).shape == (5,6)
        assert E.grad(B).shape == (6,)
        assert kayak.util.checkgrad(A, E) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, E) < MAX_GRAD_DIFF

def test_matmult_grad_vect_mat():
    npr.seed(5)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(6,)
        np_B = npr.randn(6,7)
        np_C = npr.randn(7,)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.Parameter(np_C)
        D    = kayak.MatMult(A, B)
        E    = kayak.MatSum(kayak.ElemMult(C, D))

        assert E.grad(A).shape == (6,)
        assert E.grad(B).shape == (6, 7)
        assert kayak.util.checkgrad(A, E) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, E) < MAX_GRAD_DIFF
