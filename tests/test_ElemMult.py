import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_elemmult_values_1():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.ElemMult(A, B)

        assert C.shape == np_A.shape
        assert np.all( close_float(C.value, np_A*np_B))

def test_elemmult_values_2():
    npr.seed(2)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,6)
        np_C = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.Parameter(np_C)
        D    = kayak.ElemMult(A, B, C)

        assert D.shape == np_A.shape
        assert np.all( close_float(D.value, np_A*np_B*np_C))

def test_elemmult_values_3():
    npr.seed(7)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        D    = kayak.ElemMult(A, B, A)

        assert D.shape == (5,6)
        assert np.all( close_float(D.value, np_A**2 * np_B))

def test_elemmult_values_4():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):

        np_A = npr.randn(5,1)
        np_B = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.ElemMult(A, B)

        assert C.shape == (5,6)
        assert np.all( close_float(C.value, np_A*np_B))

def test_elemmult_values_5():
    npr.seed(2)

    for ii in xrange(NUM_TRIALS):

        np_A = npr.randn(5,1)
        np_B = npr.randn(1,6)
        np_C = npr.randn(1,1)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.Parameter(np_C)
        D    = kayak.ElemMult(A, B, C)

        assert D.shape == (5,6)
        assert np.all( close_float(D.value, np_A*np_B*np_C))

def test_elemmult_values_6():
    npr.seed(7)

    for ii in xrange(NUM_TRIALS):

        np_A = npr.randn(1,1)
        np_B = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        D    = kayak.ElemMult(A, B, A)

        assert D.shape == (5,6)
        assert np.all( close_float(D.value, np_A**2 * np_B))


def test_elemmult_grad_1():
    npr.seed(8)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.ElemMult(A, B)
        D    = kayak.MatSum(C)

        D.value
        assert kayak.util.checkgrad(A, D) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, D) < MAX_GRAD_DIFF

def test_elemmult_grad_2():
    npr.seed(9)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,6)
        np_C = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.Parameter(np_C)
        D    = kayak.ElemMult(A, B, C)
        E    = kayak.MatSum(D)

        E.value
        assert E.grad(A).shape == np_A.shape
        assert E.grad(B).shape == np_B.shape
        assert E.grad(C).shape == np_C.shape
        assert kayak.util.checkgrad(A, E) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, E) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(C, E) < MAX_GRAD_DIFF

def test_elemmult_grad_3():
    npr.seed(14)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        D    = kayak.ElemMult(A, B, A)
        E    = kayak.MatSum(D)

        E.value
        assert E.grad(A).shape == np_A.shape
        assert E.grad(B).shape == np_B.shape
        assert kayak.util.checkgrad(A, E) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, E) < MAX_GRAD_DIFF

def test_elemmult_grad_4():
    npr.seed(15)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        D    = kayak.ElemMult(A, A)
        E    = kayak.MatSum(D)

        E.value
        assert E.grad(A).shape == np_A.shape
        assert kayak.util.checkgrad(A, E) < MAX_GRAD_DIFF

def test_elemmult_grad_5():
    npr.seed(8)

    for ii in xrange(NUM_TRIALS):

        np_A = npr.randn(5,1)
        np_B = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.ElemMult(A, B)
        D    = kayak.MatSum(C)

        D.value
        assert D.grad(A).shape == np_A.shape
        assert D.grad(B).shape == np_B.shape
        assert kayak.util.checkgrad(A, D) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, D) < MAX_GRAD_DIFF

def test_elemmult_grad_6():
    npr.seed(9)

    for ii in xrange(NUM_TRIALS):

        np_A = npr.randn(5,1)
        np_B = npr.randn(1,6)
        np_C = npr.randn(1,1)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.Parameter(np_C)
        D    = kayak.ElemMult(A, B, C)
        E    = kayak.MatSum(D)

        E.value
        assert E.grad(A).shape == np_A.shape
        assert E.grad(B).shape == np_B.shape
        assert E.grad(C).shape == np_C.shape
        assert kayak.util.checkgrad(A, E) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, E) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(C, E) < MAX_GRAD_DIFF

def test_elemmult_grad_7():
    npr.seed(14)

    for ii in xrange(NUM_TRIALS):

        np_A = npr.randn(5,6)
        np_B = npr.randn(1,1)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        D    = kayak.ElemMult(A, B, A)
        E    = kayak.MatSum(D)

        E.value
        assert E.grad(A).shape == np_A.shape
        assert E.grad(B).shape == np_B.shape
        assert kayak.util.checkgrad(A, E) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, E) < MAX_GRAD_DIFF

def test_elemmult_grad_8():
    npr.seed(15)

    for ii in xrange(NUM_TRIALS):

        np_A = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        D    = kayak.ElemMult(A, A)
        E    = kayak.MatSum(D)

        assert E.grad(A).shape == np_A.shape
        assert kayak.util.checkgrad(A, E) < MAX_GRAD_DIFF
