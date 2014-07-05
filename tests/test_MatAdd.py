import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_matadd_values_1():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.MatAdd(A, B)

        assert C.shape() == np_A.shape
        assert np.all( close_float(C.value(True), np_A+np_B))

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

        assert D.shape() == np_A.shape
        assert np.all( close_float(D.value(True), np_A+np_B+np_C))

def test_matadd_values_3():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(1,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.MatAdd(A, B)

        assert C.shape() == (5,6)
        assert np.all( close_float(C.value(True), np_A+np_B))

def test_matadd_values_4():
    npr.seed(4)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,1)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.MatAdd(A, B)

        assert C.shape() == (5,6)
        assert np.all( close_float(C.value(True), np_A+np_B))

def test_matadd_values_5():
    npr.seed(5)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(1,6)
        np_B = npr.randn(5,1)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.MatAdd(A, B)

        assert C.shape() == (5,6)
        assert np.all( close_float(C.value(True), np_A+np_B))

def test_matadd_values_6():
    npr.seed(6)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(1,1)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.MatAdd(A, B)

        assert C.shape() == (5,6)
        assert np.all( close_float(C.value(True), np_A+np_B))

def test_matadd_values_7():
    npr.seed(7)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        D    = kayak.MatAdd(A, B, A)

        assert D.shape() == (5,6)
        assert np.all( close_float(D.value(True), 2*np_A + np_B))

def test_matadd_grad_1():
    npr.seed(8)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.MatAdd(A, B)
        D    = kayak.MatSum(C)

        D.value(True)
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

        E.value(True)
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

        D.value(True)
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

        D.value(True)
        assert D.grad(A).shape == np_A.shape
        assert D.grad(B).shape == np_B.shape
        assert kayak.util.checkgrad(A, D) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, D) < MAX_GRAD_DIFF
