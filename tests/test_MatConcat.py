import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_matconcat_values_1():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)

        C    = kayak.Concatenate(0, A, B)
        assert C.value.shape == (10,6)

        C    = kayak.Concatenate(1, A, B)
        assert C.value.shape == (5,12)



def test_matconcat_grad_1():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        
        np_A = npr.randn(5,6)
        np_B = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.Concatenate(0, A, B)
        D    = kayak.MatSum(C)

        D.value
        assert D.grad(A).shape == (5,6)
        assert D.grad(B).shape == (5,6)
        assert kayak.util.checkgrad(A, D) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, D) < MAX_GRAD_DIFF


def test_matconcat_grad_2():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):

        np_A = npr.randn(5,6)
        np_B = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        B    = kayak.Parameter(np_B)
        C    = kayak.Concatenate(1, A, B)
        D    = kayak.MatSum(C)

        D.value
        assert D.grad(A).shape == (5,6)
        assert D.grad(B).shape == (5,6)
        assert kayak.util.checkgrad(A, D) < MAX_GRAD_DIFF
        assert kayak.util.checkgrad(B, D) < MAX_GRAD_DIFF


def test_matconcat_grad_3():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):

        np_A = npr.randn(5,6)
        A    = kayak.Parameter(np_A)
        C    = kayak.Concatenate(0, A, A)
        D    = kayak.MatSum(C)

        D.value
        assert D.grad(A).shape == (5,6)
        assert kayak.util.checkgrad(A, D) < MAX_GRAD_DIFF



