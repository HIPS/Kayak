import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_transpose_1():
    npr.seed(1)

    np_A = npr.randn(5,10)
    A    = kayak.Parameter(np_A)
    B    = kayak.Transpose(A)

    B.value()
    assert B.shape() == (10,5)
    for ii in xrange(np_A.shape[0]):
        for jj in xrange(np_A.shape[1]):
            assert np_A[ii,jj] == B.value()[jj,ii]

def test_transpose_2():
    npr.seed(2)

    np_A = npr.randn(5,10,15)
    A    = kayak.Parameter(np_A)
    B    = kayak.Transpose(A)

    B.value()
    assert B.shape() == (15,10,5)
    for ii in xrange(np_A.shape[0]):
        for jj in xrange(np_A.shape[1]):
            for kk in xrange(np_A.shape[2]):
                assert np_A[ii,jj,kk] == B.value()[kk,jj,ii]

def test_transpose_3():
    npr.seed(3)

    np_A = npr.randn(5,10)
    A    = kayak.Parameter(np_A)
    B    = kayak.Transpose(A)
    C    = kayak.Parameter(npr.randn(5,5))
    D    = kayak.MatMult(B, C)
    out  = kayak.MatSum(D)

    out.value()
    assert out.grad(A).shape == np_A.shape
    assert kayak.util.checkgrad(A, out) < MAX_GRAD_DIFF

def test_transpose_3():
    npr.seed(3)

    np_A = npr.randn(5,10)
    A    = kayak.Parameter(np_A)
    B    = kayak.Transpose(A)
    C    = kayak.Parameter(npr.randn(5,5))
    D    = kayak.MatMult(B, C)
    out  = kayak.MatSum(D)

    out.value()
    assert out.grad(A).shape == np_A.shape
    assert kayak.util.checkgrad(A, out) < MAX_GRAD_DIFF
