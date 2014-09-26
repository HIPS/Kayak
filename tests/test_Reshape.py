import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_reshape_1():
    npr.seed(1)

    np_A = npr.randn(5,10)
    A    = kayak.Parameter(np_A)
    B    = kayak.Reshape(A, (25,2))

    B.value
    assert B.shape == (25,2)

def test_reshape_2():
    npr.seed(2)

    np_A = npr.randn(5,10)
    A    = kayak.Parameter(np_A)
    B    = kayak.Reshape(A, (2,25))
    C    = kayak.Parameter(npr.randn(25,5))
    D    = kayak.MatMult(B, C)
    out  = kayak.MatSum(D)

    out.value
    assert out.grad(A).shape == np_A.shape
    assert kayak.util.checkgrad(A, out) < MAX_GRAD_DIFF


