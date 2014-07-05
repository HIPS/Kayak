import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_graph_chain():
    N  = 10
    D  = 5
    H1 = 6
    H2 = 7

    X  = kayak.Inputs(npr.randn(N,D))
    W1 = kayak.Parameter(npr.randn(D,H1))
    W2 = kayak.Parameter(npr.randn(H1,H2))
    W3 = kayak.Parameter(npr.randn(H2,1))

    U1 = kayak.SoftReLU(kayak.MatMult(X, W1))
    U2 = kayak.SoftReLU(kayak.MatMult(U1, W2))
    U3 = kayak.SoftReLU(kayak.MatMult(U2, W3))
    
    out = kayak.MatSum(U3)

    out.value(True)
    assert kayak.util.checkgrad(W1, out) < MAX_GRAD_DIFF
    assert kayak.util.checkgrad(W2, out) < MAX_GRAD_DIFF
    assert kayak.util.checkgrad(W3, out) < MAX_GRAD_DIFF

def test_graph_diamond():
    N  = 10
    D  = 5
    H1 = 6
    H2 = 7

    X   = kayak.Inputs(npr.randn(N,D))
    W1  = kayak.Parameter(npr.randn(D,H1))
    W2a = kayak.Parameter(npr.randn(H1,H2))
    W2b = kayak.Parameter(npr.randn(H1,H2))
    W3  = kayak.Parameter(npr.randn(H2,1))

    U1 = kayak.SoftReLU(kayak.MatMult(X, W1))
    U2a = kayak.SoftReLU(kayak.MatMult(U1, W2a))
    U2b = kayak.SoftReLU(kayak.MatMult(U1, W2b))
    U3a = kayak.SoftReLU(kayak.MatMult(U2a, W3))
    U3b = kayak.SoftReLU(kayak.MatMult(U2b, W3))
    
    out = kayak.MatSum(kayak.MatAdd(U3a, U3b))

    out.value(True)
    assert kayak.util.checkgrad(W1, out) < MAX_GRAD_DIFF
    assert kayak.util.checkgrad(W2a, out) < MAX_GRAD_DIFF
    assert kayak.util.checkgrad(W2b, out) < MAX_GRAD_DIFF
    assert kayak.util.checkgrad(W3, out) < MAX_GRAD_DIFF



    
