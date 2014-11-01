import numpy        as np
import numpy.random as npr

import kayak

from . import *

def check_tensormult(A_shape, B_shape, axes):

    np_A = npr.randn(*A_shape)
    np_B = npr.randn(*B_shape)
    A = kayak.Parameter(np_A)
    B = kayak.Parameter(np_B)
    C = kayak.TensorMult(A, B, axes)
    D = kayak.Parameter(npr.randn(*C.shape))
    L = kayak.MatSum(kayak.ElemMult(C, D))
    
    assert np.all(close_float(C.value, np.tensordot(np_A, np_B, axes)))
    assert kayak.util.checkgrad(A, L) < MAX_GRAD_DIFF
    assert kayak.util.checkgrad(B, L) < MAX_GRAD_DIFF

def test_matmult_grad_1():
    check_tensormult((3, 4), (4, 5), ((1,), (0,)))

def test_matmult_grad_2():
    check_tensormult((4, 3), (5, 4), ((0,), (1,)))

def test_matmult_grad_3():
    check_tensormult((3, 4), (4, 5, 6), ((1,), (0,)))

def test_matmult_grad_4():
    check_tensormult((2, 3, 4), (5, 7, 4, 3), ((1, 2), (3, 2)))
    check_tensormult((2, 3, 4), (5, 7, 4, 3), ((2, 1), (2, 3)))
