import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_identity():
    npr.seed(1)
    np_A = npr.randn(6,7)
    A    = kayak.Parameter(np_A)
    B    = kayak.Identity(A)
    assert np.all(close_float(B.value, np_A))
    assert np.all(close_float(B.grad(A), np.ones((6,7))))
