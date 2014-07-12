import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_convolve1d_1():
    npr.seed(1)

    A = npr.randn(20)
    B = npr.randn(7)
    for ii in xrange(A.shape[0]):
        for jj in xrange(B.shape[0]):
            pass # TODO
    
