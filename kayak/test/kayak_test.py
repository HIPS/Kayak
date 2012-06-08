import numpy        as np
import numpy.random as npr

import kayak

def test_1():

    N  = 10;
    Xc = npr.randn(N)
    X  = kayak.constants(Xc)
    Y  = kayak.mat_sum( X )

    print Y.gradient(X).value()


    assert 'b' == 'b'
