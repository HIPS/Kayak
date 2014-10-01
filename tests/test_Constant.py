import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_constant_scalar():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):

        np_X = npr.randn()    
        X    = kayak.Constant(np_X)
        
        assert close_float(X.value, np_X)

def test_constant_vector():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):

        np_X = npr.randn(10)    
        X    = kayak.Constant(np_X)
        
        assert np.all(close_float(X.value, np_X))

def test_constant_matrix():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):

        np_X = npr.randn(10,20)
        X    = kayak.Constant(np_X)
        
        assert np.all(close_float(X.value, np_X))

