import numpy        as np
import numpy.random as npr

import kayak

from . import *

def test_scalar_value():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):
        np_X = npr.randn()
        
        X   = kayak.Parameter(np_X)
        out = kayak.NExp(X)

        assert close_float(out.value, 1.0 - np.exp(-np.abs(np_X)))

def test_scalar_grad():
    npr.seed(2)

    for ii in xrange(NUM_TRIALS):
        while True:
            np_X = npr.randn()
            if np.abs(np_X) > 0.1:
                break
        
        X   = kayak.Parameter(np_X)
        out = kayak.NExp(X)

        assert kayak.util.checkgrad(X, out) < MAX_GRAD_DIFF

def test_scalar_value_2():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        np_X = npr.randn()
        wt   = np.exp(npr.randn())
        
        X   = kayak.Parameter(np_X)
        out = kayak.NExp(X, weight=wt)

        assert close_float(out.value, wt * (1.0 - np.exp(-np.abs(np_X))))

def test_scalar_grad_2():
    npr.seed(4)

    for ii in xrange(NUM_TRIALS):
        while True:
            np_X = npr.randn()
            if np.abs(np_X) > 0.1:
                break
        wt   = np.exp(npr.randn())
        
        X   = kayak.Parameter(np_X)
        out = kayak.NExp(X, weight=wt)

        assert kayak.util.checkgrad(X, out) < MAX_GRAD_DIFF

def test_vector_value():
    npr.seed(5)

    for ii in xrange(NUM_TRIALS):
        np_X = npr.randn(10,1)
        wt   = np.exp(npr.randn())
        
        X   = kayak.Parameter(np_X)
        out = kayak.NExp(X, weight=wt)

        assert close_float(out.value, wt * np.sum(1.0 - np.exp(-np.abs(np_X))))

def test_vector_grad():
    npr.seed(6)

    for ii in xrange(NUM_TRIALS):
        while True:
            np_X = npr.randn()
            if np.all(np.abs(np_X) > 0.1):
                break
        wt   = np.exp(npr.randn())
        
        X   = kayak.Parameter(np_X)
        out = kayak.NExp(X, weight=wt)

        assert kayak.util.checkgrad(X, out) < MAX_GRAD_DIFF

def test_matrix_value():
    npr.seed(7)

    for ii in xrange(NUM_TRIALS):
        np_X = npr.randn(10,20)
        wt   = np.exp(npr.randn())
        
        X   = kayak.Parameter(np_X)
        out = kayak.NExp(X, weight=wt)

        assert close_float(out.value, wt * np.sum(1.0 - np.exp(-np.abs(np_X))))

def test_matrix_grad():
    npr.seed(8)

    for ii in xrange(NUM_TRIALS):
        while True:
            np_X = npr.randn()
            if np.all(np.abs(np_X) > 0.1):
                break
        wt   = np.exp(npr.randn())
        
        X   = kayak.Parameter(np_X)
        out = kayak.NExp(X, weight=wt)

        assert kayak.util.checkgrad(X, out) < MAX_GRAD_DIFF

def test_tensor_value():
    npr.seed(9)

    for ii in xrange(NUM_TRIALS):
        np_X = npr.randn(10,20,5)
        wt   = np.exp(npr.randn())
        
        X   = kayak.Parameter(np_X)
        out = kayak.NExp(X, weight=wt)

        assert close_float(out.value, wt * np.sum(1.0 - np.exp(-np.abs(np_X))))

def test_tensor_grad():
    npr.seed(10)

    for ii in xrange(NUM_TRIALS):
        while True:
            np_X = npr.randn()
            if np.all(np.abs(np_X) > 0.1):
                break
        wt   = np.exp(npr.randn())
        
        X   = kayak.Parameter(np_X)
        out = kayak.NExp(X, weight=wt)

        assert kayak.util.checkgrad(X, out) < MAX_GRAD_DIFF

