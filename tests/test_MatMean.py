import numpy        as np
import numpy.random as npr

import kayak
from nose.tools import assert_less

from . import *

def test_scalar_value():
    npr.seed(1)

    for ii in xrange(NUM_TRIALS):
        npX = npr.randn()
        X = kayak.Parameter( npX )
        Y = kayak.MatMean(X)

        # Verify that a scalar is reproduced.
        assert close_float(Y.value, npX)

def test_scalar_grad():
    npr.seed(2)

    for ii in xrange(NUM_TRIALS):
        npX = npr.randn()
        X = kayak.Parameter( npX )
        Y = kayak.MatMean(X)

        # Verify that the gradient is one.
        Y.value
        assert Y.grad(X) == 1.0
        assert_less(kayak.util.checkgrad(X, Y), MAX_GRAD_DIFF)

def test_vector_value_1():
    npr.seed(3)

    for ii in xrange(NUM_TRIALS):
        npX = npr.randn(10,1)
        X = kayak.Parameter( npX )
        Y = kayak.MatMean(X)
        # Verify the sum.
        assert close_float(Y.value, np.mean(npX))

def test_vector_grad_1():
    npr.seed(4)

    for ii in xrange(NUM_TRIALS):
        npX = npr.randn(10,1)
        X = kayak.Parameter( npX )
        Y = kayak.MatMean(X)

        # Verify the gradient.
        Y.value
        assert Y.grad(X).shape == npX.shape
        assert np.all(close_float(Y.grad(X), 1.0/float(npX.size) * np.ones(npX.shape)))
        assert_less(kayak.util.checkgrad(X, Y), MAX_GRAD_DIFF)

def test_vector_value_2():
    npr.seed(5)

    for ii in xrange(NUM_TRIALS):
        npX = npr.randn(1,10)
        X = kayak.Parameter( npX )
        Y = kayak.MatMean(X)

        # Verify the sum.
        assert close_float(Y.value, np.mean(npX))

def test_vector_grad_2():
    npr.seed(6)

    for ii in xrange(NUM_TRIALS):
        npX = npr.randn(1,10)
        X = kayak.Parameter( npX )
        Y = kayak.MatMean(X)

        # Verify the gradient.
        Y.value
        assert Y.grad(X).shape == npX.shape
        assert np.all(close_float(Y.grad(X), 1.0/float(np.prod(npX.shape)) * np.ones(npX.shape)))
        assert_less(kayak.util.checkgrad(X, Y), MAX_GRAD_DIFF)

def test_matrix_value():
    npr.seed(7)

    for ii in xrange(NUM_TRIALS):
        npX = npr.randn(10,20)
        X   = kayak.Parameter( npX )
        Y   = kayak.MatMean(X)

        # Verify the value.
        assert close_float(Y.value, np.mean(npX))

def test_matrix_grad():
    npr.seed(8)

    for ii in xrange(NUM_TRIALS):
        npX = npr.randn(10,20)
        X   = kayak.Parameter( npX )
        Y   = kayak.MatMean(X)

        # Verify the value.
        Y.value
        assert Y.grad(X).shape == npX.shape
        assert np.all(close_float(Y.grad(X), 1.0/float(np.prod(npX.shape)) * np.ones(npX.shape)))
        assert_less(kayak.util.checkgrad(X, Y), MAX_GRAD_DIFF)

def test_nested_value_1():
    npr.seed(9)

    for ii in xrange(NUM_TRIALS):
        npX = npr.randn(10,20)
        X   = kayak.Parameter( npX )
        Y   = kayak.MatMean(X, axis=0)
        Z   = kayak.MatMean(Y)

        assert np.all(close_float(Y.value, np.mean(npX, axis=0)))
        assert close_float(Z.value, np.mean(npX))

def test_nested_grad_1():
    npr.seed(10)

    for ii in xrange(NUM_TRIALS):
        npX = npr.randn(10,20)
        X   = kayak.Parameter( npX )
        Y   = kayak.MatMean(X, axis=0)
        Z   = kayak.MatMean(Y)

        assert Z.grad(X).shape == npX.shape
        assert np.all(close_float(Z.grad(X),1.0/float(np.prod(npX.shape)) * np.ones(npX.shape)))
        assert_less(kayak.util.checkgrad(X, Z), MAX_GRAD_DIFF)

def test_nested_value_2():
    npr.seed(11)

    for ii in xrange(NUM_TRIALS):
        npX = npr.randn(10,20)
        X   = kayak.Parameter( npX )
        Y   = kayak.MatMean(X, axis=1)
        Z   = kayak.MatMean(Y)

        assert np.all(close_float(Y.value.ravel(), np.mean(npX, axis=1)))
        assert close_float(Z.value, np.mean(npX))

def test_nested_grad_2():
    npr.seed(12)

    for ii in xrange(NUM_TRIALS):
        npX = npr.randn(10,20)
        X   = kayak.Parameter( npX )
        Y   = kayak.MatMean(X, axis=1)
        Z   = kayak.MatMean(Y)

        assert Z.grad(X).shape == npX.shape
        assert np.all(close_float(Z.grad(X), 1.0/float(np.prod(npX.shape)) * np.ones(npX.shape)))
        assert_less(kayak.util.checkgrad(X, Z), MAX_GRAD_DIFF)

def test_tensor_value_1():
    npr.seed(13)

    for ii in xrange(NUM_TRIALS):
        npX = npr.randn(10,20,30)
        X   = kayak.Parameter( npX )
        Y   = kayak.MatMean(X)

        assert X.shape == npX.shape
        assert close_float(Y.value, np.mean(npX))

def test_tensor_value_2():
    npr.seed(14)

    for ii in xrange(NUM_TRIALS):
        npX = npr.randn(10,20,30)
        X   = kayak.Parameter( npX )
        Y   = kayak.MatMean(X, axis=2)
        
        assert np.all(close_float(Y.value, np.expand_dims(np.mean(npX, axis=2), axis=2)))

def test_tensor_value_3():
    npr.seed(15)

    for ii in xrange(NUM_TRIALS):
        npX = npr.randn(10,20,30)
        X   = kayak.Parameter( npX )
        Y   = kayak.MatMean(X, axis=1)
        
        assert np.all(close_float(Y.value, np.expand_dims(np.mean(npX, axis=1), axis=1)))

def test_tensor_value_4():
    npr.seed(16)

    for ii in xrange(NUM_TRIALS):
        npX = npr.randn(10,20,30)
        X   = kayak.Parameter( npX )
        Y   = kayak.MatMean(X, axis=0)
        
        assert np.all(close_float(Y.value, np.expand_dims(np.mean(npX, axis=0), axis=0)))

def test_keepdims_value_1():
    npr.seed(9)

    for ii in xrange(NUM_TRIALS):
        npX = npr.randn(10,20)
        X   = kayak.Parameter( npX )
        Y   = kayak.MatMean(X, axis=0, keepdims=False)
        Z   = kayak.MatMean(Y)

        assert Y.shape == np.mean(npX, axis=0, keepdims=False).shape
        assert np.all(close_float(Y.value, np.mean(npX, axis=0, keepdims=False)))
        assert close_float(Z.value, np.mean(npX))

def test_keepdims_grad_1():
    npr.seed(10)

    for ii in xrange(NUM_TRIALS):
        npX = npr.randn(10,20)
        X   = kayak.Parameter( npX )
        Y   = kayak.MatMean(X, axis=0, keepdims=False)
        Z   = kayak.MatMean(Y)

        assert Z.grad(X).shape == npX.shape
        assert np.all(close_float(Z.grad(X), 1.0/float(np.prod(npX.shape)) * np.ones(npX.shape)))
        assert_less(kayak.util.checkgrad(X, Z), MAX_GRAD_DIFF)

def test_keepdims_grad_2():
    npr.seed(10)

    for ii in xrange(NUM_TRIALS):
        npW = npr.randn(5,10,20)
        npX = npr.randn(5,10,20)
        W   = kayak.Parameter( npW )
        X   = kayak.Parameter( npX )
        Y   = W * X
        Z   = kayak.MatMean(Y, axis=2, keepdims=False)
        S   = kayak.MatMean(Z)

        assert S.grad(W).shape == npW.shape
        assert_less(kayak.util.checkgrad(X, S), MAX_GRAD_DIFF)