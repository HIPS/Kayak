import numpy        as np
import numpy.random as npr

from .  import close_float
from .. import kayak

def test_scalar_value():
    npr.seed(1)

    for ii in xrange(100):
        np_pred = npr.randn()
        np_targ = npr.randn()
        
        pred = kayak.Parameter(np_pred)
        targ = kayak.Targets(np_targ)
        out  = kayak.L2Loss(pred, targ)

        # Verify that a scalar is reproduced.
        assert close_float(out.value(True), (np_pred-np_targ)**2)

def test_scalar_grad():
    npr.seed(2)

    for ii in xrange(100):
        np_pred = npr.randn()
        np_targ = npr.randn()
        
        pred = kayak.Parameter(np_pred)
        targ = kayak.Targets(np_targ)
        out  = kayak.L2Loss(pred, targ)

        assert close_float(out.grad(pred), 2*(np_pred-np_targ))
        assert kayak.util.checkgrad(pred, out) < 1e-6

def test_vector_value():
    npr.seed(3)

    for ii in xrange(100):
        np_pred = npr.randn(10,1)
        np_targ = npr.randn(10,1)
        
        pred = kayak.Parameter(np_pred)
        targ = kayak.Targets(np_targ)
        out  = kayak.L2Loss(pred, targ)

        assert close_float(out.value(True), np.sum((np_pred-np_targ)**2))

def test_vector_grad():
    npr.seed(4)

    for ii in xrange(100):
        np_pred = npr.randn(10,1)
        np_targ = npr.randn(10,1)
        
        pred = kayak.Parameter(np_pred)
        targ = kayak.Targets(np_targ)
        out  = kayak.L2Loss(pred, targ)

        assert np.all(close_float(out.grad(pred), 2*(np_pred-np_targ)))
        assert kayak.util.checkgrad(pred, out) < 1e-6

def test_matrix_value_1():
    npr.seed(5)

    for ii in xrange(100):
        np_pred = npr.randn(10,20)
        np_targ = npr.randn(10,20)
        
        pred = kayak.Parameter(np_pred)
        targ = kayak.Targets(np_targ)
        out  = kayak.L2Loss(pred, targ)

        print out.value(True), (np_pred-np_targ)**2
        assert close_float(out.value(True), np.sum((np_pred-np_targ)**2))

def test_matrix_grad():
    npr.seed(6)

    for ii in xrange(100):
        np_pred = npr.randn(10,20)
        np_targ = npr.randn(10,20)
        
        pred = kayak.Parameter(np_pred)
        targ = kayak.Targets(np_targ)
        out  = kayak.L2Loss(pred, targ)

        assert np.all(close_float(out.grad(pred), 2*(np_pred-np_targ)))
        assert kayak.util.checkgrad(pred, out) < 1e-6

def test_matrix_value_2():
    npr.seed(7)

    for ii in xrange(100):
        np_pred = npr.randn(10,20)
        np_targ = npr.randn(10,20)
        
        pred = kayak.Parameter(np_pred)
        targ = kayak.Targets(np_targ)
        out  = kayak.L2Loss(pred, targ, axis=0)

        print out.value(True), np.sum((np_pred-np_targ)**2, axis=0)
        assert np.all(close_float(out.value(True), np.sum((np_pred-np_targ)**2, axis=0)))

def test_matrix_value_3():
    npr.seed(8)

    for ii in xrange(100):
        np_pred = npr.randn(10,20)
        np_targ = npr.randn(10,20)
        
        pred = kayak.Parameter(np_pred)
        targ = kayak.Targets(np_targ)
        out  = kayak.L2Loss(pred, targ, axis=1)

        print out.value(True), np.sum((np_pred-np_targ)**2, axis=1)
        assert np.all(close_float(out.value(True), np.sum((np_pred-np_targ)**2, axis=1)))

