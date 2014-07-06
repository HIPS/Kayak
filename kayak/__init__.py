import sys
import hashlib
import numpy as np

EPSILON = sys.float_info.epsilon

class Differentiable(object):

    def __init__(self):
        self._value = None
        self._grad  = {}

    def value(self, reset=False, rng=None):
        if reset or self._value is None:
            self._value = self.compute_value(reset, rng)
            self._grad  = {}
        return self._value

    def grad(self, other, outgrad=1.0):
        outgrad_hash = int(hashlib.sha1(np.atleast_1d(outgrad).view(np.uint8)).hexdigest(), 16)
        if not self._grad.has_key((other,outgrad_hash)):
            self._grad[(other,outgrad_hash)] = self.compute_grad(other, outgrad)
        return self._grad[(other,outgrad_hash)]

    def compute_value(self, other):
        raise Exception("Class 'Differentiable' is abstract.")

    def compute_grad(self, other):
        raise Exception("Class 'Differentiable' is abstract.")

    def local_grad(self, outgrad):
        raise Exception("Class 'Differentiable' is abstract.")

    def depends(self, other):
        raise Exception("Class 'Differentiable' is abstract.")

from constants      import Constant, Parameter
from batcher        import Batcher
from inputs         import Inputs
from targets        import Targets
from matrix_ops     import MatAdd, MatMult, MatSum, Transpose, Reshape
from elem_ops       import ElemAdd
from nonlinearities import SoftReLU, HardReLU, LogSoftMax
from losses         import L2Loss, LogMultinomialLoss
from dropout        import Dropout
from regularizers   import L2Norm, L1Norm, Horseshoe, NExp

