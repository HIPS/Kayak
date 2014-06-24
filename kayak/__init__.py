
class Differentiable(object):

    def __init__(self):
        self._value = None
        self._grad  = None

    def value(self, reset=False, rng=None):
        if reset or self._value is None:
            self._value = self.compute_value(reset, rng)
            self._grad  = None
        return self._value

    def grad(self, other, outgrad=1.0):
        if self._grad is None:
            self._grad = self.compute_grad(other, outgrad)
        return self._grad

    def compute_value(self, other):
        raise Exception("Class 'Differentiable' is abstract.")

    def compute_grad(self, other):
        raise Exception("Class 'Differentiable' is abstract.")

    def local_grad(self, other):
        raise Exception("Class 'Differentiable' is abstract.")

    def depends(self, other):
        raise Exception("Class 'Differentiable' is abstract.")

from constants      import Constant, Parameter
from batcher        import Batcher
from inputs         import Inputs
from targets        import Targets
from matrix_ops     import MatAdd, MatMult, MatSum
from nonlinearities import SoftReLU, HardReLU, LogSoftMax
from losses         import L2Loss, LogMultinomialLoss
from dropout        import Dropout
from regularizers   import L2Norm, L1Norm, Horseshoe, NExp

