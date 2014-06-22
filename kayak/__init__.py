
class Differentiable(object):

    def __init__(self):
        raise Exception("Class 'Differentiable' is abstract.")

    def value(self, reset=False):
        raise Exception("Class 'Differentiable' is abstract.")

    def grad(self, other):
        raise Exception("Class 'Differentiable' is abstract.")

    def depends(self, other):
        raise Exception("Class 'Differentiable' is abstract.")

from constants      import Constant, Parameter, Zeros
from batcher        import Batcher
from inputs         import Inputs
from targets        import Targets
from matrix_ops     import MatAdd, MatMult, MatSum
from elem_ops       import ElemAdd, ElemMult, Scale
from nonlinearities import ReLU, Logistic
from losses         import L2Loss
