import numpy as np

from . import Differentiable
import matrix_ops

class Elementwise(Differentiable):

    def __init__(self, X):
        super(Elementwise, self).__init__()
        self.X = X

    def shape(self):
        return self.X.shape()

    def depends(self, other):
        return self.X == other or self.X.depends(other)

# Just an alias for matrix addition.
ElemAdd = matrix_ops.MatAdd

class ElemExp(Elementwise):
    pass

class ElemLog(Elementwise):
    pass
