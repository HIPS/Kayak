# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import numpy as np

from . import Differentiable
import matrix_ops

class Elementwise(Differentiable):

    def __init__(self, X):
        super(Elementwise, self).__init__(X)
        self.X = X

    def _compute_shape(self, inputs=None):
        return self.X.shape

# Just an alias for matrix addition and elementwise multiplication.
ElemAdd = matrix_ops.MatAdd
ElemMult = matrix_ops.MatElemMult


class ElemExp(Elementwise):
    """
    Elementwise exponentiation of an array
    """
    def __init__(self, A):
        super(ElemExp, self).__init__([A])

        self.A = A

    def _compute_value(self):
        return np.exp(self.A.value)

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            return d_out_d_self * np.exp(self.A.value)
        else:
            raise Exception("Not a parent of me")


class ElemLog(Elementwise):
    """
    Elementwise logarithm of an array
    """
    def __init__(self, A):
        super(ElemLog, self).__init__([A])

        self.A = A

    def _compute_value(self):
        return np.log(self.A.value)

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            return d_out_d_self / self.A.value
        else:
            raise Exception("Not a parent of me")
