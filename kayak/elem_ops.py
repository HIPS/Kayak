# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import numpy as np
from . import Differentiable
import matrix_ops

class Elementwise(Differentiable):
    __slots__ = ['X']
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
    __slots__ = ['A']
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
    __slots__ = ['A']
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

class ElemPower(Elementwise):
    """
    Elementwise power of an array.

    NOTE: Fractional powers are only defined for positive bases.
          We do not check for this; numpy will throw a runtime exception.
    """
    __slots__ = ['A', 'pow']
    def __init__(self, A, pow):
        super(ElemPower, self).__init__([A])
        self.A = A
        assert np.isscalar(pow), 'Power must be a scalar value.'
        self.pow = pow

    def _compute_value(self):
        return np.power(self.A.value, self.pow)

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            return d_out_d_self * self.pow * np.power(self.A.value, self.pow-1)
        else:
            raise Exception("Not a parent of me")

class ElemAbs(Elementwise):
    """
    Elementwise absolute value of an array.
    """
    __slots__ = ['A']
    def __init__(self, A):
        super(ElemAbs, self).__init__([A])
        self.A = A

    def _compute_value(self):
        return abs(self.A.value)

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            return d_out_d_self * np.sign(self.A.value)
        else:
            raise Exception("Not a parent of me")
