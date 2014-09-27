# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import numpy as np

from .        import Differentiable
from util     import broadcast

class MatMult(Differentiable):

    def __init__(self, A, B, *args):
        # Recurse to handle lists of arguments.
        if len(args) > 0:
            B = MatMult(B, *args)

        super(MatMult, self).__init__([A, B])

        if A.shape[1] != B.shape[0]:
            raise Exception("Cannot multiply %s by %s matrices." % (A.shape, B.shape))
        self.A = A
        self.B = B

    def _compute_value(self):
        return np.dot(self.A.value, self.B.value)

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            # Numpy is really, really bad.  Have to handle length-1 vectors differently.
            if len(self.B.shape) == 1:
                return np.outer(d_out_d_self, self.B.value)
            else:
                return np.dot(d_out_d_self, self.B.value.T)
        elif parent == 1:
            # Oh Numpy, you suck so much.
            if len(self.A.shape) == 1:
                return np.outer(self.A.value, d_out_d_self)
            else:
                return np.dot(self.A.value.T, d_out_d_self)
        else:
            raise Exception("Not a parent of me")

class MatSum(Differentiable):
     
    def __init__(self, A, axis=None):
        super(MatSum, self).__init__([A])
        if axis is not None and type(axis) != int:
            raise Exception("Can only sum over one axis at a time.")
        self.A    = A
        self.axis = axis

    def _compute_value(self):
        # Handle a sum and reexpansion over one dimension.
        return np.sum(self.A.value, axis=self.axis, keepdims=True)

    def _local_grad(self, parent, d_out_d_self):
        return d_out_d_self * np.ones(self.A.shape)

class MatAdd(Differentiable):

    def __init__(self, A, B, *args):
        # Recurse to handle lists of arguments.
        if len(args) > 0:
            B = MatAdd(B, *args)
        super(MatAdd, self).__init__([A,B])
        if broadcast(A.shape, B.shape) is None:
            raise Exception("Matrices are not broadcastable: %s vs %s" % (A.shape, B.shape))
        self.A = A
        self.B = B

    def _compute_value(self):
        return self.A.value + self.B.value

    def axes_for_sum(self, mat_shape, d_out_d_self_shape):
        mat_shape = list(mat_shape)
        d_out_d_self_shape = list(d_out_d_self_shape)
        to_sum = []
        for dim, sz in enumerate(d_out_d_self_shape[::-1]):
            if len(mat_shape) == 0:
                to_sum.append(len(d_out_d_self_shape)-dim-1)
            elif mat_shape.pop() == 1:
                to_sum.append(len(d_out_d_self_shape)-dim-1)
        return tuple(to_sum[::-1])

    def _local_grad(self, parent, d_out_d_self):
        P = self._parents[parent]
        if np.atleast_1d(d_out_d_self).shape == P.shape:
            return d_out_d_self
        else:
            broadcast_axes = self.axes_for_sum(P.shape, d_out_d_self.shape)
            return np.sum(d_out_d_self, axis=broadcast_axes).reshape(P.shape)

class MatDet(Differentiable):
    pass

class MatLogDet(Differentiable):
    pass

class MatTrace(Differentiable):
    pass

class Transpose(Differentiable):

    def __init__(self, A, axes=None):
        super(Transpose, self).__init__([A])
        self.A    = A
        self.axes = axes

    def _compute_value(self):
        return np.transpose(self.A.value, axes=self.axes)

    def _local_grad(self, parent, d_out_d_self):
        if self.axes is None:
            return np.transpose(d_out_d_self)
        else:
            return np.transpose(d_out_d_self, axes=np.argsort(self.axes))

class Reshape(Differentiable):

    def __init__(self, A, new_shape):
        super(Reshape, self).__init__([A])
        self.A         = A
        self.new_shape = new_shape

    def _compute_value(self):
        return np.reshape(self.A.value, self.new_shape)

    def _local_grad(self, parent, d_out_d_self):
        return np.reshape(d_out_d_self, self.A.shape)

class Concatenate(Differentiable):

    def __init__(self, axis, A, B, *args):
        # Recurse to handle lists of arguments.
        if len(args) > 0:
            B = Concatenate(axis, B, *args)
        super(Concatenate, self).__init__([A, B])
        self.A = A
        self.B = B
        self.axis = axis

    def _compute_value(self):
        return np.concatenate((self.A.value,
                               self.B.value), axis=self.axis)

    def _local_grad(self, parent, d_out_d_self):
        local_grad_both = np.split(d_out_d_self, [self.A.shape[self.axis]], axis=self.axis)
        return local_grad_both[parent]

class TensorMult(Differentiable):
    pass
       
