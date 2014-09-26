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

        if A.shape(reset=False)[1] != B.shape()[0]:
            raise Exception("Cannot multiply %s by %s matrices." % (A.shape(reset=False), B.shape()))

        self.A, self.B = A, B

    def _compute_value(self, rng, inputs):
        return np.dot( self.A.value(False, rng, inputs), self.B.value(False, rng, inputs) )

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            # Numpy is really, really bad.  Have to handle length-1 vectors differently.
            if len(self.B.shape(reset=False)) == 1:
                return np.outer(d_out_d_self, self.B.value(False, ))
            else:
                return np.dot(d_out_d_self, self.B.value(False, ).T)
        elif parent == 1:
            # Oh Numpy, you suck so much.
            if len(self.A.shape(reset=False)) == 1:
                return np.outer(self.A.value(False, ), d_out_d_self)
            else:
                return np.dot(self.A.value(False, ).T, d_out_d_self)
        else:
            raise Exception("Not a parent of me")

    def _compute_shape(self, inputs=None):
        if len(self.B.shape(inputs, reset=False)) == 1:
            return (self.A.shape(inputs, reset=False)[0],)
        else:
            return (self.A.shape(inputs, reset=False)[0], self.B.shape(inputs)[1],)

class MatSum(Differentiable):
     
    def __init__(self, A, axis=None):
        super(MatSum, self).__init__([A])

        if axis is not None and type(axis) != int:
            raise Exception("Can only sum over one axis at a time.")

        self.A    = A
        self.axis = axis

    def _compute_value(self, rng, inputs):
        if self.axis is None:
            # Handle the sum over all elements.
            A_val = self.A.value(False, rng, inputs)
            return np.sum(A_val).reshape([1] * len(A_val.shape))
        else:
            # Handle a sum and reexpansion over one dimension.
            return np.expand_dims(np.sum(self.A.value(False, rng, inputs), axis=self.axis), axis=self.axis)

    def _local_grad(self, parent, d_out_d_self):
        return d_out_d_self * np.ones(self.A.shape(reset=False))

    def _compute_shape(self, inputs=None):
        if self.axis is None:
            return tuple( [1] * len(self.A.shape(inputs, reset=False)) )
        else:
            A_shape = list(self.A.shape(inputs, reset=False))
            A_shape[self.axis] = 1
            return tuple(A_shape)

    def depends(self, other):
        return self.A == other or self.A.depends(other)

class MatAdd(Differentiable):

    def __init__(self, A, B, *args):
        # Recurse to handle lists of arguments.
        if len(args) > 0:
            B = MatAdd(B, *args)

        super(MatAdd, self).__init__([A,B])

        if broadcast(A.shape(reset=False), B.shape()) is None:
            raise Exception("Matrices are not broadcastable: %s vs %s" % (A.shape(reset=False), B.shape()))

        self.A = A
        self.B = B

    def _compute_value(self, rng, inputs):
        return self.A.value(False, rng, inputs) + self.B.value(False, rng, inputs)

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
        if np.atleast_1d(d_out_d_self).shape == P.shape(reset=False):
            return d_out_d_self
        else:
            broadcast_axes = self.axes_for_sum(P.shape(reset=False), d_out_d_self.shape)
            return np.sum(d_out_d_self, axis=broadcast_axes).reshape(P.shape(reset=False))

    def _compute_shape(self, inputs=None):
        return broadcast(self.A.shape(inputs, reset=False), self.B.shape(inputs))

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

    def _compute_value(self, rng, inputs):
        return np.transpose(self.A.value(False, rng, inputs), axes=self.axes)

    def _local_grad(self, parent, d_out_d_self):
        if self.axes is None:
            return np.transpose(d_out_d_self)
        else:
            return np.transpose(d_out_d_self, axes=np.argsort(self.axes))

    def _compute_shape(self, inputs=None):
        if self.axes is None:
            return self.A.shape(inputs, reset=False)[::-1]
        else:
            shape = self.A.shape(inputs, reset=False)
            return tuple([shape[ii] for ii in self.axes])

class Reshape(Differentiable):

    def __init__(self, A, new_shape):
        super(Reshape, self).__init__([A])

        self.A         = A
        self.new_shape = new_shape

    def _compute_value(self, rng, inputs):
        return np.reshape(self.A.value(False, rng, inputs), self.new_shape)

    def _local_grad(self, parent, d_out_d_self):
        return np.reshape(d_out_d_self, self.A.shape(reset=False))

    def _compute_shape(self, inputs=None):
        return self.new_shape

class Concatenate(Differentiable):

    def __init__(self, axis, A, B, *args):
        # Recurse to handle lists of arguments.
        if len(args) > 0:
            B = Concatenate(axis, B, *args)

        super(Concatenate, self).__init__([A, B])

        self.A = A
        self.B = B
        self.axis = axis

    def _compute_value(self, rng, inputs):
        return np.concatenate((self.A.value(False, rng, inputs),
                               self.B.value(False, rng, inputs)), axis=self.axis)

    def _local_grad(self, parent, d_out_d_self):
        local_grad_both = np.split(d_out_d_self, [self.A.shape(reset=False)[self.axis]], axis=self.axis)
        return local_grad_both[parent]

    def _compute_shape(self, inputs=None):
        a_shape = list(self.A.shape(inputs, reset=False))
        b_shape = list(self.B.shape(inputs, reset=False))
        shape = a_shape
        shape[self.axis] = a_shape[self.axis] + b_shape[self.axis]
        return tuple(shape)

class TensorMult(Differentiable):
    pass
       
