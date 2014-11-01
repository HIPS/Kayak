# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

import numpy as np
from .        import Differentiable

class MatMult(Differentiable):
    __slots__ = ['A', 'B']
    def __init__(self, A, B, *args):
        # Recurse to handle lists of arguments.
        if len(args) > 0:
            B = MatMult(B, *args)
        super(MatMult, self).__init__((A, B))
        self.A = A
        self.B = B

    def _compute_value(self):
        if self.A.shape[1] != self.B.shape[0]:
            raise Exception("Cannot multiply %s by %s matrices." % (self.A.shape, self.B.shape))
        if len(self.A.shape) != 2 or len(self.B.shape) != 2:
            raise Exception("Inputs of shape %s and %s are not matrices" % (self.A.shape, self.B.shape))
        return np.dot(self.A.value, self.B.value)

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            return np.dot(d_out_d_self, self.B.value.T)
        elif parent == 1:
            return np.dot(self.A.value.T, d_out_d_self)
        else:
            raise Exception("Not a parent of me")

class MatSum(Differentiable):
    __slots__ = ['A', 'axis', 'keepdims']
    def __init__(self, A, axis=None, keepdims=True):
        super(MatSum, self).__init__((A,))
        if axis is not None and type(axis) != int:
            raise Exception("Can only sum over one axis at a time.")
        self.A    = A
        self.axis = axis
        self.keepdims = keepdims

    def _compute_value(self):
        return np.sum(self.A.value, axis=self.axis, keepdims=self.keepdims)

    def _local_grad(self, parent, d_out_d_self):
        # If self.keepdims == False then we need to
        # broadcast d_out_d_self along the summation axis
        if not self.keepdims and self.axis is not None:
            expanded_d_out_d_self = np.expand_dims(d_out_d_self, self.axis)
            return expanded_d_out_d_self * np.ones(self.A.shape)
        else:
            return d_out_d_self * np.ones(self.A.shape)

class MatAdd(Differentiable):
    __slots__ = []
    def __init__(self, *args):
        super(MatAdd, self).__init__(args)

    def _compute_value(self):
        return sum([p.value for p in self._parents])

    def _local_grad(self, parent, d_out_d_self):
        parent_shape = self._parents[parent].shape
        num_singletons = len(d_out_d_self.shape) - len(parent_shape)
        if num_singletons > 0:
            extra_singletons = tuple(range(num_singletons))
            result = np.sum(d_out_d_self, axis=extra_singletons, keepdims=False)
        else:
            result = d_out_d_self

        assert len(result.shape) == len(parent_shape)
        original_singletons = tuple(np.where(np.array(parent_shape) == 1)[0])
        return np.sum(result, axis=original_singletons, keepdims=True)

class MatElemMult(Differentiable):
    """
    Elementwise multiplication of two arrays of the same size.
    Note: This does not support broadcasting yet. Look at MatAdd for ideas.
    """
    __slots__ = ['A', 'B']
    def __init__(self, A, B, *args):
        # Recurse to handle lists of arguments.
        if len(args) > 0:
            B = MatElemMult(B, *args)

        super(MatElemMult, self).__init__((A,B))

        self.A = A
        self.B = B

    def _compute_value(self):
        return self.A.value * self.B.value

    def _local_grad(self, parent, d_out_d_self):
        """
        For element-wise multiplication d(A*B)/dA = d_out_d_self * B.
        However, to support  broadcasting, we need to sum over the broadcast dimensions.
        For  example, d(A*x)/dx, where A is a matrix and x is a scalar, is
        given by \sum_{d1} \ldots \sum_{dD} (d_out_d_self * A)[d1,...,dD]
        """
        parent_shape = self._parents[parent].shape
        other_parent = 1 if parent == 0 else 0
        other_parent_value = self._parents[other_parent].value

        # Compute how many dimensions was parent broadcast along
        num_singletons = len(d_out_d_self.shape) - len(parent_shape)
        if num_singletons > 0:
            extra_singletons = tuple(range(num_singletons))
            # Sum out the broadcast dimensions
            result = np.sum(d_out_d_self*other_parent_value, axis=extra_singletons, keepdims=False)
        else:
            result = d_out_d_self*other_parent_value

        # In mutliplying, we may have broadcast the parent.
        # Sum out those dimensions as well.
        assert len(result.shape) == len(parent_shape)
        original_singletons = tuple(np.where(np.array(parent_shape) == 1)[0])
        return np.sum(result, axis=original_singletons, keepdims=True)

class MatDet(Differentiable):
    pass

class MatLogDet(Differentiable):
    pass

class MatTrace(Differentiable):
    pass

class Transpose(Differentiable):
    __slots__ = ['A', 'axes']
    def __init__(self, A, axes=None):
        super(Transpose, self).__init__((A,))
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
    __slots__ = ['A', 'new_shape']

    def __init__(self, A, new_shape):
        super(Reshape, self).__init__((A,))
        self.A         = A
        self.new_shape = new_shape

    def _compute_value(self):
        return np.reshape(self.A.value, self.new_shape)

    def _local_grad(self, parent, d_out_d_self):
        return np.reshape(d_out_d_self, self.A.shape)

class Concatenate(Differentiable):
    __slots__ = ['axis']
    def __init__(self, axis, *args):
        super(Concatenate, self).__init__(args)
        self.axis = axis

    def _compute_value(self):
        return np.concatenate([p.value for p in self._parents], axis=self.axis)

    def _local_grad(self, parent_ix, d_out_d_self):
        # Return the gradient only w.r.t. the matrix indexed by parent.
        start_ix = sum([p.shape[self.axis] for p in self._parents[0:parent_ix]])
        end_ix = start_ix + self._parents[parent_ix].shape[self.axis]
        return index_along_axis(d_out_d_self, self.axis, start_ix, end_ix)

def index_along_axis(array, axis, start, end):
    """Return everything up to but not including end.

    For example:
    >>> index_along_axis(np.randn(10,20), 0, 10, 12).shape
    (2, 20)
    """
    full_slice = [slice(None),] * array.ndim
    full_slice[axis] = slice(start,end)
    return array[full_slice]

class TensorMult(Differentiable):
    __slots__ = ['axes']
    def __init__(self, A, B, axes):
        super(TensorMult, self).__init__((A, B))
        self.axes = axes

    def _compute_value(self):
        A = self._parents[0].value
        B = self._parents[1].value
        return np.tensordot(A, B, self.axes)

    def _local_grad(self, parent, d_out_d_self):
        diff = lambda A, B : [a for a in A if a not in B]
        rank = lambda L : list(np.argsort(np.argsort(L)))
        val = [p.value for p in self._parents]
        axes = self.axes
        n_axes = len(axes[0])
        ignore_dims = [diff(range(val[i].ndim), axes[i]) for i in (0, 1)]
        ignore_ndims = [len(x) for x in ignore_dims]
        output_dims = (range(ignore_ndims[0]),
                       range(ignore_ndims[0], ignore_ndims[0] + ignore_ndims[1]))
        X, Y = parent, 1 - parent
        wrong_order = np.tensordot(val[Y], d_out_d_self, (ignore_dims[Y], output_dims[Y]))
        permutation = [None] * val[X].ndim
        for final, cur in zip(list(axes[X]) + ignore_dims[X],
                              rank(axes[Y]) + range(n_axes, val[X].ndim)):
            permutation[final] = cur

        return np.transpose(wrong_order, permutation)

class Identity(Differentiable):
    __slots__ = []
    def __init__(self, A):
        super(Identity, self).__init__((A,))

    def _compute_value(self):
        return self._parents[0].value

    def _local_grad(self, parent_ix, d_out_d_self):
        return d_out_d_self
