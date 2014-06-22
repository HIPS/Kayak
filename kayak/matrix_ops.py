import numpy as np

from .        import Differentiable
from elem_ops import ElemAdd

# Point this class to another one.
MatAdd = ElemAdd

class MatMult(Differentiable):

    def __init__(self, A, B):
        if A.shape()[1] != B.shape()[0]:
            raise Exception("Cannot multiply %s by %s matrices." % (A.shape(), B.shape()))

        self.A      = A
        self.B      = B
        self._value = None

    def value(self, reset=False, rng=None):
        if reset or self._value is None:
            self._value = np.dot( self.A.value(reset, rng=rng), self.B.value(reset, rng=rng) )
        return self._value

    def grad(self, other, outgrad):
        if other == self.A:
            return np.dot(outgrad, self.B.value().T)
        elif other == self.B:
            return np.dot(self.A.value().T, outgrad)
        else:

            dep_A = self.A.depends(other)
            dep_B = self.B.depends(other)

            if dep_A and dep_B:
                return (self.A.grad(other, np.dot(outgrad, self.B.value().T))
                        + self.B.grad(other, np.dot(self.A.value().T, outgrad)))
            elif dep_A:
                return self.A.grad(other, np.dot(outgrad, self.B.value().T))
            elif dep_B:
                return self.B.grad(other, np.dot(self.A.value().T, outgrad))
            else:
                return np.zeros(other.shape())

    def depends(self, other):
        return other == self.A or other == self.B or self.A.depends(other) or self.B.depends(other)

    def shape(self):
        return (self.A.shape()[0], self.B.shape()[1],)

class MatSum(Differentiable):
    
    def __init__(self, A, axis=None):
        if axis is not None and type(axis) != int:
            raise Exception("Can only sum over one axis at a time.")

        self.A      = A
        self.axis   = axis
        self._value = None

    def value(self, reset=False, rng=None):
        if reset or self._value is None:
            if self.axis is None:
                # Handle the sum over all elements.
                A_val = self.A.value(reset, rng=rng)
                self._value = np.sum(A_val).reshape([1] * len(A_val.shape))
            else:
                # Handle a sum and reexpansion over one dimension.
                self._value = np.expand_dims(np.sum(self.A.value(reset, rng=rng), axis=self.axis), axis=self.axis)
        return self._value

    def grad(self, other, outgrad=1.0):
        if other == self.A:
            return outgrad * np.ones(self.A.shape())
        elif self.A.depends(other):
            return self.A.grad(other, outgrad * np.ones(self.A.shape()))
        else:
            return np.zeros(other.shape())
    
    def shape(self):
        if self.axis is None:
            return tuple( [1] * len(self.A.shape()) )
        else:
            A_shape = list(self.A.shape())
            A_shape[self.axis] = 1
            return tuple(A_shape)

    def depends(self, other):
        return self.A == other or self.A.depends(other)
