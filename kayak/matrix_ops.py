import numpy as np

from .        import Differentiable
from util     import broadcast

class MatMult(Differentiable):

    def __init__(self, A, B):
        super(MatMult, self).__init__()

        if A.shape()[1] != B.shape()[0]:
            raise Exception("Cannot multiply %s by %s matrices." % (A.shape(), B.shape()))

        self.A = A
        self.B = B

    def compute_value(self, reset, rng):
        return np.dot( self.A.value(reset, rng=rng), self.B.value(reset, rng=rng) )

    def local_grad_A(self, outgrad):
        return np.dot(outgrad, self.B.value().T)

    def local_grad_B(self, outgrad):
        return np.dot(self.A.value().T, outgrad)

    def compute_grad(self, other, outgrad):
        if other == self.A:
            return self.local_grad_A(outgrad)
        elif other == self.B:
            return self.local_grad_B(outgrad)
        else:

            dep_A = self.A.depends(other)
            dep_B = self.B.depends(other)

            if dep_A and dep_B:
                return (self.A.grad(other, self.local_grad_A(outgrad))
                        + self.B.grad(other, self.local_grad_B(outgrad)))
            elif dep_A:
                return self.A.grad(other, self.local_grad_A(outgrad))
            elif dep_B:
                return self.B.grad(other, self.local_grad_B(outgrad))
            else:
                return np.zeros(other.shape())

    def depends(self, other):
        return other == self.A or other == self.B or self.A.depends(other) or self.B.depends(other)

    def shape(self):
        return (self.A.shape()[0], self.B.shape()[1],)

class MatSum(Differentiable):
     
    def __init__(self, A, axis=None):
        super(MatSum, self).__init__()

        if axis is not None and type(axis) != int:
            raise Exception("Can only sum over one axis at a time.")

        self.A      = A
        self.axis   = axis

    def compute_value(self, reset, rng):
        if self.axis is None:
            # Handle the sum over all elements.
            A_val = self.A.value(reset, rng=rng)
            return np.sum(A_val).reshape([1] * len(A_val.shape))
        else:
            # Handle a sum and reexpansion over one dimension.
            return np.expand_dims(np.sum(self.A.value(reset, rng=rng), axis=self.axis), axis=self.axis)

    def local_grad(self, outgrad):
        return outgrad * np.ones(self.A.shape())

    def compute_grad(self, other, outgrad):
        if other == self.A:
            return self.local_grad(outgrad)
        elif self.A.depends(other):
            return self.A.grad(other, self.local_grad(outgrad))
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

class MatAdd(Differentiable):

    def __init__(self, A, B, *args):
        super(MatAdd, self).__init__()

        # Recurse to handle lists of arguments.
        if len(args) > 0:
            B = ElemAdd(B, *args)
        
        if util.broadcast(A.shape(), B.shape()) is None:
            raise Exception("Matrices are not broadcastable: %s vs %s" % (A.shape(), B.shape()))

        self.A = A
        self.B = B

    def compute_value(self, reset, rng):
        return self.A.value(reset, rng) + self.B.value(reset, rng)

    def local_grad_A(self, outgrad):
        broadcast_axes = tuple(np.nonzero(np.array(self.A.shape())==1)[0])
        return np.sum(outgrad, axis=broadcast_axes).reshape(self.A.shape())

    def local_grad_B(self, outgrad):
        broadcast_axes = tuple(np.nonzero(np.array(self.B.shape())==1)[0])
        return np.sum(outgrad, axis=broadcast_axes).reshape(self.B.shape())

    def compute_grad(self, other, outgrad):
        if outgrad is None:
            outgrad = np.ones(util.broadcast(self.A.shape(), self.B.shape()))

        if other == self.A:
            return self.local_grad_A(outgrad)

        elif other == self.B:
            return self.local_grad_B(outgrad)

        else:

            dep_A = self.A.depends(other)
            dep_B = self.B.depends(other)

            if dep_A and dep_B:
                return (self.A.grad(other, self.local_grad_A(outgrad))
                        + self.B.grad(other, self.local_grad_B(outgrad)))

            elif dep_A:
                return self.A.grad(other, self.local_grad_A(outgrad))

            elif dep_B:
                return self.B.grad(other, self.local_grad_B(outgrad))

            else:
                return np.zeros(other.shape())

    def depends(self, other):
        return other == self.A or other == self.B or self.A.depends(other) or self.B.depends(other)

    def shape(self):
        return util.broadcast(self.A.shape(), self.B.shape())
