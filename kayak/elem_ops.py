import numpy as np

from . import Differentiable

def broadcast(shape1, shape2):
    if len(shape1) != len(shape2):
        # Return None for failure.
        return None
    else:
        shape = []
        for ii, dim1 in enumerate(shape1):
            dim2 = shape2[ii]
            
            if dim1 == dim2:
                shape.append(dim1)
            elif (dim1 is None or dim1 > 1) and dim2 == 1:
                shape.append(dim1)
            elif (dim2 is None or dim2 > 1) and dim1 == 1:
                shape.append(dim2)
            else:
                return None
        return tuple(shape)

class ElemAdd(Differentiable):

    def __init__(self, A, B, *args):

        # Recurse to handle lists of arguments.
        if len(args) > 0:
            B = ElemAdd(B, *args)
        
        if broadcast(A.shape(), B.shape()) is None:
            raise Exception("Matrices are not broadcastable: %s vs %s" % (A.shape(), B.shape()))

        self.A      = A
        self.B      = B
        self._value = None

    def value(self, reset=False, rng=None):
        if reset or self._value is None:
            self._value = self.A.value(reset, rng=rng) + self.B.value(reset, rng=rng)
        return self._value

    def grad(self, other, outgrad=None):
        if outgrad is None:
            outgrad = np.ones(broadcast(self.A.shape(), self.B.shape()))

        if other == self.A:
            broadcast_axes = tuple(np.nonzero(np.array(self.A.shape())==1)[0])
            return np.sum(outgrad, axis=broadcast_axes).reshape(self.A.shape())

        elif other == self.B:
            broadcast_axes = tuple(np.nonzero(np.array(self.B.shape())==1)[0])
            return np.sum(outgrad, axis=broadcast_axes).reshape(self.B.shape())

        else:

            dep_A = self.A.depends(other)
            dep_B = self.B.depends(other)

            if dep_A and dep_B:
                bcast_A = tuple(np.nonzero(np.array(self.A.shape())==1)[0])
                bcast_B = tuple(np.nonzero(np.array(self.B.shape())==1)[0])
                return (self.A.grad(other, np.sum(outgrad, axis=bcast_A).reshape(self.A.shape()))
                        + self.B.grad(other, np.sum(outgrad, axis=bcast_B).reshape(self.B.shape())))

            elif dep_A:
                bcast_A = tuple(np.nonzero(np.array(self.A.shape())==1)[0])
                return self.A.grad(other, np.sum(outgrad, axis=bcast_A).reshape(self.A.shape()))

            elif dep_B:
                bcast_B = tuple(np.nonzero(np.array(self.B.shape())==1)[0])
                return self.B.grad(other, np.sum(outgrad, axis=bcast_B).reshape(self.B.shape()))

            else:
                return np.zeros(other.shape())

    def depends(self, other):
        return other == self.A or other == self.B or self.A.depends(other) or self.B.depends(other)

    def shape(self):
        return broadcast(self.A.shape(), self.B.shape())
