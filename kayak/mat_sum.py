import kayak

from Differentiable import Differentiable

def mat_sum(X, dim=0):
    # Make this handle multiple dimensions.
    return MatSumFunc(X, dim)

class MatSumFunc(Differentiable):

    def __init__(self, X, dim):
        self.X          = X
        self.dim        = dim
        self.shape      = list(X.shape)
        self.shape[dim] = 1

    def depends(self, other):
        if self.X == other:
            return True
        else:
            return self.X.depends(other)

    def value(self):
        return np.sum(self.X.value(), self.dim).reshape(self.new_shape)

    def gradient(self, other, incoming=None):
        if incoming is None:
            incoming = kayak.ones(self.shape)

        local = kayak.ones(self.X.shape)
        
        return self.X.gradient(other, kayak.bc_mult(local, incoming))
