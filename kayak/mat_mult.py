import numpy as np

from el_sum import el_sum

from Differentiable import Differentiable

def mat_mult(X, Y):
    return MatMultFunc(X, Y)

class MatMultFunc(Differentiable):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def value(self):
        return np.dot(self.X.value(), self.Y.value())

    def gradient(self, other):
        if self == other:
            return ones(self.shape)

        return el_sum( mat_mult(self.X, self.Y.gradient(other)),
                       mat_mult(self.X.gradient(other), self.Y) )
