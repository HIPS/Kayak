import numpy as np

from Differentiable import Differentiable

def ones(shape):
    return constants(np.ones(shape))

def zeros(shape):
    return constants(np.zeros(shape))

def constants(X):
    return ConstantsFunc(X)

class ConstantsFunc(Differentiable):

    def __init__(self, X):
        self.X     = X
        self.shape = X.shape

    def value(self):
        return self.X

    def gradient(self, other, incoming=None):
        if incoming is None:
            incoming = kayak.ones(self.shape)

        if self == other:
            return incoming
        else:
            return kayak.zeros(self.shape)
        
