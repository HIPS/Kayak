import numpy as np

from Differentiable import Differentiable

def ones(*shape):
    if isinstance(shape[0], (list, tuple)):
        if len(shape) > 1:
            raise Exception("Multiple shapes specified.")
        shape = shape[0]

    return constants(np.ones(shape))

def zeros(*shape):
    if isinstance(shape[0], (list, tuple)):
        if len(shape) > 1:
            raise Exception("Multiple shapes specified.")
        shape = shape[0]

    return constants(np.zeros(shape))

def constants(X):
    return ConstantsFunc(X)

class ConstantsFunc(Differentiable):

    def __init__(self, X):
        self.X     = X
        self.shape = X.shape

    def value(self):
        return self.X

    def gradient(self, other):
        return zeros(self.shape)
        
