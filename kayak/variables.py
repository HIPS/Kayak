import numpy as np

from constants      import ones, zeros
from Differentiable import Differentiable

def variables(*shape):
    if isinstance(shape[0], (np.ndarray)):
        return VariablesFunc(value=shape[0])
    elif isinstance(shape[0], (list, tuple)):
        return VariablesFunc(shape=shape[0])
    else:
        return VariablesFunc(shape=shape)

class VariablesFunc(Differentiable):

    def __init__(self, value=None, shape=None):
        if value is not None:
            self.X = value
        elif shape is not None:
            self.X = np.zeros(shape)
        else:
            raise Exception("No size or value specified.")

    def value(self):
        return self.X

    def gradient(self, other):
        if self == other:
            return ones(self.X.shape)
        else:
            return zeros(self.X.shape)


        
