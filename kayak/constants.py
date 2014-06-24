import numpy as np

from . import Differentiable

class Constant(Differentiable):

    def __init__(self, val):
        self._value = np.atleast_1d(val)

    def value(self, reset=False, rng=None):
        return self._value

    def grad(self, other):
        return np.zeros(other.shape())

    def depends(self, other):
        return self == other

    def shape(self):
        return self._value.shape

class Parameter(Constant):

    def __init__(self, val):
        super(Parameter, self).__init__(val)

    def add(self, addend):
        self._value += addend
