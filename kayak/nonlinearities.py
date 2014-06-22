import numpy as np

from . import Differentiable

class SoftReLU(Differentiable):

    def __init__(self, A, scale=1.0):
        self.A      = A
        self.scale  = scale
        self._value = None

    def value(self, reset=False, rng=None):
        if reset or self._value is None:
            self._value = np.log(1.0 + np.exp( self.A.value(reset, rng=rng)/self.scale ))*self.scale
        return self._value

    def grad(self, other, outgrad=1.0):
        if other == self.A:
            return outgrad/(1.0 + np.exp( - self.A.value()/self.scale ))

        elif self.A.depends(other):
            return self.A.grad(other, outgrad/(1.0 + np.exp( - self.A.value()/self.scale )))

        else:
            return np.zeros(other.shape())

    def depends(self, other):
        return self.A == other or self.A.depends(other)

    def shape(self):
        return self.A.shape()

class HardReLU(Differentiable):

    def __init__(self, A):
        self.A      = A
        self._value = None

    def value(self, reset=False, rng=None):
        if reset or self._value is None:
            self._value = np.maximum(self.A.value(reset, rng=rng), 0.0)
        return self._value

    def grad(self, other, outgrad=1.0):
        if other == self.A:
            return outgrad * (self.A.value() > 0)

        elif self.A.depends(other):
            return self.A.grad(other, outgrad * (self.A.value() > 0))

        else:
            return np.zeros(other.shape())

    def depends(self, other):
        return self.A == other or self.A.depends(other)

    def shape(self):
        return self.A.shape()
