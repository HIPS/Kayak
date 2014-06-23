import numpy as np

import util

from . import Differentiable

class Nonlinearity(Differentiable):

    def __init__(self, X):
        self.X      = X
        self._value = None

    def grad(self, other, outgrad=1.0):
        if other == self.X:
            return self.local_grad(other, outgrad)
        elif self.X.depends(other):
            return self.X.grad(other, self.local_grad(other, outgrad))
        else:
            return np.zeros(other.shape())

    def depends(self, other):
        return self.X == other or self.X.depends(other)

    def shape(self):
        return self.X.shape()


class SoftReLU(Nonlinearity):

    def __init__(self, X, scale=1.0):
        super(SoftReLU, self).__init__(X)
        self.scale  = scale

    def value(self, reset=False, rng=None):
        if reset or self._value is None:
            self._value = np.log(1.0 + np.exp( self.X.value(reset, rng)/self.scale ))*self.scale
        return self._value

    def local_grad(self, other, outgrad):
        return outgrad/(1.0 + np.exp( - self.X.value()/self.scale ))


class HardReLU(Nonlinearity):

    def __init__(self, X):
        super(HardReLU, self).__init__(X)

    def value(self, reset=False, rng=None):
        if reset or self._value is None:
            self._value = np.maximum(self.X.value(reset, rng), 0.0)
        return self._value

    def local_grad(self, other, outgrad):
        return outgrad * (self.X.value() > 0)

class LogSoftMax(Nonlinearity):

    def __init__(self, X, axis=1):
        super(LogSoftMax, self).__init__(X)
        self.axis = axis

    def value(self, reset=False, rng=None):
        if reset or self._value is None:
            X = self.X.value(reset, rng)
            self._value = X - util.logsumexp(X, axis=self.axis)
        return self._value

    def local_grad(self, other, outgrad):
        return outgrad - (np.exp(self.value()) * np.expand_dims(np.sum(outgrad, axis=self.axis), axis=self.axis))
