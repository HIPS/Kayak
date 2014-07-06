import numpy as np

import util

from . import Differentiable

class Nonlinearity(Differentiable):

    def __init__(self, X):
        super(Nonlinearity, self).__init__()
        self.X      = X

    def compute_grad(self, other, outgrad=1.0):
        if other == self.X:
            return self.local_grad(outgrad)
        elif self.X.depends(other):
            return self.X.grad(other, self.local_grad(outgrad))
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

    def compute_value(self, reset=False, rng=None):
        X = self.X.value(reset, rng)
        se = np.seterr(over='ignore')
        exp_X  = np.exp(X / self.scale)
        result = np.log(1.0 + np.exp( X/self.scale ))*self.scale
        over   = np.isinf(exp_X)
        result[over] = X[over]/self.scale
        return result

    def local_grad(self, outgrad):
        return outgrad/(1.0 + np.exp( - self.X.value()/self.scale ))


class HardReLU(Nonlinearity):

    def __init__(self, X):
        super(HardReLU, self).__init__(X)

    def compute_value(self, reset=False, rng=None):
        return np.maximum(self.X.value(reset, rng), 0.0)

    def local_grad(self, outgrad):
        return outgrad * (self.X.value() > 0)

class LogSoftMax(Nonlinearity):

    def __init__(self, X, axis=1):
        super(LogSoftMax, self).__init__(X)
        self.axis = axis

    def compute_value(self, reset=False, rng=None):
        X = self.X.value(reset, rng)
        return X - util.logsumexp(X, axis=self.axis)

    def local_grad(self, outgrad):
        return outgrad - (np.exp(self.value()) * np.expand_dims(np.sum(outgrad, axis=self.axis), axis=self.axis))
