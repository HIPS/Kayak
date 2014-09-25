# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import numpy as np

import util

from . import Differentiable

class Nonlinearity(Differentiable):

    def __init__(self, X):
        super(Nonlinearity, self).__init__([X])
        self.X = X

    def shape(self, inputs=None):
        return self.X.shape(inputs)

    def local_grad(self, parent, outgrad):
        assert parent is self.X
        return self._local_grad(outgrad)

class SoftReLU(Nonlinearity):

    def __init__(self, X, scale=1.0):
        super(SoftReLU, self).__init__(X)
        self.scale  = scale

    def compute_value(self, rng, inputs):
        # Somewhat complicated to handle overflow.
        X            = self.X.value(rng, inputs)
        se           = np.seterr(over='ignore')
        exp_X        = np.exp(X / self.scale)
        result       = np.log(1.0 + np.exp( X/self.scale ))*self.scale
        over         = np.isinf(exp_X)
        result[over] = X[over]/self.scale
        return result

    def _local_grad(self, outgrad):
        return outgrad/(1.0 + np.exp( - self.X.value()/self.scale ))

class HardReLU(Nonlinearity):

    def __init__(self, X):
        super(HardReLU, self).__init__(X)

    def compute_value(self, rng, inputs):
        return np.maximum(self.X.value(rng, inputs), 0.0)

    def _local_grad(self, outgrad):
        return outgrad * (self.X.value() > 0)

class TanH(Nonlinearity):

    def __init__(self, X):
        super(TanH, self).__init__(X)

    def compute_value(self, rng, inputs):
        return np.tanh(self.X.value(rng, inputs))

    def _local_grad(self, outgrad):
        return 1.0 - np.tanh(self.X.value())**2

class Logistic(Nonlinearity):

    def __init__(self, X):
        super(Logistic, self).__init__(X)

    def compute_value(self, rng, inputs):
        return 1.0/(1.0 + np.exp(-self.X.value(rng, inputs)))

    def _local_grad(self, outgrad):
        y = 1.0/(1.0 + np.exp(-self.X.value()))
        return y*(1.0-y)

class LogSoftMax(Nonlinearity):

    def __init__(self, X, axis=1):
        super(LogSoftMax, self).__init__(X)
        self.axis = axis

    def compute_value(self, rng, inputs):
        X = self.X.value(rng, inputs)
        return X - util.logsumexp(X, axis=self.axis)

    def _local_grad(self, outgrad):
        return outgrad - (np.exp(self.value()) * np.expand_dims(np.sum(outgrad, axis=self.axis), axis=self.axis))


class SoftMax(Nonlinearity):

    def __init__(self, X, axis=1):
        super(SoftMax, self).__init__(X)
        self.axis = axis

    def compute_value(self, rng, inputs):
        X = self.X.value(rng, inputs)
        return np.exp(X - util.logsumexp(X, axis=self.axis))

    def _local_grad(self, outgrad):
        oldgrad = outgrad - (np.exp(self.value()) * np.expand_dims(np.sum(outgrad, axis=self.axis), axis=self.axis))
        X = self.X.value(rng, inputs)
        return oldgrad * np.exp(np.exp(X - util.logsumexp(X, axis=self.axis)))
