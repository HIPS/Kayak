# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import numpy as np

import util

from . import Differentiable

class Nonlinearity(Differentiable):

    def __init__(self, X):
        super(Nonlinearity, self).__init__([X])
        self.X = X

class SoftReLU(Nonlinearity):

    def __init__(self, X, scale=1.0):
        super(SoftReLU, self).__init__(X)
        self.scale  = scale

    def _compute_value(self):
        # Somewhat complicated to handle overflow.
        X            = self.X.value
        se           = np.seterr(over='ignore')
        exp_X        = np.exp(X / self.scale)
        result       = np.log(1.0 + np.exp( X/self.scale ))*self.scale
        over         = np.isinf(exp_X)
        result[over] = X[over]/self.scale
        return result

    def _local_grad(self, parent, d_out_d_self):
        return d_out_d_self/(1.0 + np.exp( - self.X.value/self.scale ))

class HardReLU(Nonlinearity):

    def __init__(self, X):
        super(HardReLU, self).__init__(X)

    def _compute_value(self):
        return np.maximum(self.X.value, 0.0)

    def _local_grad(self, parent, d_out_d_self):
        return d_out_d_self * (self.X.value > 0)

class TanH(Nonlinearity):

    def __init__(self, X):
        super(TanH, self).__init__(X)

    def _compute_value(self):
        return np.tanh(self.X.value)

    def _local_grad(self, parent, d_out_d_self):
        return 1.0 - np.tanh(self.X.value)**2

class Logistic(Nonlinearity):

    def __init__(self, X):
        super(Logistic, self).__init__(X)

    def _compute_value(self):
        return 1.0/(1.0 + np.exp(-self.X.value))

    def _local_grad(self, parent, d_out_d_self):
        y = 1.0/(1.0 + np.exp(-self.X.value))
        return y*(1.0-y)

class LogSoftMax(Nonlinearity):

    def __init__(self, X, axis=1):
        super(LogSoftMax, self).__init__(X)
        self.axis = axis

    def _compute_value(self):
        X = self.X.value
        return X - util.logsumexp(X, axis=self.axis)

    def _local_grad(self, parent, d_out_d_self):
        return d_out_d_self - (np.exp(self.value) * np.sum(d_out_d_self, axis=self.axis, keepdims=True))


class SoftMax(Nonlinearity):

    def __init__(self, X, axis=1):
        super(SoftMax, self).__init__(X)
        self.axis = axis

    def _compute_value(self):
        X = self.X.value
        return np.exp(X - util.logsumexp(X, axis=self.axis))

    def _local_grad(self, parent, d_out_d_self):
        oldgrad = d_out_d_self - (np.exp(self.value) * np.sum(d_out_d_self, axis=self.axis, keepdims=True))
        X = self.X.value
        return oldgrad * np.exp(np.exp(X - util.logsumexp(X, axis=self.axis)))
