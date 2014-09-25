# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import numpy as np

from . import Differentiable

class Regularizer(Differentiable):

    def __init__(self, X, weight):
        super(Regularizer, self).__init__([X])
        self.X      = X
        self.weight = weight

    def shape(self, inputs=None):
        return tuple([1] * len(self.X.shape(inputs)))

class L2Norm(Regularizer):

    def __init__(self, X, weight=1.0):
        super(L2Norm, self).__init__(X, weight)

    def compute_value(self, rng, inputs):
        return self.weight * np.sum(self.X.value(rng, inputs)**2)

    def local_grad(self, parent, d_out_d_self):
        return self.weight * 2.0 * self.X.value() * d_out_d_self

class L1Norm(Regularizer):

    def __init__(self, X, weight=1.0):
        super(L1Norm, self).__init__(X, weight)

    def compute_value(self, rng, inputs):
        return self.weight * np.sum(np.abs(self.X.value(rng, inputs)))

    def local_grad(self, parent, d_out_d_self):
        return self.weight * np.sign(self.X.value()) * d_out_d_self

class Horseshoe(Regularizer):

    def __init__(self, X, weight=1.0):
        super(Horseshoe, self).__init__(X, weight)

    def compute_value(self, rng, inputs):
        return -self.weight * np.sum(np.log(np.log(1.0 + self.X.value(rng, inputs)**(-2))))

    def local_grad(self, parent, d_out_d_self):
        return -(self.weight * d_out_d_self * (1 / (np.log(1.0 + self.X.value()**(-2))))
                 * (1.0/(1 + self.X.value()**(-2))) * (-2*self.X.value()**(-3)))

class NExp(Regularizer):

    def __init__(self, X, weight=1.0):
        super(NExp, self).__init__(X, weight)

    def compute_value(self, rng, inputs):
        return self.weight * np.sum(1.0 - np.exp(-np.abs(self.X.value(rng, inputs))))

    def local_grad(self, parent, d_out_d_self):
        return self.weight * d_out_d_self * np.exp(-np.abs(self.X.value())) * np.sign(self.X.value())
