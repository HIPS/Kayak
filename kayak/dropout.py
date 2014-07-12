# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import numpy        as np
import numpy.random as npr

from . import Differentiable, EPSILON

class Dropout(Differentiable):

    def __init__(self, X, drop_prob=0.5, rng=None):
        super(Dropout, self).__init__()
        self.X         = X
        self.drop_prob = drop_prob
        self._mask     = None

        if rng is None:
            self.rng = npr.RandomState()
        elif type(rng) == int:
            self.rng = npr.RandomState(rng)
        else:
            self.rng = rng

    def compute_value(self, reset, rng, inputs):
        # If someone gave us an RNG, use it and pass it on.
        # Otherwise, use the instance-specific RNG.
        local_rng = self.rng if rng is None else rng
        self._mask  = local_rng.rand(*self.X.shape()) > self.drop_prob

        return ((1.0+EPSILON)/(1.0-self.drop_prob+EPSILON)) * self._mask * self.X.value(reset, rng, inputs)

    def local_grad(self, outgrad):
        return outgrad * self._mask * ((1.0 + EPSILON)/(1.0-self.drop_prob + EPSILON))

    def compute_grad(self, other, outgrad):
        if other == self.X:
            return self.local_grad(outgrad)
        elif self.X.depends(other):
            return self.X.grad(other, self.local_grad(outgrad))
        else:
            return np.zeros(self.X.shape())

    def depends(self, other):
        return other == self.X or self.X.depends(other)

    def shape(self):
        return self.X.shape()
