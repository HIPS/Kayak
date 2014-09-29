# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

#import math as m

import numpy        as np
import numpy.random as npr

from . import Differentiable, EPSILON

class Dropout(Differentiable):

    def __init__(self, X, drop_prob=0.5, rng=None):
        super(Dropout, self).__init__([X])
        self.X         = X
        self.drop_prob = drop_prob
        self._mask     = None

        if rng is None:
            self.rng = npr.RandomState()
        elif type(rng) == int:
            self.rng = npr.RandomState(rng)
        else:
            self.rng = rng

    def _compute_value(self, rng=None, inputs=None):
        # Dropout overrides the usual value because it can change at each call
        # Calling _clear_value() wipes the cache of the descendants
        if inputs is None:
            # If someone gave us an RNG, use it and pass it on.
            # Otherwise, use the instance-specific RNG.
            local_rng = self.rng if rng is None else rng
            self._mask  = local_rng.rand(*self.X.shape(inputs, reset=False)) > self.drop_prob
            return (((1.0+EPSILON)/(1.0-self.drop_prob+EPSILON)) *
                    self._mask * self.X.value(False, rng, inputs))
        else:
            # Assume we're at test time if there are inputs.
            return self.X.value(False, rng, inputs)

    def _local_grad(self, parent, d_out_d_self):
        return d_out_d_self * self._mask * ((1.0 + EPSILON)/(1.0-self.drop_prob + EPSILON))

    def _compute_grad(self, other, d_out_d_self):
        if other == self.X:
            return self._local_grad(d_out_d_self)
        elif self.X.depends(other):
            return self.X.grad(other, self._local_grad(d_out_d_self))
        else:
            return np.zeros(self.X.shape())

    def _compute_shape(self, rng=None, inputs=None):
        return self.X.shape(rng, inputs)
