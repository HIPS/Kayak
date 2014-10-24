# Author: Ryan P. Adams <rpa@seas.harvard.edu>, Jasper Snoek <jsnoek@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

#import math as m

import numpy        as np
import numpy.random as npr

from . import Differentiable

class Take(Differentiable):
    __slots__ = ['X', '_inds', '_axis']

    def __init__(self, X, inds, axis=1):
        super(Take, self).__init__([X])

        self.X         = X
        self._inds     = inds
        self._axis     = axis

    def _compute_value(self):
        return np.take(self.X.value, self._inds, axis=self._axis)

    def _local_grad(self, parent, d_out_d_self):
        result = np.zeros(self.X.shape)
        if self._axis == 0: # Need a better way to slice by axis
            result[self._inds,:] = d_out_d_self
        elif self._axis == 1:
            result[:,self._inds] = d_out_d_self
        else:
            raise("Only up to two dimensional arrays supported.")

        return result