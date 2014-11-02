# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

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
        slice_list = [slice(None), ] * self.X.value.ndim
        slice_list[self._axis] = self._inds
        return self.X.value[slice_list]

    def _local_grad(self, parent, d_out_d_self):
        result = np.zeros(self.X.shape)
        slice_list = [slice(None), ] * result.ndim
        slice_list[self._axis] = self._inds
        result[slice_list] = d_out_d_self
        return result
