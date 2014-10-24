# Author: Ryan P. Adams <rpa@seas.harvard.edu>, Jasper Snoek <jsnoek@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import numpy        as np
import numpy.random as npr

from . import Differentiable

class Hstack(Differentiable):
    __slots__ = ['A', 'B']

    def __init__(self, A, B):
        super(Hstack, self).__init__([A, B])

        self.A         = A
        self.B         = B

    def _compute_value(self):
        return np.hstack((self.A.value, self.B.value))

    def _local_grad(self, parent, d_out_d_self):
        if parent == 0:
            return d_out_d_self[:,:self.A.shape[1]]
        if parent == 1:
            return d_out_d_self[:,self.A.shape[1]:]
