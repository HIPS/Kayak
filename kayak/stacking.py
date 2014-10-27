# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

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
