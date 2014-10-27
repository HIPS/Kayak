# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

import numpy        as np
import numpy.random as npr

from . import Differentiable, EPSILON

class Dropout(Differentiable):
    __slots__ = ['X', 'drop_prob', '_rng', '_enhancement', '_mask']

    def __init__(self, X, drop_prob=0.5, rng=None, batcher=None):
        if batcher is not None:
            super(Dropout, self).__init__([X, batcher])
            batcher.add_dropout_node(self)
        else:
            super(Dropout, self).__init__([X])

        self.X         = X
        self.drop_prob = drop_prob

        if rng is None:
            self._rng = npr.RandomState()
        else:
            self._rng = rng

        self._enhancement = (1.0 + EPSILON)/(1.0 - self.drop_prob+EPSILON)
        self.draw_new_mask()

    def draw_new_mask(self):
        self._mask = self._enhancement * (self._rng.rand(*self.X.shape)
                                          > self.drop_prob)
        self._clear_value_cache()

    def reinstate_units(self):
        self._mask = np.ones(self.X.shape)
        self._clear_value_cache()

    def _compute_value(self):
        return self._mask * self.X.value

    def _local_grad(self, parent, d_out_d_self):
        return d_out_d_self * self._mask

