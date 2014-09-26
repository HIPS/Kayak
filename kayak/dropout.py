# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

#import math as m

import numpy        as np
import numpy.random as npr

from . import Differentiable, EPSILON

class Dropout(Differentiable):

    def __init__(self, X, drop_prob=0.5):
        super(Dropout, self).__init__()
        self.X         = X
        self.drop_prob = drop_prob
        self._enhancement = (1.0 + EPSILON)/(1.0 - self.drop_prob+EPSILON)
        self.draw_new_mask()

    def draw_new_mask(self):
        self._mask = self._enhancement * (npr.rand(*self.X.shape) > self.drop_prob)
        self._clear_value_cache()

    def reinstate_units(self):
        self._mask = np.ones(self.X.shape)
        self._clear_value_cache()

    def _compute_value(self):
        return self._mask * self.X.value

    def _local_grad(self, parent, d_out_d_self):
        return d_out_d_self * self._mask
