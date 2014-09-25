# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import numpy as np

from . import Differentiable

class Constant(Differentiable):

    def __init__(self, val):
        super(Constant, self).__init__()
        self._value = np.atleast_1d(val)

    def value(self, rng=None, inputs=None):
        if inputs is not None and inputs.has_key(self):
            return inputs[self]
        else:
            return self._value

    def grad(self, other):
        return np.zeros(other.shape())

    def depends(self, other):
        return self == other

    def _compute_shape(self, inputs=None):
        if inputs is not None and inputs.has_key(self):
            return inputs[self].shape
        else:
            return self._value.shape

class Parameter(Constant):

    def __init__(self, val):
        super(Parameter, self).__init__(val)

    def add(self, addend):
        new_value = self._value  + addend
        self._clear_value()
        self._value = new_value

