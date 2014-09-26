# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import numpy as np

from . import Differentiable

class Constant(Differentiable):

    def __init__(self, val):
        super(Constant, self).__init__()
        self._value = np.atleast_1d(val)

    def value(self, reset=True, rng=None, inputs=None):
        if inputs is not None and inputs.has_key(self):
            return inputs[self]
        else:
            return self._value

    def grad(self, other):
        return np.zeros(other.shape(reset=False))

    def depends(self, other):
        return self == other

    def _compute_shape(self, inputs=None):
        if inputs is not None and inputs.has_key(self):
            return inputs[self].shape
        else:
            return self._value.shape

    def _clear_cache(self):
        # Overriding base class method. Parameter never needs to clear its cache
        return

class Parameter(Constant):

    def __init__(self, val):
        super(Parameter, self).__init__(val)

    def set_value(self, new_value):
        self._value = new_value

    def add(self, addend):
        new_value = self._value  + addend
        self._value = new_value
