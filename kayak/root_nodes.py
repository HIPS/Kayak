# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

import numpy as np
from . import Differentiable

class DataNode(Differentiable):
    __slots__ = ['_batcher', '_data']
    def __init__(self, data, batcher=None):
        if batcher is None:
            super(DataNode, self).__init__([])
        else:
            super(DataNode, self).__init__([batcher])

        self._data    = np.atleast_1d(data)
        self._batcher = batcher

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = new_data
        self._clear_value_cache()

    def _compute_value(self):
        if self._batcher is None:
            return self.data
        else:
            return self.data[self._batcher.value,...]

    def _local_grad(self, parent, d_out_d_self):
        raise Exception("Can't take gradient w.r.t. data")

class Parameter(Differentiable):
    __slots__ = []
    def __init__(self, val):
        super(Parameter, self).__init__([])
        self.value = np.atleast_1d(val)

    def grad(self, other):
        return np.zeros(other.shape)

    def _compute_value(self):
        raise Exception("Shouldn't need this. Value should be cached")

    def _local_grad(self, parent, d_out_d_self):
        raise Exception("Shouldn't get here.")

# These are just aliases
Inputs = DataNode
Targets = DataNode
Constant = Parameter
