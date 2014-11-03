# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

import numpy as np

from input_checking import check_equal_ndims_for_broadcasting
from . import Differentiable

class Loss(Differentiable):
    __slots__ = ['preds', 'targs']
    def __init__(self, predictions, targets):
        super(Loss, self).__init__((predictions, targets))
        self.preds  = predictions
        self.targs  = targets

    _check_inputs = check_equal_ndims_for_broadcasting

class L2Loss(Loss):
    __slots__ = ['axis', 'keepdims']
    def __init__(self, predictions, targets, axis=None, keepdims=True):
        super(L2Loss, self).__init__(predictions, targets)
        self.axis = axis
        self.keepdims = keepdims

    def _compute_value(self):
        return np.sum((self.preds.value - self.targs.value)**2,
                      axis=self.axis, keepdims=self.keepdims)

    def _local_grad(self, parent, d_out_d_self):
        assert parent is 0, "Shouldn't be taking derivative wrt targets"
        return 2 * (self.preds.value - self.targs.value) * d_out_d_self

class LogMultinomialLoss(Loss):
    __slots__ = ['axis', 'keepdims']
    def __init__(self, predictions, targets, axis=1, keepdims=True):
        # Predictions are log probabilities and targets are counts.
        super(LogMultinomialLoss, self).__init__(predictions, targets)
        self.axis = axis
        self.keepdims = keepdims

    def _compute_value(self):
        return -np.sum(self.targs.value * self.preds.value,
                       axis=self.axis, keepdims=self.keepdims)

    def _local_grad(self, parent, d_out_d_self):
        return - d_out_d_self * self.targs.value
