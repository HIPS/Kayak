# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import numpy as np

from . import Differentiable

class Loss(Differentiable):
    
    def __init__(self, predictions, targets):
        super(Loss, self).__init__([predictions, targets])
        self.preds  = predictions
        self.targs  = targets

class L2Loss(Loss):

    def __init__(self, predictions, targets, axis=None):
        super(L2Loss, self).__init__(predictions, targets)
        self.axis = axis

    def _compute_value(self):
        return np.sum((self.preds.value - self.targs.value)**2,
                      axis=self.axis, keepdims=True)

    def _local_grad(self, parent, d_out_d_self):
        return 2 * (self.preds.value - self.targs.value) * d_out_d_self

class LogMultinomialLoss(Loss):

    def __init__(self, predictions, targets, axis=1):
        # Predictions are log probabilities and targets are counts.
        super(LogMultinomialLoss, self).__init__(predictions, targets)
        self.axis = axis

    def _compute_value(self):
        return -np.sum(self.targs.value * self.preds.value,
                       axis=self.axis, keepdims=True)

    def _local_grad(self, parent, d_out_d_self):
        return - d_out_d_self * self.targs.value
