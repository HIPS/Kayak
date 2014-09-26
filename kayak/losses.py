# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import numpy as np

from . import Differentiable

class Loss(Differentiable):
    
    def __init__(self, predictions, targets):
        super(Loss, self).__init__([predictions])

        if predictions.shape(reset=False) != targets.shape():
            raise Exception("Predictions and targets have different shapes: %s vs %s" % (predictions.shape(reset=False), targets.shape()))

        self.preds  = predictions
        self.targs  = targets

    def _compute_shape(self, inputs=None):
        return self.preds.shape(inputs, reset=False)

class L2Loss(Loss):

    def __init__(self, predictions, targets, axis=None):
        super(L2Loss, self).__init__(predictions, targets)
        self.axis = axis

    def _compute_value(self, rng, inputs):
        return np.atleast_1d(np.sum((self.preds.value(False, rng, inputs) - self.targs.value(rng, inputs))**2, axis=self.axis))

    def _local_grad(self, parent, d_out_d_self):
        return 2 * (self.preds.value(False) - self.targs.value()) * d_out_d_self

class LogMultinomialLoss(Loss):

    def __init__(self, predictions, targets, axis=1):
        # Predictions are log probabilities and targets are counts.
        super(LogMultinomialLoss, self).__init__(predictions, targets)
        self.axis = axis

    def _compute_value(self, rng, inputs):
        return -np.atleast_1d(np.sum( self.targs.value(False, rng, inputs) * self.preds.value(rng, inputs), axis=self.axis))

    def _local_grad(self, parent, d_out_d_self):
        return -d_out_d_self * self.targs.value(False)
