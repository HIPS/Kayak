# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import numpy as np

from . import Differentiable

class Loss(Differentiable):
    
    def __init__(self, predictions, targets):
        super(Loss, self).__init__()

        if predictions.shape() != targets.shape():
            raise Exception("Predictions and targets have different shapes: %s vs %s" % (predictions.shape(), targets.shape()))

        self.preds  = predictions
        self.targs  = targets

    def compute_grad(self, other, outgrad):
        if other == self.preds:
            return self.local_grad(outgrad)
        elif other == self.targs or self.targs.depends(other):
            raise Exception("Don't try to take the gradient in terms of the target.")
        elif self.preds.depends(other):
            return self.preds.grad(other, self.local_grad(outgrad))
        else:
            return np.zeros(other.shape())

    def shape(self, inputs=None):
        return self.preds.shape(inputs)

    def depends(self, other):
        return self.preds == other or self.preds.depends(other)

class L2Loss(Loss):

    def __init__(self, predictions, targets, axis=None):
        super(L2Loss, self).__init__(predictions, targets)
        self.axis = axis

    def compute_value(self, reset, rng, inputs):
        return np.atleast_1d(np.sum((self.preds.value(reset, rng, inputs) - self.targs.value(reset, rng, inputs))**2, axis=self.axis))

    def local_grad(self, outgrad):
        return 2 * (self.preds.value() - self.targs.value()) * outgrad

class LogMultinomialLoss(Loss):

    def __init__(self, predictions, targets, axis=1):
        # Predictions are log probabilities and targets are counts.
        super(LogMultinomialLoss, self).__init__(predictions, targets)
        self.axis = axis

    def compute_value(self, reset, rng, inputs):
        return -np.atleast_1d(np.sum( self.targs.value(reset, rng, inputs) * self.preds.value(reset, rng, inputs), axis=self.axis))

    def local_grad(self, outgrad):
        return -outgrad * self.targs.value()
