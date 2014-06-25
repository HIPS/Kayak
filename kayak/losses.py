import numpy as np

from . import Differentiable

class Loss(Differentiable):
    
    def __init__(self, predictions, targets):
        super(Loss, self).__init__()

        if predictions.shape() != targets.shape():
            raise Exception("Predictions and targets have different shapes: %s vs %s" % (predictions.shape(), target.shape()))

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

    def shape(self):
        return self.preds.shape()

    def depends(self, other):
        return self.preds == other or self.preds.depends(other)

class L2Loss(Loss):

    def __init__(self, predictions, targets, axis=None):
        super(L2Loss, self).__init__(predictions, targets)
        self.axis = axis

    def compute_value(self, reset, rng):
        return np.atleast_1d(np.sum((self.preds.value(reset, rng) - self.targs.value(reset, rng))**2, axis=self.axis))

    def local_grad(self, outgrad):
        return 2 * (self.preds.value() - self.targs.value()) * outgrad

class LogMultinomialLoss(Loss):

    def __init__(self, predictions, targets, axis=1):
        # Predictions are log probabilities and targets are counts.
        super(LogMultinomialLoss, self).__init__(predictions, targets)
        self.axis = axis

    def compute_value(self, reset, rng):
        return -np.atleast_1d(np.sum( self.targs.value(reset,rng) * self.preds.value(reset,rng), axis=self.axis))

    def local_grad(self, outgrad):
        return -outgrad * self.targs.value()
