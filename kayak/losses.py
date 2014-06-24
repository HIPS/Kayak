import numpy as np

from . import Differentiable

class Loss(Differentiable):
    
    def __init__(self, predictions, targets):
        if predictions.shape() != targets.shape():
            raise Exception("Predictions and targets have different shapes: %s vs %s" % (predictions.shape(), target.shape()))
        self.preds  = predictions
        self.targs  = targets
        self._value = None
        self._grad  = None

    def value(self, reset=False, rng=None):
        if reset or self._value is None:
            self._value = self.compute_value(reset, rng)
            self._grad  = None
        return self._value

    def grad(self, other, outgrad=1.0):
        if self._grad is None:
            self._grad = self.compute_grad(other, outgrad)
        return self._grad

    def shape(self):
        return self.preds.shape()

    def depends(self, other):
        return self.preds == other or self.preds.depends(other)

class L2Loss(Loss):

    def __init__(self, predictions, targets):
        super(L2Loss, self).__init__(predictions, targets)

    def compute_value(self, reset, rng):
        return np.sum((self.preds.value(reset, rng) - self.targs.value(reset, rng))**2)        

    def local_grad(self, other, outgrad):
        return 2 * (self.preds.value() - self.targs.value()) * outgrad

    def compute_grad(self, other, outgrad):
        if other == self.preds:
            return self.local_grad(other, outgrad)
        elif other == self.targs or self.targs.depends(other):
            raise Exception("Don't try to take the gradient in terms of the target.")
        elif self.preds.depends(other):
            return self.preds.grad(other, self.local_grad(other, outgrad))
        else:
            return np.zeros(other.shape())

class LogMultinomialLoss(Loss):

    def __init__(self, predictions, targets, axis=1):
        # Predictions are log probabilities and targets are counts.
        super(LogMultinomialLoss, self).__init__(predictions, targets)
        self.axis = axis

    def compute_value(self, reset, rng):
        return -np.expand_dims(np.sum( self.targs.value(reset,rng) * self.preds.value(reset,rng), axis=self.axis), axis=self.axis)

    def local_grad(self, other, outgrad):
        return -outgrad * self.targs.value()

    def grad(self, other, outgrad=1.0):
        if other == self.preds:
            return self.local_grad(other, outgrad)
        elif other == self.targs:
            raise Exception("Don't try to take the gradient in terms of the target.")
        elif self.preds.depends(other):
            return self.preds.grad(other, self.local_grad(other, outgrad))
        else:
            return np.zeros(other.shape())
