import numpy as np

from . import Differentiable

class L2Loss(Differentiable):

    def __init__(self, prediction, target):
        if prediction.shape() != target.shape():
            raise Exception("Predictions and targets have different shapes: %s vs %s" % (prediction.shape(), target.shape()))

        self.pred   = prediction
        self.targ   = target
        self._value = None

    def value(self, reset=False, rng=None):
        if reset or self._value is None:
            self._value = np.sum((self.pred.value(reset, rng=rng) - self.targ.value(reset, rng=rng))**2)
        return self._value

    def grad(self, other, outgrad=1.0):
        if other == self.pred:
            return 2 * (self.pred.value() - self.targ.value()) * outgrad
        elif other == self.targ:
            return -2 * (self.pred.value() - self.targ.value()) * outgrad
        else:

            dep_pred = self.pred.depends(other)
            dep_targ = self.targ.depends(other)

            if dep_pred and dep_targ:
                return (self.pred.grad(other, 2*(self.pred.value() - self.targ.value()) * outgrad)
                        + self.targ.grad(other, -2*(self.pred.value() - self.targ.value())) * outgrad)

            elif dep_pred:
                return self.pred.grad(other, 2*(self.pred.value() - self.targ.value()) * outgrad)

            elif dep_targ:
                return self.targ.grad(other, -2*(self.pred.value() - self.targ.value()) * outgrad)

            else:
                return np.zeros(other.shape())
    
    def shape(self):
        return self.pred.shape()

    def depends(self, other):
        return self.pred == other or self.targ == other or self.pred.depends(other) or self.targ.depends(other)

class LogMultinomial(Differentiable):

    def __init__(self, log_probs, counts, axis=1):
        if log_probs.shape() != counts.shape():
            raise Exception("Predictions and targets have different shapes: %s vs %s" % (log_probs.shape(), counts.shape()))

        self.log_probs = log_probs
        self.counts    = counts
        self.axis      = axis
        self._value    = None

    def value(self, reset=False, rng=None):
        if reset or self._value is None:
            self._value = np.expand_dims(np.sum( self.counts.value(reset,rng) * self.log_probs.value(reset,rng), axis=self.axis), axis=self.axis)
        return self._value

    def grad(self, other, outgrad=1.0):
        if other == self.log_probs:
            return outgrad * self.counts.value()
        elif other == self.counts:
            return outgrad * self.log_probs.value()
        else:

            dep_log_probs = self.log_probs.depend(other)
            dep_counts    = self.counts.depend(other)

            if dep_log_probs and dep_counts:
                return ( self.log_probs.grad(other, outgrad * self.counts.value())
                         + self.counts.grad(other, outgrad * self.log_probs.value()))
            
            elif dep_log_probs:
                return self.log_probs.grad(other, outgrad * self.counts.value())

            elif dep_counts:
                return self.counts.grad(other, outgrad * self.log_probs.value())

            else:
                return np.zeros(other.shape())

    def shape(self):
        prob_shape = list(self.log_probs.shape())
        prob_shape[self.axis] = 1
        return tuple(prob_shape)

    def depends(self, other):
        return self.log_probs == other or self.log_probs.depends(other) or self.counts == other or self.counts.depends(other)
                           
