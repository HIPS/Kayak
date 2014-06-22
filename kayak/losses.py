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

    def grad(self, other, out):
        if other == self.pred:
            return 2 * (self.pred.value() - self.targ.value())
        elif other == self.targ:
            return -2 * (self.pred.value() - self.targ.value())
        else:

            dep_pred = self.pred.depends(other)
            dep_targ = self.targ.depends(other)

            if dep_pred and dep_targ:
                return (self.pred.grad(other, 2*(self.pred.value() - self.targ.value()))
                        + self.targ.grad(other, -2*(self.pred.value() - self.targ.value())))

            elif dep_pred:
                return self.pred.grad(other, 2*(self.pred.value() - self.targ.value()))

            elif dep_targ:
                return self.targ.grad(other, -2*(self.pred.value() - self.targ.value()))

            else:
                return np.zeros(other.shape())
    
    def shape(self):
        return self.pred.shape()

    def depends(self, other):
        return self.pred == other or self.targ == other or self.pred.depends(other) or self.targ.depends(other)


                
            
        
            
        
