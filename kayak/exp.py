from Differentiable import Differentiable
from el_mult import el_mult

def exp(X):
    return ExpFunc(X)

class ExpFunc(Differentiable):

    def __init__(self, X):
        self.X = X

    def value(self):
        return self.X.value()

    def gradient(self, other, incoming=None):
        if incoming is None:
            incoming = kayak.ones(self.shape)

        # FIXME
        return el_mult(self.X.gradient(other), exp(self.X))
