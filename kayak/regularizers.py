import numpy as np

from . import Differentiable

class L2Norm(Differentiable):

    def __init__(self, A, scale):
        self.A      = A
        self.scale  = scale
        self._value = None

    def value(self, reset=False, rng=None):
        if reset or self._value is None:
            self._value = self.scale * np.sum(self.A.value(reset, rng=rng)**2)
        return self._value

    def grad(self, other, outgrad):
        if other == self.A:
            return self.scale * 2.0 * self.A.value() * outgrad
        elif self.A.depends(other):
            return self.A.grad(other, self.scale * 2.0 * self.A.value() * outgrad)
        else:
            return np.zeros(self.A.shape())

    def shape(self):
        return tuple([1] * len(self.A.shape()))

    def depends(self, other):
        return self.A == other or self.A.depends(other)
