import numpy        as np
import numpy.random as npr

from . import Differentiable

class Dropout(Differentiable):

    def __init__(self, A, drop_prob=0.5, rng=None):
        self.A         = A
        self.drop_prob = drop_prob
        self._value    = None
        self._mask     = None

        if rng is None:
            self.rng = npr.RandomState()
        elif type(rng) == int:
            self.rng = npr.RandomState(rng)
        else:
            self.rng = rng

    def value(self, reset=False, rng=None):
        if reset or self._value is None or self._mask is None:

            # If someone gave us an RNG, use it and pass it on.
            # Otherwise, use the instance-specific RNG.
            local_rng = self.rng if rng is None else rng
            self._mask  = local_rng.rand(*self.A.shape()) > self.drop_prob

            self._value = (1.0/(1.0-self.drop_prob)) * self._mask * self.A.value(reset, rng)
        return self._value

    def grad(self, other, outgrad=1.0):
        if other == self.A:
            return outgrad * self._mask * (1.0/(1.0-self.drop_prob))
        elif self.A.depends(other):
            return self.A.grad(other, outgrad * self._mask * (1.0/(1.0-self.drop_prob)))
        else:
            return np.zeros(self.A.shape())

    def depends(self, other):
        return other == self.A or self.A.depends(other)

    def shape(self):
        return self.A.shape()
