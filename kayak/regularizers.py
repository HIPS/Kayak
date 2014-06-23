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

class L1Norm(Differentiable):

    def __init__(self, A, scale):
        self.A      = A
        self.scale  = scale
        self._value = None

    def value(self, reset=False, rng=None):
        if reset or self._value is None:
            self._value = self.scale * np.sum(np.abs(self.A.value(reset, rng=rng)))
        return self._value

    def grad(self, other, outgrad):
        if other == self.A:
            return self.scale * np.sign(self.A.value()) * outgrad
        elif self.A.depends(other):
            return self.A.grad(other, self.scale * np.sign(self.A.value()) * outgrad)
        else:
            return np.zeros(self.A.shape())

    def shape(self):
        return tuple([1] * len(self.A.shape()))

    def depends(self, other):
        return self.A == other or self.A.depends(other)

class Horseshoe(Differentiable):

    def __init__(self, A, scale):
        self.A      = A
        self.scale  = scale
        self._value = None

    def value(self, reset=False, rng=None):
        if reset or self._value is None:
            self._value = -self.scale * np.sum(np.log(np.log(1.0 + self.A.value(reset, rng)**(-2))))
        return self._value

    def grad(self, other, outgrad):
        if other == self.A:
            return -(self.scale * outgrad * (1 / (np.log(1.0 + self.A.value()**(-2))))
                    * (1.0/(1 + self.A.value()**(-2))) * (-2*self.A.value()**(-3)))

        elif self.A.depends(other):
            return self.A.grad(other, -(self.scale * outgrad * (1 / (np.log(1.0 + self.A.value()**(-2))))
                                        * (1.0/(1 + self.A.value()**(-2))) * (-2*self.A.value()**(-3))))
        else:
            return np.zeros(self.A.shape())

    def shape(self):
        return tuple([1] * len(self.A.shape()))

    def depends(self, other):
        return self.A == other or self.A.depends(other)

class NExp(Differentiable):

    def __init__(self, A, scale):
        self.A      = A
        self.scale  = scale
        self._value = None

    def value(self, reset=False, rng=None):
        if reset or self._value is None:
            self._value = self.scale * np.sum(1.0 - np.exp(-np.abs(self.A.value(reset, rng))))
        return self._value

    def grad(self, other, outgrad):
        if other == self.A:
            return self.scale * outgrad * np.exp(-np.abs(self.A.value())) * np.sign(self.A.value())

        elif self.A.depends(other):
            return self.A.grad(other, self.scale * outgrad * np.exp(-np.abs(self.A.value())) * np.sign(self.A.value()))

        else:
            return np.zeros(self.A.shape())

    def shape(self):
        return tuple([1] * len(self.A.shape()))

    def depends(self, other):
        return self.A == other or self.A.depends(other)
