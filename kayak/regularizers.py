import numpy as np

from . import Differentiable

class Regularizer(Differentiable):

    def __init__(self, X, weight):
        super(Regularizer, self).__init__()
        self.X = X
        self.weight = weight

    def compute_grad(self, other, outgrad):
        if other == self.X:
            return self.local_grad(outgrad)
        elif self.X.depends(other):
            return self.X.grad(other, self.local_grad(outgrad))
        else:
            return np.zeros(self.X.shape())

    def shape(self):
        return tuple([1] * len(self.X.shape()))

    def depends(self, other):
        return self.X == other or self.X.depends(other)

class L2Norm(Regularizer):

    def __init__(self, X, weight):
        super(L2Norm, self).__init__(X, weight)

    def compute_value(self, reset, rng):
        return self.weight * np.sum(self.X.value(reset, rng)**2)

    def local_grad(self, outgrad):
        return self.weight * 2.0 * self.X.value() * outgrad

class L1Norm(Regularizer):

    def __init__(self, X, weight):
        super(L1Norm, self).__init__(X, weight)

    def compute_value(self, reset, rng):
        return self.weight * np.sum(np.abs(self.X.value(reset, rng)))

    def local_grad(self, outgrad):
        return weight * np.sign(self.X.value()) * outgrad

class Horseshoe(Differentiable):

    def __init__(self, X, weight):
        super(Horseshoe, self).__init__(X, weight)

    def compute_value(self, reset, rng):
        return -self.weight * np.sum(np.log(np.log(1.0 + self.X.value(reset, rng)**(-2))))

    def local_grad(self, outgrad):
        return -(self.weight * outgrad * (1 / (np.log(1.0 + self.X.value()**(-2))))
                 * (1.0/(1 + self.X.value()**(-2))) * (-2*self.X.value()**(-3)))

class NExp(Differentiable):

    def __init__(self, A, weight):
        super(NExp, self).__init__(X, weight)

    def compute_value(self, reset, rng):
        return self.weight * np.sum(1.0 - np.exp(-np.abs(self.X.value(reset, rng))))

    def local_grad(self, outgrad):
        return self.weight * outgrad * np.exp(-np.abs(self.X.value())) * np.sign(self.X.value())
