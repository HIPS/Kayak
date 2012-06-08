from Differentiable import Differentiable

def el_sum(X, Y):
    return ElSumFunc(X, Y)

class ElSumFunc(Differentiable):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def value(self):
        return self.X.value() + self.Y.value()

    def gradient(self, other):
        return el_sum(self.X.gradient(other),
                      self.Y.gradient(other))
