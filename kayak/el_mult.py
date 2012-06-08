from Differentiable import Differentiable

def el_mult(X, Y):
    return ElMultFunc(X, Y)

class ElMultFunc(Differentiable):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def value(self):
        return self.X.value() * self.Y.value()

    def gradient(self, other):
        return el_sum(el_mult(self.X, self.Y.gradient(other)),
                      el_mult(self.Y, self.X.gradient(other)))
