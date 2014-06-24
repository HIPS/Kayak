import numpy as np

from . import Differentiable

class Elementwise(Differentiable):

    def __init__(self, X):
        super(Elementwise, self).__init__()
        self.X = X

    def shape(self):
        return self.X.shape()

    def depends(self, other):
        return self.X == other or self.X.depends(other)

