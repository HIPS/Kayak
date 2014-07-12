# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import numpy as np

import util

from . import Differentiable

class Convolve1d(Differentiable):

    def __init__(self, A, B, axis=-1):
        super(Differentiable, self).__init__()
        self.A    = A
        self.B    = B
        self.axis = axis

    def compute_value(self, reset, rng, inputs):
        pass  # TODO

    def local_grad_A(self, outgrad):
        pass

    def local_grad_B(self, outgrad):
        pass

    def compute_grad(self, other, outgrad):
        gradient = np.zeros(other.shape())

        if other == self.A:
            gradient += self.local_grad_A(outgrad)
        elif self.A.depends(other):
            gradient += self.A.grad(other, self.local_grad_A(outgrad))

        if other == self.B:
            gradient += self.local_grad_B(outgrad)
        elif self.B.depends(other):
            gradient += self.B.grad(other, self.local_grad_B(outgrad))

        return gradient

    def depends(self, other):
        return self.A == other or self.B == other or self.A.depends(other) or self.B.depends(other)

    def shape(self):
        return self.A.shape()

