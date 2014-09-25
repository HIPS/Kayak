# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import numpy as np

from . import Differentiable
import matrix_ops

class Elementwise(Differentiable):

    def __init__(self, X):
        super(Elementwise, self).__init__()
        self.X = X

    def shape(self, inputs=None):
        return self.X.shape(inputs)

# Just an alias for matrix addition.
ElemAdd = matrix_ops.MatAdd

class ElemExp(Elementwise):
    pass

class ElemLog(Elementwise):
    pass
