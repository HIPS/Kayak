# Author: Ryan P. Adams <rpa@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import numpy as np

class Targets(object):

    def __init__(self, Y, batcher=None):
        self.data    = np.atleast_1d(Y)
        self.batcher = batcher

    def value(self, reset=False, rng=None, inputs=None):
        if inputs is not None and inputs.has_key(self):
            return inputs[self]
        elif self.batcher is None:
            return self.data
        else:
            return self.data[self.batcher.indices(),...]

    def shape(self):
        if self.batcher is None:
            return self.data.shape
        else:
            return self.data[self.batcher.indices(),...].shape

    def grad(self, other):
        raise Exception("Not sensible to take gradient in terms of targets.")

    def depends(self, other):
        return False
