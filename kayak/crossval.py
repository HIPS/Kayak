# Authors: Harvard Intelligent Probabilistic Systems (HIPS) Group
#          http://hips.seas.harvard.edu
#          Ryan Adams, David Duvenaud, Scott Linderman,
#          Dougal Maclaurin, Jasper Snoek, and others
# Copyright 2014, The President and Fellows of Harvard University
# Distributed under an MIT license. See license.txt file.

import itertools
import numpy        as np
import numpy.random as npr

class Fold(object):
    
    def __init__(self, cv, train, valid):
        self._cv    = cv
        self._train = train
        self._valid = valid

    def train(self):
        if self._cv.targets is None:
            return self._cv.inputs[self._train,...]
        else:
            return self._cv.inputs[self._train,...], self._cv.targets[self._train,...]

    def valid(self):
        if self._cv.targets is None:
            return self._cv.inputs[self._valid,...]
        else:
            return self._cv.inputs[self._valid,...], self._cv.targets[self._valid,...]

class CrossValidator(object):

    def __init__(self, num_folds, inputs, targets=None, permute=True):
        
        if permute:
            # Make a copy of the data, with a random permutation.
            self.ordering = npr.permutation(inputs.shape[0])
            self.inputs   = inputs[self.ordering,...].copy()
            if targets is not None:
                self.targets = targets[self.ordering,...].copy()
            else:
                self.targets = None
        else:
            self.ordering = np.arange(inputs.shape[0], dtype=int)
            self.inputs   = inputs
            self.targets  = targets

        self.fold_idx  = 0
        self.num_folds = num_folds
        self.edges     = np.linspace(0, self.inputs.shape[0], self.num_folds+1).astype(int)
        self.indices   = []
        for ii in xrange(self.num_folds):
            self.indices.append( np.arange(self.edges[ii], self.edges[ii+1], dtype=int) )
        self.folds = []
        for ii in xrange(self.num_folds):
            self.folds.append(Fold(self,
                                   np.array(list(itertools.chain.from_iterable([self.indices[jj] for jj in range(0,ii)+range(ii+1,self.num_folds)])), dtype=int),
                                   np.array(self.indices[ii], dtype=int)))

    def __iter__(self):
        return self
        
    def next(self):
        try:
            result = self.folds[self.fold_idx]
            self.fold_idx += 1
            return result
        except IndexError:
            self.fold_idx = 0
            raise StopIteration
            
            

        
