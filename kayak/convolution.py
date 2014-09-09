# Author: Ryan P. Adams <rpa@seas.harvard.edu>, Jasper Snoek <jsnoek@seas.harvard.edu>
# Copyright 2014, The President and Fellows of Harvard University

import numpy as np

import util

from . import Differentiable
import sys

class Convolve1d(Differentiable):

    def __init__(self, A, B, ncolors=1, axis=-1):
        super(Differentiable, self).__init__()
        self.A    = A
        self.B    = B
        self.ncolors = ncolors
        self.axis = axis

    def compute_value(self, reset, rng, inputs):
        A = self.A.value(reset=reset, rng=rng)
        B = self.B.value(reset=reset, rng=rng)
        filtersize = B.shape[0]/self.ncolors

        # Broadcast to get color channels
        A = np.reshape(A, (A.shape[0], self.ncolors, -1))
        D = A.shape[-1] - filtersize + 1
        output = np.zeros((A.shape[0], D, B.shape[1]))

        offset = 0
        for j in xrange(D):
            output[:,j,:] = np.dot(A[:, :,offset:offset+filtersize].reshape((A.shape[0],-1)), B)
            offset += 1

        return output.reshape((A.shape[0], D*B.shape[1]))

    def local_grad_A(self, outgrad):
        filtersize = self.B.shape()[0]/self.ncolors
        output     = np.zeros((self.A.shape()[0], self.A.shape()[1]))
        B          = self.B.value().squeeze()
        output = output.reshape((output.shape[0], -1, self.ncolors))
        outgrad = outgrad.reshape(outgrad.shape[0], -1, B.shape[-1])
        for i in xrange(outgrad.shape[1]):
            output[:,i:i+filtersize,:] += np.dot(outgrad[:,i,:], B.T).reshape((output.shape[0], filtersize, self.ncolors))
 
        return output.reshape((output.shape[0], -1))

    def local_grad_B(self, outgrad):
        filtersize = self.B.shape()[0]/self.ncolors
        output     = np.zeros((self.B.shape()[0], self.B.shape()[1]))
        A          = self.A.value()
        A          = np.reshape(A, (A.shape[0], self.ncolors, -1))
        filtersize = self.B.shape()[0]/self.ncolors
        D = A.shape[-1] - filtersize + 1
        outgrad    = np.reshape(outgrad, (outgrad.shape[0], -1, self.B.shape()[1]))
        offset = 0
        for j in xrange(outgrad.shape[1]):
            output += np.dot(A[:,:,offset:offset+filtersize].reshape((A.shape[0],-1)).T, outgrad[:,j,:])
            offset += 1
        return output

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

    def shape(self, inputs=None):
        filtersize = self.B.shape()[0]/self.ncolors
        D = self.A.shape(inputs)[-1]/self.ncolors - filtersize + 1
        return (self.A.shape(inputs)[0], D*self.B.shape()[1])


